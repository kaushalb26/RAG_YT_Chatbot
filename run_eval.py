import os
import re
import json
import time
import hashlib
import statistics
from collections import defaultdict
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from openai import OpenAI
from pydantic import BaseModel
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models import DeepEvalBaseLLM

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "3600"
load_dotenv()

# ---------------------------------------------------------------------------
# Config (locked models for valid baseline vs candidate comparison)
# ---------------------------------------------------------------------------
CHATBOT_MODEL = "gemini-3.1-flash-lite"
JUDGE_MODEL = "qwen/qwen3.8-27b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MAX_RETRIES = 12
REPORT_FILE = "candidate.json"

CONFIG_HASH = hashlib.md5(
    f"{CHATBOT_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{TOP_K}".encode()
).hexdigest()[:8]

# 1. Load Golden Dataset
with open("golden_dataset.json", "r", encoding="utf-8") as f:
    golden_data = json.load(f)

# 2. Initialize Chatbot Models
rewrite_model = ChatGoogleGenerativeAI(model=CHATBOT_MODEL, temperature=0, max_retries=5)
answer_model = ChatGoogleGenerativeAI(model=CHATBOT_MODEL, temperature=0.2, max_retries=5)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
rag_chains_by_video = {}


def get_rag_chain(video_id: str):
    if video_id in rag_chains_by_video:
        return rag_chains_by_video[video_id]

    print(f"Building FAISS index for video: {video_id}...")
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id=video_id, languages=["en", "hi"])
    transcript = " ".join(doc.text for doc in transcript_list)

    # Loaded lazily so cached runs consume zero PyTorch/Windows virtual memory
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    documents = splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rephrase the latest user question into a standalone question using the chat history. Do not answer it.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        rewrite_model, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only this video transcript context:\n\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(answer_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    rag_chains_by_video[video_id] = rag_chain
    return rag_chain


# 3. Phase 1: Run RAG Pipeline (with local disk cache)
CACHE_FILE = "rag_outputs_cache.json"
cached_outputs = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cached_outputs = json.load(f)

test_cases = []
latencies = []

print("\n--- Phase 1: Running YT_RAG_Chatbot ---")
for idx, item in enumerate(golden_data):
    key = f"{CONFIG_HASH}::{item['video_id']}::{item['input']}"

    if key in cached_outputs:
        entry = cached_outputs[key]
        actual_output = entry["actual_output"]
        retrieved_chunks = entry["retrieval_context"]
        elapsed = entry["latency_sec"]
        print(f"[{idx + 1}/{len(golden_data)}] Loaded from cache ({elapsed:.2f}s)")
    else:
        chain = get_rag_chain(item["video_id"])
        for attempt in range(8):
            try:
                start_t = time.perf_counter()
                res = chain.invoke({"input": item["input"], "chat_history": []})
                elapsed = time.perf_counter() - start_t
                break
            except Exception as e:
                if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 7:
                    wait_sec = 20 + (attempt * 10)
                    print(
                        f"503 high demand in Phase 1 — waiting {wait_sec}s (retry {attempt + 1}/8)..."
                    )
                    time.sleep(wait_sec)
                else:
                    raise

        actual_output = res["answer"]
        retrieved_chunks = [doc.page_content for doc in res["context"]]

        cached_outputs[key] = {
            "actual_output": actual_output,
            "retrieval_context": retrieved_chunks,
            "latency_sec": elapsed,
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cached_outputs, f, indent=2, ensure_ascii=False)

        print(
            f"[{idx + 1}/{len(golden_data)}] ({item['video_id']}) Answered in {elapsed:.2f}s"
        )
        time.sleep(8)

    latencies.append(elapsed)
    test_cases.append(
        LLMTestCase(
            input=item["input"],
            actual_output=actual_output,
            expected_output=item["expected_output"],
            retrieval_context=retrieved_chunks,
        )
    )


# 4. Token-Optimized Groq Judge Wrapper (Qwen3.8 27B)
def _extract_json(text: str) -> str:
    # Strip any <think>...</think> blocks if present, plus markdown fences
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


class CustomGroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment / .env file.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel = None):
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(5)  # Safe pacing for Groq's 8,000 TPM free ceiling
                messages = [{"role": "user", "content": prompt}]
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0,
                    "reasoning_effort": "none",
                }

                if schema is not None:
                    messages[0]["content"] = (
                        f"{prompt}\n\nRespond ONLY with a valid JSON object matching "
                        f"this JSON schema, with no extra text or markdown:\n"
                        f"{json.dumps(schema.model_json_schema())}"
                    )
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                if schema is not None:
                    return schema.model_validate_json(_extract_json(text))
                return text

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    if "tokens per day" in err_str.lower() or "tpd" in err_str.lower():
                        raise RuntimeError(
                            f"Daily token quota (TPD) reached for {self.model_name}."
                        ) from e
                    wait_sec = 25 + (attempt * 15)
                    print(
                        f"429 Groq TPM/RPM limit — sleeping {wait_sec}s before retry {attempt + 1}/{MAX_RETRIES}..."
                    )
                    time.sleep(wait_sec)
                elif (
                    "503" in err_str
                    or "validation" in err_str.lower()
                    or "json" in err_str.lower()
                ):
                    print(
                        f"Transient/JSON error from {self.model_name} — retry {attempt + 1}/{MAX_RETRIES}..."
                    )
                    time.sleep(3)
                else:
                    raise

        raise RuntimeError(f"Judge failed after {MAX_RETRIES} attempts.")

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


judge_llm = CustomGroqJudge(model_name=JUDGE_MODEL)

# include_reason=False cuts 4 extra LLM calls per question (~40% fewer tokens)
metrics = [
    FaithfulnessMetric(threshold=0.7, model=judge_llm, include_reason=False),
    AnswerRelevancyMetric(threshold=0.7, model=judge_llm, include_reason=False),
    ContextualPrecisionMetric(threshold=0.7, model=judge_llm, include_reason=False),
    ContextualRecallMetric(threshold=0.7, model=judge_llm, include_reason=False),
]

# 5. Grade One Case at a Time & Checkpoint After Every Case
detailed_cases = []
if os.path.exists(REPORT_FILE):
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        detailed_cases = json.load(f).get("test_cases", [])

completed_indices = {c["index"] for c in detailed_cases}
print(
    f"\n--- Phase 2: Grading with DeepEval ({len(completed_indices)}/{len(golden_data)} already saved) ---"
)

for idx, tc in enumerate(test_cases):
    case_num = idx + 1
    if case_num in completed_indices:
        print(f"[{case_num}/{len(test_cases)}] Already saved in {REPORT_FILE} — skipping.")
        continue

    res = evaluate(
        test_cases=[tc],
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )
    test_result = res.test_results[0]
    gd = golden_data[idx]

    detailed_cases.append(
        {
            "index": case_num,
            "video_id": gd.get("video_id"),
            "question_type": gd.get("question_type"),
            "input": test_result.input,
            "actual_output": test_result.actual_output,
            "expected_output": test_result.expected_output,
            "latency_sec": round(latencies[idx], 3),
            "metrics": {
                m.name: {"score": m.score, "passed": m.success}
                for m in (test_result.metrics_data or [])
            },
        }
    )
    detailed_cases.sort(key=lambda x: x["index"])

    metric_scores = defaultdict(list)
    for c in detailed_cases:
        for m_name, m_val in c["metrics"].items():
            if m_val["score"] is not None:
                metric_scores[m_name].append(m_val["score"])

    sorted_lat = sorted(latencies)
    p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))

    baseline_summary = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "search_type": "similarity",
            "k": TOP_K,
            "chatbot_model": CHATBOT_MODEL,
            "judge_model": judge_llm.get_model_name(),
        },
        "average_scores": {
            name: round(statistics.mean(scores), 4)
            for name, scores in metric_scores.items()
            if scores
        },
        "operational_metrics": {
            "p50_latency_sec": round(statistics.median(latencies), 3),
            "p95_latency_sec": round(sorted_lat[p95_idx], 3),
        },
        "test_cases": detailed_cases,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(baseline_summary, f, indent=2, ensure_ascii=False)

    print(f"[{case_num}/{len(test_cases)}] Saved checkpoint to {REPORT_FILE}")

print(f"\n=== BASELINE SUMMARY SAVED TO {REPORT_FILE} ===")
print(json.dumps(baseline_summary["average_scores"], indent=2))
print("Latency:", baseline_summary["operational_metrics"])