"""
YouTube RAG Chatbot — Production Gradio UI
==========================================
Powered by: LangChain · Gemini 2.5 Flash · FAISS · HuggingFace Embeddings
Author   : (your name here)
Deploy   : Hugging Face Spaces — set GOOGLE_API_KEY and HF_TOKEN as Space secrets
"""

from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
import gradio as gr

# ── LangChain ──────────────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# ── Environment ────────────────────────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

STARTER_PROMPTS = [
    "📝  Summarize the key takeaways of this video",
    "📌  List the main topics covered",
    "💡  What is the main argument or thesis?",
    "🗣️  What are the most important quotes or statements?",
    "❓  What questions does this video answer?",
]

YT_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?.*v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)

# ── Shared singletons (loaded once per process) ────────────────────────────
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


# ── Helpers ─────────────────────────────────────────────────────────────────


def parse_video_id(raw: str) -> str | None:
    """Accept a full YouTube URL or a bare 11-char video ID."""
    raw = raw.strip()
    m = YT_URL_PATTERN.search(raw)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        return raw
    return None


def cache_path(video_id: str) -> Path:
    return CACHE_DIR / video_id


def build_chain(retriever):
    """Construct the full conversational RAG chain from a retriever."""
    rewrite_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=60,
        max_retries=2,
    )
    answer_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_tokens=None,
        timeout=60,
        max_retries=2,
    )

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rephrase the latest user question into a standalone question using"
                " the chat history. Do not answer it.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions about YouTube videos."
                " Answer the question using ONLY the following video transcript context."
                " If the context does not contain enough information, say so honestly."
                "\n\nContext:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        rewrite_model, retriever, contextualize_prompt
    )
    qa_chain = create_stuff_documents_chain(answer_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain


# ── Core: load / index video ───────────────────────────────────────────────


def load_video(raw_input: str):
    """
    Fetches transcript, builds (or loads cached) FAISS index, returns
    (video_id, retriever, detected_language, transcript_snippet).
    Raises ValueError with a user-friendly message on any failure.
    """
    video_id = parse_video_id(raw_input)
    if not video_id:
        raise ValueError(
            "⚠️ Invalid input. Please enter a YouTube URL or an 11-character video ID."
        )

    if not GOOGLE_API_KEY:
        raise ValueError(
            "⚠️ GOOGLE_API_KEY is not set. Add it to your .env file or Space secrets."
        )

    cp = cache_path(video_id)
    embeddings = get_embeddings()

    # ── Load from disk cache if available ──────────────────────────────────
    if cp.exists():
        vectorstore = FAISS.load_local(
            str(cp), embeddings, allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": TOP_K}
        )
        return video_id, retriever, "cached", "(loaded from cache)"

    # ── Fetch transcript ───────────────────────────────────────────────────
    try:
        api = YouTubeTranscriptApi()
        try:
            transcript_list = api.fetch(video_id=video_id, languages=["en"])
            detected_lang = "English"
        except Exception:
            transcript_list = api.fetch(video_id=video_id, languages=["hi"])
            detected_lang = "Hindi"
    except TranscriptsDisabled:
        raise ValueError(
            "⚠️ Transcripts are disabled for this video. Try another video."
        )
    except Exception as e:
        raise ValueError(f"⚠️ Could not fetch transcript: {e}")

    transcript = " ".join(doc.text for doc in transcript_list)

    # ── Chunk & embed ──────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    documents = splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

    # ── Persist to disk ────────────────────────────────────────────────────
    vectorstore.save_local(str(cp))

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )
    snippet = transcript[:300].replace("\n", " ") + "…"
    return video_id, retriever, detected_lang, snippet


# ── Gradio event handlers ──────────────────────────────────────────────────


def handle_load_video(raw_input: str, state: dict):
    """
    Called when the user clicks "Load & Index Video".
    Returns: status_msg, embed_html, updated_state, cleared_history
    """
    if not raw_input.strip():
        yield (
            "⚠️ Please enter a YouTube URL or video ID.",
            gr.update(),
            state,
            [],
        )
        return

    yield (
        "⏳ Fetching transcript and building index… (first load may take ~30 s)",
        gr.update(),
        state,
        [],
    )

    try:
        video_id, retriever, lang, snippet = load_video(raw_input)
    except ValueError as e:
        yield str(e), gr.update(), state, []
        return

    # Store retriever + fresh session store in state
    new_state = {
        "video_id": video_id,
        "retriever": retriever,
        "chain": build_chain(retriever),  # built once per video
        "store": {},
    }

    source_note = (
        "✅ Loaded from cache" if lang == "cached" else f"✅ Indexed · Language: {lang}"
    )
    status = f"{source_note} · Video ID: `{video_id}`"

    embed_html = f"""
    <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px;">
      <iframe
        src="https://www.youtube.com/embed/{video_id}"
        style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen>
      </iframe>
    </div>
    """

    yield status, embed_html, new_state, []


def handle_chat(
    user_msg: str,
    history: list[dict],
    state: dict,
) -> Generator:
    """
    Streams the bot response token-by-token.
    history is in Gradio's messages format: [{"role": ..., "content": ...}, ...]
    """
    if not user_msg.strip():
        yield history
        return

    if not state.get("video_id"):
        history = history + [
            {"role": "user", "content": user_msg},
            {
                "role": "assistant",
                "content": "⚠️ Please load a video first using the left panel.",
            },
        ]
        yield history
        return

    # retriever  = state["retriever"]
    store = state["store"]
    session_id = state["video_id"]

    rag_chain = state["chain"]

    def get_session_history(sid: str):
        if sid not in store:
            store[sid] = InMemoryChatMessageHistory()
        return store[sid]

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    config = {"configurable": {"session_id": session_id}}

    # Append user message immediately
    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ""},
    ]
    yield history

    # Stream tokens
    partial = ""
    try:
        for chunk in conversational_rag.stream({"input": user_msg}, config=config):
            token = chunk.get("answer", "")
            if token:
                partial += token
                history[-1]["content"] = partial
                yield history
    except Exception as e:
        history[-1]["content"] = f"❌ Error: {e}"
        yield history


def handle_starter_prompt(prompt_text: str, history, state):
    """Route a starter prompt click into the chat handler."""
    # Strip the emoji prefix before sending
    clean = re.sub(r"^[^\w]+", "", prompt_text).strip()
    yield from handle_chat(clean, history, state)


def handle_clear(state: dict):
    """Clear chat history for the current video session."""
    if state.get("video_id") and "store" in state:
        state["store"].pop(state["video_id"], None)
    return [], state


# ── Custom CSS ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global ── */
.gradio-container { max-width: 1400px !important; margin: 0 auto; }

/* ── Header ── */
#app-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 8px;
    color: white;
    text-align: center;
}
#app-header h1 { font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; color: white; }
#app-header p  { font-size: 1rem; opacity: 0.88; margin: 0; color: white; }

/* ── Panels ── */
#left-panel, #right-panel {
    border-radius: 16px;
    padding: 4px;
}

/* ── Status box ── */
#status-box {
    font-size: 0.88rem;
    border-radius: 10px;
    padding: 10px 14px;
    background: #f8faff;
    border: 1px solid #e2e8f0;
    min-height: 44px;
}

/* ── Load button ── */
#load-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
}
#load-btn:hover { opacity: 0.9; }

/* ── Starter prompts ── */
.starter-btn button {
    background: #f0f4ff !important;
    border: 1.5px solid #c7d2fe !important;
    border-radius: 20px !important;
    color: #4338ca !important;
    font-size: 0.82rem !important;
    padding: 6px 14px !important;
    transition: all 0.2s;
    white-space: normal;
    text-align: left;
}
.starter-btn button:hover {
    background: #e0e7ff !important;
    border-color: #818cf8 !important;
}

/* ── Chatbot ── */
#chatbot { border-radius: 14px; }

/* ── Clear button ── */
#clear-btn {
    font-size: 0.82rem !important;
    color: #6b7280 !important;
    border-color: #d1d5db !important;
}
"""

# ── Build Gradio App ───────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="YouTube RAG Chatbot") as demo:

        # ── State ──────────────────────────────────────────────────────────
        app_state = gr.State({})  # holds video_id, retriever, store

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="app-header">
          <h1>🎬 YouTube RAG Chatbot</h1>
          <p>Paste any YouTube URL · Index the transcript · Chat with the video</p>
        </div>
        """)

        # ── Main layout ────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── Left Panel ─────────────────────────────────────────────────
            with gr.Column(scale=4, elem_id="left-panel"):
                gr.Markdown("### 📹 Video Context")

                url_input = gr.Textbox(
                    label="YouTube URL or Video ID",
                    placeholder="https://youtube.com/watch?v=... or dQw4w9WgXcQ",
                    lines=1,
                )

                load_btn = gr.Button(
                    "🚀 Load & Index Video",
                    variant="primary",
                    elem_id="load-btn",
                )

                status_box = gr.Markdown(
                    value="Enter a YouTube URL above and click **Load & Index Video** to begin.",
                    elem_id="status-box",
                )

                video_embed = gr.HTML(
                    value="",
                    label="Video Preview",
                )

                gr.Markdown("#### 💬 Suggested Starter Prompts")
                starter_btns = []
                for prompt in STARTER_PROMPTS:
                    btn = gr.Button(prompt, size="sm", elem_classes=["starter-btn"])
                    starter_btns.append(btn)

            # ── Right Panel ────────────────────────────────────────────────
            with gr.Column(scale=6, elem_id="right-panel"):
                gr.Markdown("### 🤖 Chat")

                chatbot = gr.Chatbot(
                    label="Conversation",
                    elem_id="chatbot",
                    height=520,
                    avatar_images=(
                        None,
                        "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png",
                    ),
                    buttons=["copy"],
                    allow_tags=False,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your question",
                        placeholder="Ask anything about the video…",
                        lines=2,
                        scale=5,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button(
                        "🗑️  Clear Chat",
                        variant="secondary",
                        size="sm",
                        elem_id="clear-btn",
                    )

        # ── Footer ─────────────────────────────────────────────────────────
        gr.Markdown(
            "<center style='opacity:0.5; font-size:0.8rem; margin-top:16px;'>"
            "Built with LangChain · Gemini 2.5 Flash · FAISS · Gradio"
            "</center>"
        )

        # ── Event wiring ───────────────────────────────────────────────────

        # Load video (button click or Enter in URL box)
        load_outputs = [status_box, video_embed, app_state, chatbot]

        load_btn.click(
            handle_load_video,
            inputs=[url_input, app_state],
            outputs=load_outputs,
        )
        url_input.submit(
            handle_load_video,
            inputs=[url_input, app_state],
            outputs=load_outputs,
        )

        # Send message (button or Enter)
        chat_inputs = [msg_input, chatbot, app_state]
        chat_outputs = [chatbot]

        send_fn = send_btn.click(
            handle_chat,
            inputs=chat_inputs,
            outputs=chat_outputs,
        ).then(
            lambda: "", outputs=msg_input
        )  # clear input after send

        msg_input.submit(
            handle_chat,
            inputs=chat_inputs,
            outputs=chat_outputs,
        ).then(lambda: "", outputs=msg_input)

        # Clear chat
        clear_btn.click(
            handle_clear,
            inputs=[app_state],
            outputs=[chatbot, app_state],
        )

        # Starter prompts → pipe into chat handler
        for btn in starter_btns:
            btn.click(
                handle_starter_prompt,
                inputs=[btn, chatbot, app_state],
                outputs=[chatbot],
            )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os as _os
    # HF Spaces sets SPACE_ID; use 0.0.0.0 there, 127.0.0.1 locally
    # so the printed URL is directly clickable in your terminal.
    _host = "0.0.0.0" if _os.getenv("SPACE_ID") else "127.0.0.1"
    app = build_app()
    app.launch(
        theme=THEME,
        css=CUSTOM_CSS,
        server_name=_host,
        server_port=7860,
        share=False,
        show_error=True,
        ssr_mode=False,   # disable SSR — prevents blank page on Windows
    )
