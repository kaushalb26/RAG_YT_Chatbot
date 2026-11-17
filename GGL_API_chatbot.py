from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import (
    PromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

video_id = "Gfr50f6ZBvo"
try:
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id=video_id, languages=["en", "hi"])
except TranscriptsDisabled:
    print("Sorry for Inconvinience!, Transcript is not available for this video.")

transcript = " ".join(doc.text for doc in transcript_list)
print("****=======================================*****")
print("Transcript : ", transcript)
print("****=======================================*****")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", max_tokens=None, timeout=None, max_retries=2
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


documents = splitter.create_documents([transcript])

# embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2")
# -------------------------------------------------------

embeddings2 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings2)

# vectorstore.index_to_docstore_id(whatever ids text you want to see - enter Id )


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# retrieved_docs = retriever.invoke(query)
# context = " ".join(doc.page_content for doc in retrieved_docs)

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Rephrase the latest user question into a standalone question using"
                " the chat history. Do not answer it."
            ),
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_prompt
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer the question using only this video transcript"
                " context:\n\n{context}"
            ),
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
qa_chain = create_stuff_documents_chain(model, qa_prompt)

# 4. Combine Retriever + QA Chain, then wrap with Memory
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- Usage ---
config = {"configurable": {"session_id": "video_rag_1"}}

# First question
res1 = conversational_rag.invoke(
    {"input": "What topics are discussed in this video?"}, config=config
)
print(res1["answer"])

# Follow-up question (rewrites "it" -> "photosynthesis" before searching the vector DB)
res2 = conversational_rag.invoke(
    {"input": "What was my previous question?"}, config=config
)
print(res2["answer"])
