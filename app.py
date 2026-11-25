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

video_id = input('Enter YouTube video ID: ').strip()

try:
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id=video_id, languages=["en", "hi"])
except TranscriptsDisabled:
    print("Sorry for Inconvenience!, Transcript is not available for this video.")
    raise SystemExit(1)

transcript = " ".join(doc.text for doc in transcript_list)
print("****=======================================*****")
print("Transcript : ", transcript)
print("****=======================================*****")

rewrite_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature= 0, max_tokens=None, timeout=None, max_retries=2
)

answer_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature= 0.2, max_tokens=None, timeout=None, max_retries=2
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
    rewrite_model, retriever, contextualize_prompt
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
qa_chain = create_stuff_documents_chain(answer_model, qa_prompt)

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

print("\nVideo loaded. Ask anything about it.")
print("Type 'exit' or 'quit' to stop, 'clear' to reset chat memory.\n")


while True:
    try:
        question = input('You: ').strip()
    except (KeyboardInterrupt, EOFError):
        print('\nBye!')
        break
    
    if not question:
        continue
    if question.lower() in {'exit','quit','q'}:
        print('Bye!')
        break

    if question.lower() == 'clear':
        store.pop('video_rag_1',None)
        print('Chat memory cleared.\n')
        continue

    try:
        res = conversational_rag.invoke({'input': question}, config = config)
        print(f"\nBot: {res['answer']}\n")
    except Exception as e:
        print(f'\nError: {e}\n')
