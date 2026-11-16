from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    """
    Encapsulates the full RAG pipeline for a YouTube video.
    - load_video(video_id): fetches transcript, builds FAISS vectorstore & LangChain chain
    - ask(question): runs the chain and returns the LLM answer
    """

    def __init__(self):
        self.final_chain = None
        self.video_id = None

        # Load embedding model once at startup (expensive operation)
        self.embedding_model = HuggingFaceEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load LLM once at startup
        llm = HuggingFaceEndpoint(
            model="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation"  # Fixed typo from original code
        )
        self.model = ChatHuggingFace(llm=llm)
        self.parser = StrOutputParser()

        self.prompt = PromptTemplate(
            template=""" You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}""",
            input_variables=["context", "question"]
        )

    def load_video(self, video_id: str):
        """
        Fetches the YouTube transcript, splits it into chunks,
        generates embeddings, and builds the FAISS vectorstore + retrieval chain.
        Raises exceptions on failure so the API layer can handle them.
        """
        # Step 1: Fetch transcript
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(
            video_id=video_id,
            languages=["en", "hi"]
        )

        transcript = " ".join(chunk.text for chunk in transcript_list)

        # Step 2: Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = splitter.create_documents([transcript])

        # Step 3: Build FAISS vector store
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

        # Step 4: Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Step 5: Build the full LangChain chain
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        self.final_chain = parallel_chain | self.prompt | self.model | self.parser
        self.video_id = video_id

    def ask(self, question: str) -> str:
        """
        Runs the retrieval + LLM chain and returns the answer string.
        Raises RuntimeError if video has not been loaded yet.
        """
        if self.final_chain is None:
            raise RuntimeError("No video loaded. Call load_video() first.")
        return self.final_chain.invoke(question)
