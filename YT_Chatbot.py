from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

video_id ='Gfr50f6ZBvo'
try:
  api = YouTubeTranscriptApi()
  transcript_list = api.fetch(
      video_id = video_id,
      languages = ['en','hi'])

except TranscriptsDisabled:
  print("Opps! , Sorry for Inconvenience \nTranscript is not available for this video.")
  
transcript = " ".join(chunk.text for chunk in transcript_list)
  
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
  
documents = splitter.create_documents([transcript])

embedding_model = HuggingFaceEmbeddings(
    model ="sentence-transformers/all-MiniLM-L6-v2"
    )

vector_store = FAISS.from_documents(
    documents = documents,
    embedding = embedding_model
)

vector_store.index_to_docstore_id

retriever = vector_store.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k':3}
)

# question = 'what is deepmind?'
# retrieved_docs = retriever.invoke(question)
# context_text = " ".join(doc.page_content for doc in retrieved_docs)

prompt = PromptTemplate(
    template = """ You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}""",
    input_variables = ['context','question']

)

# final_prompt = prompt.invoke({'context' : context_text,
                              # 'question' : question})

llm = HuggingFaceEndpoint(
    model = "meta-llama/Llama-3.1-8B-Instruct",
    task = 'text-genertion'
)

model = ChatHuggingFace(llm = llm)

# model.invoke(final_prompt).content

parser = StrOutputParser()


def format_docs(retrieved_docs):
     context_text = "\n\n ".join(doc.page_content for doc in retrieved_docs)
     return context_text

parallel_chain =RunnableParallel({
    'context':retriever | RunnableLambda(format_docs),
    'question':RunnablePassthrough()}
)

final_chain = parallel_chain | prompt | model | parser

print(final_chain.invoke('what is discussed in this video?'))
