import langchain_community
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled 
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

video_id = "Gfr50f6ZBvo"
try:
     api = YouTubeTranscriptApi()
     transcript_list = api.fetch(
          video_id = video_id,
          languages = ['en','hi'] 
     )
except TranscriptsDisabled:
     print("Sorry for Inconvinience!, Transcript is not available for this video.")
     
transcript = " ".join(doc.text for doc in transcript_list)
print("****=======================================*****")
print('Transcript : ' , transcript)
print("****=======================================*****")


splitter = RecursiveCharacterTextSplitter(
     chunk_size = 1000,
     chunk_overlap = 200)


documents = splitter.create_documents([transcript])

# embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2")
# -------------------------------------------------------

embeddings2 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(
     documents = documents,
     embedding= embeddings2
) 

# vectorstore.index_to_docstore_id(whatever ids text you want to see - enter Id )


retriever = vectorstore.as_retriever(
                                      search_type = 'similarity',
                                      search_kwargs = {'k':3})

# retrieved_docs = retriever.invoke(query)
# context = " ".join(doc.page_content for doc in retrieved_docs)

prompt = PromptTemplate(
     template="""You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}""",
    input_variables = ['context','question']
)

model = ChatGoogleGenerativeAI( model = 'gemini-2.5-flash',
                                   max_tokens = None,
                                   timeout = None,
                                   max_retries = 2)

parser = StrOutputParser()

# retrieved_docs = retriever.invoke(query) 
def formatted_docs(retrieved_docs):
     context_text = "".join(doc.page_content for doc in retrieved_docs)
     return context_text



parallel_chain = RunnableParallel(
     {'context': retriever | RunnableLambda(formatted_docs),
     'question' : RunnablePassthrough()}
     )

final_chain = parallel_chain | prompt | model | parser


query = "What topics are discussed in this video?"

result = final_chain.invoke(query)

print(result)

