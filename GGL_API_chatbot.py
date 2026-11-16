import langchain_community
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled 
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

embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2")
# -------------------------------------------------------

embeddings2 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(
     documents = documents,
     embedding= embeddings2
) 

# vectorstore.index_to_docstore_id(whatever ids text you want to see - enter Id )


retiriever = vectorstore.as_retriever(
                                      search_type = 'similarity',
                                      search_kwargs = {'k':3})

query = "What topics are discussed in this video?"
retrieved_docs = retiriever.invoke(query)
context = " ".join(doc.page_content for doc in retrieved_docs)

prompt_template = PromptTemplate(
     template="""You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}""",
    input_variables = ['context','question']
)

prompt = prompt_template.invoke({'question' : query,
               'context': context})

model = ChatGoogleGenerativeAI( model = 'gemini-2.5-flash',
                                   max_tokens = None,
                                   timeout = None,
                                   max_retries = 2)


result = model.invoke(prompt)

print(result.content)

