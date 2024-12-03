import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import json

# Load config and set up API key
with open("../config/llm.json", "r") as f:
    llm_config = json.load(f)

GOOGLE_API_KEY = llm_config["api_key"]
genai.configure(api_key=GOOGLE_API_KEY)

# paths for database and files
CHROMA_PATH = "chroma"
DATA_PATH = "data/Egyptian Artifacts.pdf"

# load files
loader = PyPDFLoader(DATA_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} documents from {DATA_PATH}.")

# split files into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)
chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

document = chunks[10]
print(document.page_content)
print(document.metadata)

# clear out the database first.
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Set up embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# create a new DB from the documents.
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)
db.add_documents(chunks)
db.persist()
print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
