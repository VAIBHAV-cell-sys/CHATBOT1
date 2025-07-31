from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.templating import Jinja2Templates
from fastapi import Request

import os

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents
extracted_data=load_pdf_file(data="/Users/wft08/Desktop/CHATBOTAI 2/Data")
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks
text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))
from langchain.embeddings import HuggingFaceEmbeddings

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))
from dotenv import load_dotenv
load_dotenv()
from dotenv import load_dotenv
load_dotenv()
from dotenv import load_dotenv
load_dotenv()

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load API key
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Step 2: Create Pinecone client instance (NO init)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Step 3: Index name
index_name = "test"

# Step 4: (Optional) Create index if not exists
if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # Must match the embedding model's output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Step 5: Load PDF and split
loader = PyPDFLoader("/Users/wft08/Desktop/CHATBOTAI 2/Data/foursight.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
text_chunks = text_splitter.split_documents(documents)

# Step 6: Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 7: Store in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

import os
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_pinecone import PineconeVectorStore

llm = CTransformers(
    model="/Users/wft08/Desktop/CHATBOTAI 2/medibot/research/model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        "max_new_tokens": 56,           # Reduce token size to speed up
        "temperature": 0.3,
        "threads": 4,                    # Use 4 CPU threads if available
        "batch_size": 8,                 # Lower = less RAM needed
        "context_length": 2048          # Match model context (adjust if needed)
    }
)

print("Model loaded.")
print("Waiting for user input...")
print("Testing LLM response:")
response = llm("i am afraid to learn what should i do?")
print("Response test:", response)

from langchain.prompts import PromptTemplate

template = """
You are a helpful assistant. Use the provided context to answer the question clearly and concisely in 1–2 sentences. Ensure your response ends with a complete sentence.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="medicalbot",
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)
query = "what is next js ?"
response = qa_chain.run(query)

print("Response:\n", response)



app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get", response_class=HTMLResponse)
async def chat(request: Request, msg: str = Form(...)):
    response = qa_chain.run(msg)
    return templates.TemplateResponse("chat.html", {"request": request, "msg": msg, "response": response})
