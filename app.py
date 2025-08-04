from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('/Users/wft08/Desktop/CHATBOTAI 2/medibot/.env')

# === Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "test"

# === LangChain and Pinecone imports
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template

from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# === Flask App
app = Flask(__name__)

# === Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# === Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# === Load and split documents
loader = PyPDFLoader("/Users/wft08/Desktop/CHATBOTAI 2/Data/foursight.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
text_chunks = text_splitter.split_documents(documents)

# === Load embeddings
embeddings = download_hugging_face_embeddings()

# === Store vectors
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    namespace=None,
)

# === Setup prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)
chain_type_kwargs = {"prompt": prompt}

# === Load LLM
llm = CTransformers(
    model="/Users/wft08/Desktop/CHATBOTAI 2/medibot/research/model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        "max_new_tokens": 300,
        "temperature": 0.3,
        "threads": 4,
        "batch_size": 8,
        "context_length": 2048
    }
)

# === QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# === Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    result = qa({"query": msg})
    return jsonify({"answer": result["result"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
