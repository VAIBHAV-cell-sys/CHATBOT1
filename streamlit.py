import os
import streamlit as st
from dotenv import load_dotenv

from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template

from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# === Load environment variables ===
load_dotenv('/Users/wft08/Desktop/CHATBOTAI 2/medibot/.env')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "test"
REGION = "us-east-1"

# === Initialize Pinecone client ===
pc = Pinecone(api_key=PINECONE_API_KEY)

# === Create index if not exists ===
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Use 768 or 1536 depending on embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )

# === Load and split PDF ===
loader = PyPDFLoader("/Users/wft08/Desktop/CHATBOTAI 2/Data/foursight.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
text_chunks = text_splitter.split_documents(documents)

# === Get Embeddings ===
embeddings = download_hugging_face_embeddings()

# === Upload to Pinecone ===
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    namespace=None,
    client=pc
)

# === Prompt Template ===
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
chain_type_kwargs = {"prompt": prompt}

# === Load LLM ===
llm = CTransformers(
    model="/Users/wft08/Desktop/CHATBOTAI 2/medibot/research/model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        "max_new_tokens": 56,
        "temperature": 0.3,
        "threads": 4,
        "batch_size": 8,
        "context_length": 2048
    }
)

# === Retrieval QA Chain ===
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# === Streamlit UI ===
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("🧠 PDF Chatbot")
query = st.text_input("Ask me something:")

if query:
    with st.spinner("🤔 Thinking..."):
        result = qa.run(query)
        st.success(result)
