from flask import Flask, render_template, jsonify, request, session
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv('/Users/wft08/Desktop/CHATBOTAI 2/medibot/.env')

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# === Flask app ===
app = Flask(__name__)
app.secret_key = "super-secret-key"

# === PDF Loader helper ===
def load_pdf_file(path):
    """
    Automatically detect if path is a single file or directory
    and load PDFs accordingly.
    """
    full_path = os.path.abspath(path)

    if os.path.isfile(full_path):
        loader = PyPDFLoader(full_path)
        documents = loader.load()
    elif os.path.isdir(full_path):
        loader = DirectoryLoader(full_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
    else:
        raise ValueError(f"Invalid path: {full_path}")

    return documents

# === Text splitter helper ===
def text_split(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

# === HuggingFace embeddings helper ===
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load documents and create Pinecone index ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "test"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Load PDFs and split into chunks
documents = load_pdf_file("medibot/data/atomic.pdf")  # single file
text_chunks = text_split(documents, chunk_size=1000, chunk_overlap=150)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Create Pinecone vector store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    namespace=None,
)
retriever = docsearch.as_retriever(search_kwargs={"k": 2})

# === LLM Router class ===
import requests

class LLMRouter:
    SUPPORTED_PERPLEXITY_MODELS = [
        "sonar-small-online",
        "sonar-medium-online",
        "sonar-large-online",
        "gpt-4-turbo",
        "claude-3-sonnet-20240229",
        "gemini-pro"
    ]
    
    SUPPORTED_DEEPSEEK_MODELS = [
        "deepseek-chat",
        "deepseek-coder"
    ]

    def __init__(self, config):
        self.provider = config.get("provider", "local_llama")
        self.config = config

    def generate(self, prompt_or_messages):
        if self.provider == "openai":
            return self._call_openai(prompt_or_messages)
        elif self.provider == "local_llama":
            return self._call_local_llama(prompt_or_messages)
        elif self.provider == "perplexity":
            return self._call_perplexity(prompt_or_messages)
        elif self.provider == "deepseek":
            return self._call_deepseek(prompt_or_messages)
        elif self.provider == "aimlapi":
            return self._call_aimlapi(prompt_or_messages)
        else:
            return f"[Unsupported provider: {self.provider}]"

    # --- Provider implementations ---
    def _call_openai(self, prompt_or_messages):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config["api_key"])
            messages = (
                [{"role": "user", "content": prompt_or_messages}]
                if isinstance(prompt_or_messages, str) else prompt_or_messages
            )
            response = client.chat.completions.create(
                model=self.config.get("model", "gpt-3.5-turbo"),
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error] {str(e)}"

    def _call_local_llama(self, prompt_or_messages):
        try:
            from langchain_community.llms import CTransformers
            if isinstance(prompt_or_messages, list):
                prompt = "\n".join(f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in prompt_or_messages)
            else:
                prompt = prompt_or_messages
            llm = CTransformers(
                model=self.config["model_path"],
                model_type="llama",
                config=self.config.get("params", {})
            )
            return llm(prompt)
        except Exception as e:
            return f"[Local LLaMA Error] {str(e)}"

    def _call_perplexity(self, prompt_or_messages):
        try:
            api_key = self.config.get("api_key")
            model = self.config.get("model", "claude-3-sonnet-20240229")
            if model not in self.SUPPORTED_PERPLEXITY_MODELS:
                return f"[Perplexity Error] Unsupported model '{model}'."
            messages = [{"role": "user", "content": prompt_or_messages}] if isinstance(prompt_or_messages, str) else prompt_or_messages
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": 0.7}
            )
            if response.status_code != 200:
                return f"[Perplexity Error] {response.status_code} - {response.text}"
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"[Perplexity Error] {str(e)}"

    def _call_deepseek(self, prompt_or_messages):
        try:
            api_key = self.config.get("api_key")
            model = self.config.get("model", "deepseek-chat")
            if model not in self.SUPPORTED_DEEPSEEK_MODELS:
                return f"[DeepSeek Error] Unsupported model '{model}'."
            messages = [{"role": "user", "content": prompt_or_messages}] if isinstance(prompt_or_messages, str) else prompt_or_messages
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": 0.7}
            )
            if response.status_code != 200:
                return f"[DeepSeek Error] {response.status_code} - {response.text}"
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"[DeepSeek Error] {str(e)}"

    def _call_aimlapi(self, prompt_or_messages):
        try:
            api_key = self.config.get("api_key")
            model = self.config.get("model", "gpt-4o")
            messages = [{"role": "user", "content": prompt_or_messages}] if isinstance(prompt_or_messages, str) else prompt_or_messages
            response = requests.post(
                "https://api.aimlapi.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 256}
            )
            if response.status_code != 200:
                return f"[AIML API Error] {response.status_code} - {response.text}"
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"[AIML API Error] {str(e)}"

# === Helper: Get LLM Router ===
def get_router():
    model_name = session.get("llm_name", "local_llama")
    api_key = session.get("api_key")
    if model_name == "openai":
        return LLMRouter({
            "provider": "openai",
            "api_key": api_key,
            "model": "gpt-3.5-turbo"
        })
    else:
        return LLMRouter({
            "provider": "local_llama",
            "model_path": "/Users/wft08/Desktop/CHATBOTAI 2/medibot/research/model/llama-2-7b-chat.ggmlv3.q4_0.bin",
            "params": {
                "max_new_tokens": 300,
                "temperature": 0.3,
                "threads": 4,
                "batch_size": 8,
                "context_length": 2048
            }
        })

# === Flask routes ===
@app.route("/")
def index():
    return render_template("chat1.html")

@app.route("/set_model", methods=["POST"])
def set_model():
    model_name = request.form.get("model")
    api_key = request.form.get("api_key")
    session["llm_name"] = model_name
    session["api_key"] = api_key
    return jsonify({"status": "Model updated"})

@app.route("/generate_challenges", methods=["POST"])
def generate_challenges():
    goal = request.form.get("goal", "")
    router = get_router()
    context_docs = retriever.get_relevant_documents(goal)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    messages = [
        {"role": "system", "content": "You are a helpful assistant that identifies potential blockers and challenges based on a user's goal."},
        {"role": "user", "content": f"Goal: {goal}\n\nContext:\n{context}\n\nOutput only a numbered list of 8–10 challenges."}
    ]
    result = router.generate(messages)
    lines = result.splitlines()
    challenges = [re.sub(r'^[0-9.\-)\s]+', '', line.strip()) for line in lines if line.strip()]
    return jsonify({"challenges": challenges})

@app.route("/generate_plan", methods=["POST"])
def generate_plan():
    challenges = request.form.getlist("challenges[]")
    router = get_router()
    formatted = "\n".join(f"- {c}" for c in challenges if c.strip())
    messages = [
        {"role": "system", "content": "You are an expert in strategic planning."},
        {"role": "user", "content": f"Given these selected challenges:\n{formatted}\nCreate a step-by-step action plan under 300 tokens."}
    ]
    plan = router.generate(messages)
    return jsonify({"plan": plan})

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg")
    router = get_router()
    context_docs = retriever.get_relevant_documents(user_input)
    if not context_docs:
        return jsonify({"answer": "I am sorry, ask Vaibhav Chawla. I am trained only on Atomic Habits content."})
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt_full = f"""
You are an assistant trained only on Atomic Habits content.
Answer ONLY using the context below.
If not in context, respond:
"I am sorry, ask Vaibhav Chawla. I am trained only on Atomic Habits content."

Context:
{context}

Question:
{user_input}
"""
    result = router.generate(prompt_full)
    return jsonify({"answer": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
