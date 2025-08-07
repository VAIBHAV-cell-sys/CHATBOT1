from flask import Flask, render_template, jsonify, request, session
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv('/Users/wft08/Desktop/CHATBOTAI 2/medibot/.env')

from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template
from src.llm_router import LLMRouter

from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

app = Flask(__name__)
app.secret_key = "super-secret-key"

# === Pinecone setup ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "test"

pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# === Load documents ===
loader = PyPDFLoader("/Users/wft08/Desktop/CHATBOTAI 2/Data/foursight.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
text_chunks = text_splitter.split_documents(documents)
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
    namespace=None,
)
retriever = docsearch.as_retriever(search_kwargs={"k": 2})
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

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

# === Routes ===
@app.route("/")
def index():
    return render_template("chat.html")


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
        {
            "role": "system",
            "content": "You are a helpful assistant that identifies potential blockers and challenges based on a user's goal."
        },
        {
            "role": "user",
            "content": f"""Goal: {goal}

Using the context below, generate a numbered list of 8–10 realistic challenges or obstacles someone might face. Be specific. Don't explain.

Context:
{context}

Output only the list in this format:
1. ...
2. ...
"""
        }
    ]

    result = router.generate(messages)

    # TEMP: log to console for debugging
    print("=== LLM Response ===")
    print(result)

    if not result or not isinstance(result, str):
        return jsonify({"challenges": []})

    # Clean and split into numbered items
    lines = result.splitlines()
    challenges = [re.sub(r'^[0-9.\-)\s]+', '', line.strip()) for line in lines if line.strip()]

    return jsonify({"challenges": challenges})


@app.route("/generate_plan", methods=["POST"])
def generate_plan():
    challenges = request.form.getlist("challenges[]")
    router = get_router()

    formatted = "\n".join(f"- {c}" for c in challenges if c.strip())
    messages = [
        {
            "role": "system",
            "content": "You are an expert in strategic planning and implementation."
        },
        {
            "role": "user",
            "content": f"""Given these selected challenges:\n\n{formatted}\n\nCreate a step-by-step action plan including short-term, mid-term, and long-term tasks. Be specific about what, who, and when.Complete this under 300 tokens only."""
        }
    ]
    plan = router.generate(messages)
    return jsonify({"plan": plan})


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg")
    router = get_router()

    context_docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt_full = prompt_template.format(context=context, question=user_input)

    result = router.generate(prompt_full)
    return jsonify({"answer": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
