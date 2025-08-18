from flask import Flask, render_template, jsonify, request, session
from dotenv import load_dotenv
import os
import re
from src.prompt import prompt_template
from src.llm_router import LLMRouter

load_dotenv()

app = Flask(__name__)
app.secret_key = "super-secret-key"

# === Helper: Get LLM Router ===
def get_router():
    model_name = session.get("llm_name", "local_llama")
    api_key = session.get("api_key")

    if model_name == "openai":
        return LLMRouter({
            "provider": "openai",
            "api_key": api_key,
            "model": "gpt-4o-mini"
        })

    elif model_name == "perplexity":
        return LLMRouter({
            "provider": "perplexity",
            "api_key": api_key,
            "model": "sonar-small-online"
        })

    elif model_name == "deepseek":
        return LLMRouter({
            "provider": "deepseek",
            "api_key": api_key,
            "model": "deepseek-chat"
        })
    elif model_name == "aimlapi":
        return LLMRouter({
        "provider": "aimlapi",
        "api_key": api_key,
        "model": "gpt-4o"
    })

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
    session["llm_name"] = request.form.get("model")
    session["api_key"] = request.form.get("api_key")
    return jsonify({"status": "Model updated"})

@app.route("/generate_challenges", methods=["POST"])
def generate_challenges():
    goal = request.form.get("goal", "")
    router = get_router()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that identifies potential blockers and challenges based on a user's goal."
        },
        {
            "role": "user",
            "content": f"""Goal: {goal}

Generate a numbered list of 8–10 realistic challenges or obstacles someone might face. Be specific. Don’t explain.

Output only the list:
1. ...
2. ...
"""
        }
    ]

    result = router.generate(messages)

    print("=== LLM Response ===")
    print(result)

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
        {"role": "user", "content": f"""Given these challenges:\n{formatted}\n\nCreate a step-by-step action plan (short, mid, long-term) under 300 tokens."""}
    ]

    plan = router.generate(messages)
    return jsonify({"plan": plan})

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg")
    router = get_router()

    prompt_full = prompt_template.format(context="", question=user_input)
    result = router.generate(prompt_full)
    return jsonify({"answer": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)