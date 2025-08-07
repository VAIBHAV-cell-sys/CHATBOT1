from flask import Flask, render_template, jsonify, request, session
from dotenv import load_dotenv
import os
import re

from src.llm_router import LLMRouter

app = Flask(__name__)
app.secret_key = "super-secret-key"

# Load environment variables
load_dotenv('/Users/wft08/Desktop/CHATBOTAI 2/medibot/.env')

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
    import re
    goal = request.form.get("goal", "")
    router = get_router()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that identifies realistic challenges based on a user's goal."
        },
        {
            "role": "user",
            "content": f"""My goal is: {goal}

Generate a numbered list of **exactly 8** realistic and specific challenges or obstacles someone might face while pursuing this goal.

**Output must follow this format:**
1. [Challenge 1]
2. [Challenge 2]
...
8. [Challenge 8]

Do not add any introductions or explanations. Only return the list."""
        }
    ]

    result = router.generate(messages)

    print("=== LLM Response ===")
    print(result)

    if not result or not isinstance(result, str):
        return jsonify({"challenges": []})

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
            "content": f"""Given these selected challenges:\n\n{formatted}\n\nCreate a step-by-step action plan including short-term, mid-term, and long-term tasks. Be specific about what, who, and when. Must keep it under 300 tokens."""
        }
    ]
    plan = router.generate(messages)
    return jsonify({"plan": plan})

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg")
    router = get_router()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    result = router.generate(messages)
    return jsonify({"answer": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
