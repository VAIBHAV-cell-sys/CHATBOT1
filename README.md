This project explains:

* The **Retrieval-Augmented Generation (RAG)** approach in your app
* The **Direct LLM call** mechanism when context is not available
* The **Dynamic LLM model switching**
* The overall **architecture**
* The **workflow** step-by-step
* Code snippets from your implementation for clarity
* Suggestions for showcasing it professionally

---

# **Documentation: AI Chatbot with RAG, Dynamic LLM Switching & Direct LLM Calls**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Concepts](#key-concepts)

   * RAG (Retrieval-Augmented Generation)
   * Direct LLM Call
   * Dynamic LLM Model Switching
3. [Architecture Overview](#architecture-overview)
4. [Detailed Workflow](#detailed-workflow)
5. [Code Explanation](#code-explanation)

   * Document Loading and Vectorization
   * Retriever Setup
   * LLMRouter (Dynamic LLM Selection)
   * Flask Routes & Request Handling
6. [How to Showcase This Project](#how-to-showcase-this-project)
7. [Future Enhancements](#future-enhancements)

---

## 1. Project Overview

This project is a **chatbot AI** built with Flask that answers questions based on a specific knowledge base — PDFs on *Atomic Habits*. It combines **Retrieval-Augmented Generation (RAG)** and direct language model calls, with the ability to dynamically switch between various LLM providers (OpenAI, Perplexity AI, DeepSeek AI, AIMLAPI).

---

## 2. Key Concepts

### a. Retrieval-Augmented Generation (RAG)

RAG enhances language model responses by first **retrieving relevant documents** from a vector database (here: Pinecone) and then **conditioning the LLM on this retrieved context** to generate accurate, grounded answers.

* Prevents hallucinations.
* Provides domain-specific knowledge.
* Uses embeddings + vector search for retrieval.

### b. Direct LLM Call

If no relevant documents are found, the system falls back to a **direct LLM call** — the model answers without any added context but limited to its pre-trained knowledge.

### c. Dynamic LLM Model Switching

Users can choose between multiple LLM providers and models on the frontend. The backend dynamically routes requests to the selected model, supporting:

* OpenAI (GPT variants)
* Perplexity AI
* DeepSeek AI
* AIMLAPI

---

## 3. Architecture Overview

```plaintext
User
 |
 v
Frontend (HTML/JS)
 |
 v
Flask Backend
 |                        +--------------------+
 |                        | Vector Store (Pinecone) |
 +--> Retrieve relevant docs ------------------->+
 |                                             |
 +----> Compose prompt with context ------------+
 |                                             |
 +----> Dynamic LLM Router --------------------+
 |                                             |
 +----> Return response ------------------------+
 |
 v
User sees contextual answer
```

---

## 4. Detailed Workflow

### Step 1: User sends a message

* Via UI input box.
* Message sent to Flask backend `/get` route.

### Step 2: Retrieval of relevant documents

* Backend uses `retriever.get_relevant_documents(user_input)` to get matching PDF content.
* If no docs found, system prepares for fallback.

### Step 3: Prompt construction

* If context docs found:

  ```
  You are an assistant trained only on Atomic Habits content.
  Answer ONLY using the context below.
  If not in context, respond:
  "I am sorry, ask Vaibhav Chawla. I am trained only on Atomic Habits content."

  Context:
  [retrieved documents]

  Question:
  [user_input]
  ```
* Else:

  * The query is sent directly to the LLM without context.

### Step 4: Dynamic LLM model selection

* Based on user-selected model (OpenAI, Perplexity, DeepSeek, AIMLAPI), the backend dynamically routes the request.
* Uses the `LLMRouter` class.

### Step 5: LLM generates response

* Response returned to frontend.
* Displayed in the chat interface.

---

## 5. Code Explanation

### a. Document Loading and Vectorization

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

def load_pdf_file(path):
    # Load PDFs either from a single file or directory
    ...

documents = load_pdf_file("path_to_atomic_habits_pdfs")
chunks = text_split(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create or connect to Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_kwargs={"k": 2})
```

### b. Retriever Setup

```python
context_docs = retriever.get_relevant_documents(user_input)
```

* Retrieves top-2 relevant chunks to the query.

### c. LLMRouter: Dynamic LLM Switching

```python
class LLMRouter:
    def __init__(self, config):
        self.provider = config.get("provider", "openai")
        self.config = config

    def generate(self, prompt_or_messages):
        if self.provider == "openai":
            return self._call_openai(prompt_or_messages)
        elif self.provider == "perplexity":
            return self._call_perplexity(prompt_or_messages)
        elif self.provider == "deepseek":
            return self._call_deepseek(prompt_or_messages)
        elif self.provider == "aimlapi":
            return self._call_aimlapi(prompt_or_messages)
        else:
            return f"[Unsupported provider: {self.provider}]"
```

* The `generate` method routes to the right API.
* Supports different authentication & request formats per provider.

### d. Flask Routes

* `/get` route handles chat queries:

```python
@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg")
    router = get_router()
    context_docs = retriever.get_relevant_documents(user_input)

    if not context_docs:
        # Fallback to direct LLM call without context
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
```

* `/set_model` route allows frontend to update selected LLM and API key dynamically.

---

## 6. How to Showcase This Project

* **Live demo**: Host on Heroku, AWS, or Railway for real-time interaction.
* **UI walkthrough**:

  * Show model switching dropdown.
  * Enter questions related to Atomic Habits.
  * Generate challenges & action plans to demonstrate multi-step flows.
* **Explain RAG**:

  * Use diagrams showing retrieval + generation.
  * Highlight how context grounds the answers.
* **Explain fallback**:

  * Show what happens with no relevant docs (direct LLM).
* **Show logs**:

  * Print selected LLM, API key usage, etc.
* **Performance stats**:

  * Show latency differences between providers.
* **Security**:

  * Explain API key handling via sessions.

---

## 7. Future Enhancements

* Add **more knowledge sources** (web, databases).
* Enable **contextual memory** (chat history awareness).
* Improve UI with **streaming responses**.
* Add **user authentication** and personalized models.
* Support **multi-turn conversations** with retrieval of prior chats.

---

---

# **Summary**

Your project is a sophisticated **multi-provider chatbot system** that leverages:

* **RAG** for grounded, domain-specific answers
* **Direct LLM fallback** when no documents found
* **Dynamic LLM switching** allowing flexibility in backend AI engines



OPTION 2 FOR DOCUMENTATION
---

# **Documentation: My AI Chatbot with RAG, Dynamic LLM Switching & Direct LLM Calls**

---

## 1. Project Overview

I built a chatbot AI using Flask that answers questions based on a specific knowledge base — PDFs about *Atomic Habits*. I combined **Retrieval-Augmented Generation (RAG)** and direct language model calls, with the ability to dynamically switch between various LLM providers (OpenAI, Perplexity AI, DeepSeek AI, AIMLAPI).

---

## 2. Key Concepts

### a. Retrieval-Augmented Generation (RAG)

RAG improves language model responses by first **retrieving relevant documents** from a vector database (I used Pinecone), and then **conditioning the LLM on this retrieved context** to generate accurate and grounded answers.

* This approach helps prevent hallucinations.
* It lets me provide domain-specific knowledge.
* I use embeddings and vector search to retrieve relevant documents.

### b. Direct LLM Call

If I don’t find relevant documents in my knowledge base, the system falls back to a **direct LLM call** — the model answers the question without any added context but based on its pre-trained knowledge.

### c. Dynamic LLM Model Switching

I implemented the ability to choose between multiple LLM providers and models from the frontend. On the backend, I dynamically route requests to the selected model. This supports:

* OpenAI (GPT variants)
* Perplexity AI
* DeepSeek AI
* AIMLAPI

---

## 3. Architecture Overview

The overall flow looks like this:

```plaintext
User
 |
 v
Frontend (HTML/JS)
 |
 v
Flask Backend
 |                        +--------------------+
 |                        | Vector Store (Pinecone) |
 +--> Retrieve relevant docs ------------------->+
 |                                             |
 +----> Compose prompt with context ------------+
 |                                             |
 +----> Dynamic LLM Router --------------------+
 |                                             |
 +----> Return response ------------------------+
 |
 v
User sees contextual answer
```

---

## 4. Detailed Workflow

### Step 1: User sends a message

* The user types a question in the UI.
* The question is sent to my Flask backend `/get` route.

### Step 2: Retrieval of relevant documents

* I use `retriever.get_relevant_documents(user_input)` to fetch matching content from the PDFs.
* If no documents are found, I prepare to fallback.

### Step 3: Prompt construction

* When relevant docs are found, I construct a prompt like this:

```
You are an assistant trained only on Atomic Habits content.
Answer ONLY using the context below.
If not in context, respond:
"I am sorry, ask Vaibhav Chawla. I am trained only on Atomic Habits content."

Context:
[retrieved documents]

Question:
[user_input]
```

* If no docs are found, I send the question directly to the LLM without additional context.

### Step 4: Dynamic LLM model selection

* Based on the user’s selected model (OpenAI, Perplexity, DeepSeek, AIMLAPI), my backend dynamically routes the request to the chosen provider.
* I implemented this logic in the `LLMRouter` class.

### Step 5: LLM generates the response

* The LLM returns the answer.
* I send the response back to the frontend and display it in the chat.

---

## 5. Code Explanation

### a. Document Loading and Vectorization

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

def load_pdf_file(path):
    # I load PDFs either from a single file or a directory
    ...

documents = load_pdf_file("path_to_atomic_habits_pdfs")
chunks = text_split(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# I connect to Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_kwargs={"k": 2})
```

### b. Retriever Setup

```python
context_docs = retriever.get_relevant_documents(user_input)
```

* This gets the top 2 relevant chunks for the question.

### c. LLMRouter: Dynamic LLM Switching

```python
class LLMRouter:
    def __init__(self, config):
        self.provider = config.get("provider", "openai")
        self.config = config

    def generate(self, prompt_or_messages):
        if self.provider == "openai":
            return self._call_openai(prompt_or_messages)
        elif self.provider == "perplexity":
            return self._call_perplexity(prompt_or_messages)
        elif self.provider == "deepseek":
            return self._call_deepseek(prompt_or_messages)
        elif self.provider == "aimlapi":
            return self._call_aimlapi(prompt_or_messages)
        else:
            return f"[Unsupported provider: {self.provider}]"
```

* The `generate` method dynamically routes calls to the selected LLM provider.

### d. Flask Routes

In the `/get` route, I handle chat queries like this:

```python
@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg")
    router = get_router()
    context_docs = retriever.get_relevant_documents(user_input)

    if not context_docs:
        # No relevant docs found, fallback to direct LLM call
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
```

* The `/set_model` route lets me update the selected LLM and API key from the frontend dynamically.

---

## 6. How I Showcase This Project

* I plan to deploy a **live demo** on platforms like Heroku, AWS, or Railway.
* I’ll demonstrate the UI:

  * Show the model-switching dropdown.
  * Ask questions related to Atomic Habits.
  * Generate detailed challenges and action plans.
* I explain RAG with diagrams showing how retrieval + generation work together.
* I highlight the fallback mechanism when no documents match.
* I also show logs of which model is selected and API key usage.
* I share performance metrics like latency differences across providers.
* I discuss API key security, explaining how I store keys in sessions.

---

## 7. Future Enhancements I Plan

* Add more knowledge sources, like web scraping or databases.
* Implement contextual memory for multi-turn conversations.
* Improve the UI to support streaming answers.
* Add user authentication to personalize models.
* Expand support for longer, multi-turn interactions.

---

# Summary

This project showcases my ability to build a sophisticated multi-provider chatbot that:

* Uses **RAG** for accurate, domain-specific answers grounded in Atomic Habits PDFs.
* Falls back to **direct LLM calls** when no relevant docs are found.
* Supports **dynamic switching** between multiple LLM providers.


