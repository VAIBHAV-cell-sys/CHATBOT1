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
