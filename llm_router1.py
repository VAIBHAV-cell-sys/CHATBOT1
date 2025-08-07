# src/llm_router1.py

class LLMRouter:
    def __init__(self, config):
        self.provider = config.get("provider", "local_llama")
        self.config = config

    def generate(self, prompt_or_messages):
        if self.provider == "openai":
            return self._call_openai(prompt_or_messages)
        elif self.provider == "local_llama":
            return self._call_local_llama(prompt_or_messages)
        return f"[Unsupported provider: {self.provider}]"

    def _call_openai(self, prompt_or_messages):
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.config["api_key"])

            if isinstance(prompt_or_messages, str):
                messages = [{"role": "user", "content": prompt_or_messages}]
            elif isinstance(prompt_or_messages, list):
                messages = prompt_or_messages
            else:
                return "[Invalid input format for OpenAI]"

            response = client.chat.completions.create(model=self.config.get("model", "gpt-3.5-turbo"),
            messages=messages,
            temperature=0.7)
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[OpenAI Error] {str(e)}"

    def _call_local_llama(self, prompt_or_messages):
        try:
            from langchain_community.llms import CTransformers

            if isinstance(prompt_or_messages, list):
                prompt = ""
                for msg in prompt_or_messages:
                    prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
            elif isinstance(prompt_or_messages, str):
                prompt = prompt_or_messages
            else:
                return "[Invalid input format for Local LLaMA]"

            llm = CTransformers(
                model=self.config["model_path"],
                model_type="llama",
                config=self.config.get("params", {})
            )

            return llm(prompt)

        except Exception as e:
            return f"[Local LLaMA Error] {str(e)}"
