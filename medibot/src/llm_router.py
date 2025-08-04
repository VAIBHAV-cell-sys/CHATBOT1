# src/llm_router.py

class LLMRouter:
    def __init__(self, config):
        """
        config: dict
        Example for local llama:
            {
                "provider": "local_llama",
                "model_path": "path/to/model.bin",
                "params": {
                    "max_new_tokens": 300,
                    "temperature": 0.7,
                    ...
                }
            }
        Example for OpenAI:
            {
                "provider": "openai",
                "api_key": "sk-xxxx",
                "model": "gpt-3.5-turbo"
            }
        """
        self.provider = config.get("provider", "local_llama")
        self.config = config

    def generate(self, prompt):
        """
        Generate response using the selected LLM.
        """
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "local_llama":
            return self._call_local_llama(prompt)
        else:
            return "[Unsupported provider: {}]".format(self.provider)

    def _call_openai(self, prompt):
        try:
            import openai
            openai.api_key = self.config["api_key"]

            response = openai.ChatCompletion.create(
                model=self.config.get("model", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            return response.choices[0].message["content"].strip()

        except Exception as e:
            return f"[OpenAI Error] {str(e)}"

    def _call_local_llama(self, prompt):
        try:
            from langchain_community.llms import CTransformers

            llm = CTransformers(
                model=self.config["model_path"],
                model_type="llama",
                config=self.config.get("params", {})
            )

            return llm(prompt)

        except Exception as e:
            return f"[Local LLaMA Error] {str(e)}"
