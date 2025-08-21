# import requests

# class LLMRouter:
#     SUPPORTED_PERPLEXITY_MODELS = [
#         "sonar-small-online",
#         "sonar-medium-online",
#         "sonar-large-online",
#         "gpt-4-turbo",
#         "claude-3-sonnet-20240229",
#         "gemini-pro"
#     ]
    
#     SUPPORTED_DEEPSEEK_MODELS = [
#         "deepseek-chat",
#         "deepseek-coder"
#     ]

#     def __init__(self, config):
#         """
#         config: dict

#         provider: "local_llama" | "openai" | "perplexity" | "deepseek"
#         model_path: path to model for local_llama
#         api_key: for openai/perplexity/deepseek
#         model: string model ID (for openai/perplexity/deepseek)
#         params: model params (for local_llama)
#         """
#         self.provider = config.get("provider", "local_llama")
#         self.config = config

#     def generate(self, prompt_or_messages):
#         if self.provider == "openai":
#             return self._call_openai(prompt_or_messages)
#         elif self.provider == "local_llama":
#             return self._call_local_llama(prompt_or_messages)
#         elif self.provider == "perplexity":
#             return self._call_perplexity(prompt_or_messages)
#         elif self.provider == "deepseek":
#             return self._call_deepseek(prompt_or_messages)
#         elif self.provider == "aimlapi":
#             return self._call_aimlapi(prompt_or_messages)
#         else:
#             return f"[Unsupported provider: {self.provider}]"

#     def _call_openai(self, prompt_or_messages):
#         try:
#             from openai import OpenAI
            
#             client = OpenAI(api_key=self.config["api_key"])

#             if isinstance(prompt_or_messages, str):
#                 messages = [{"role": "user", "content": prompt_or_messages}]
#             elif isinstance(prompt_or_messages, list):
#                 messages = prompt_or_messages
#             else:
#                 return "[Invalid input format for OpenAI]"

#             response = client.chat.completions.create(model=self.config.get("model", "gpt-3.5-turbo"),
#             messages=messages,
#             temperature=0.7)
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             return f"[OpenAI Error] {str(e)}"

#     def _call_local_llama(self, prompt_or_messages):
#         try:
#             from langchain_community.llms import CTransformers

#             if isinstance(prompt_or_messages, list):
#                 prompt = ""
#                 for msg in prompt_or_messages:
#                     role = msg.get("role", "user")
#                     content = msg.get("content", "")
#                     prompt += f"{role.capitalize()}: {content}\n"
#             elif isinstance(prompt_or_messages, str):
#                 prompt = prompt_or_messages
#             else:
#                 return "[Invalid input format for Local LLaMA]"

#             llm = CTransformers(
#                 model=self.config["model_path"],
#                 model_type="llama",
#                 config=self.config.get("params", {})
#             )
#             return llm(prompt)
#         except Exception as e:
#             return f"[Local LLaMA Error] {str(e)}"

#     def _call_perplexity(self, prompt_or_messages):
#         try:
#             import requests
#             api_key = self.config.get("api_key")
#             model = self.config.get("model", "claude-3-sonnet-20240229")

#             if model not in self.SUPPORTED_PERPLEXITY_MODELS:
#                 return f"[Perplexity Error] Unsupported model '{model}'. Permitted: {self.SUPPORTED_PERPLEXITY_MODELS}"

#             if isinstance(prompt_or_messages, str):
#                 messages = [{"role": "user", "content": prompt_or_messages}]
#             elif isinstance(prompt_or_messages, list):
#                 messages = prompt_or_messages
#             else:
#                 return "[Invalid input format for Perplexity]"

#             response = requests.post(
#                 "https://api.perplexity.ai/chat/completions",
#                 headers={
#                     "Authorization": f"Bearer {api_key}",
#                     "Content-Type": "application/json",
#                     "Accept-Charset": "utf-8"
#                 },
#                 json={
#                     "model": model,
#                     "messages": messages,
#                     "temperature": 0.7
#                 }
#             )

#             if response.status_code != 200:
#                 return f"[Perplexity Error] {response.status_code} - {response.text}"

#             data = response.json()
#             return data['choices'][0]['message']['content'].strip()
#         except Exception as e:
#             return f"[Perplexity Error] {str(e)}"

#     def _call_deepseek(self, prompt_or_messages):
#         try:
#             import requests
#             api_key = self.config.get("api_key")
#             model = self.config.get("model", "deepseek-chat")

#             if model not in self.SUPPORTED_DEEPSEEK_MODELS:
#                 return f"[DeepSeek Error] Unsupported model '{model}'. Supported: {self.SUPPORTED_DEEPSEEK_MODELS}"

#             if isinstance(prompt_or_messages, str):
#                 messages = [{"role": "user", "content": prompt_or_messages}]
#             elif isinstance(prompt_or_messages, list):
#                 messages = prompt_or_messages
#             else:
#                 return "[Invalid input format for DeepSeek]"

#             response = requests.post(
#                 "https://api.deepseek.com/v1/chat/completions",
#                 headers={
#                     "Authorization": f"Bearer {api_key}",
#                     "Content-Type": "application/json"
#                 },
#                 json={
#                     "model": model,
#                     "messages": messages,
#                     "temperature": 0.7
#                 }
#             )

#             if response.status_code != 200:
#                 return f"[DeepSeek Error] {response.status_code} - {response.text}"

#             data = response.json()
#             return data['choices'][0]['message']['content'].strip()
#         except Exception as e:
#             return f"[DeepSeek Error] {str(e)}"
#     def _call_aimlapi(self, prompt_or_messages): 
#         try:
#             api_key = self.config.get("api_key")
#             model = self.config.get("model", "gpt-4o")

#             if isinstance(prompt_or_messages, str):
#                 messages = [{"role": "user", "content": prompt_or_messages}]
#             elif isinstance(prompt_or_messages, list):
#                 messages = prompt_or_messages
#             else:
#                 return "[Invalid input format for AIML API]"

#             response = requests.post(
#                 "https://api.aimlapi.com/v1/chat/completions",
#                 headers={
#                     "Authorization": f"Bearer {api_key}",
#                     "Content-Type": "application/json"
#                 },
#                 json={
#                     "model": model,
#                     "messages": messages,
#                     "temperature": 0.7,
#                     "max_tokens": 256
#                 }
#             )

#             if response.status_code != 200:
#                 return f"[AIML API Error] {response.status_code} - {response.text}"

#             data = response.json()
#             return data['choices'][0]['message']['content'].strip()
#         except Exception as e:
#             return f"[AIML API Error] {str(e)}"