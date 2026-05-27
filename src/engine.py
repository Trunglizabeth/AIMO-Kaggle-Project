# src/engine.py
import os
from src.config import (
    MODEL_PATH, MAX_TOKENS, GPU_MEMORY_UTILIZATION,
    N_SAMPLES, TEMPERATURE
)

def is_kaggle() -> bool:
    return os.path.exists("/kaggle")


class VLLMEngine:
    def __init__(self):
        from vllm import LLM, SamplingParams
        print(f"🚀 Khởi tạo vLLM với model: {MODEL_PATH}")
        self.llm = LLM(
            model=MODEL_PATH,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            dtype="float16",
        )
        self.SamplingParams = SamplingParams

    def generate(self, prompts: list, n: int = N_SAMPLES, temperature: float = TEMPERATURE) -> list:
        params = self.SamplingParams(n=n, temperature=temperature, max_tokens=MAX_TOKENS)
        results = self.llm.generate(prompts, params)
        return [[o.text for o in r.outputs] for r in results]


class GroqEngine:
    def __init__(self):
        from groq import Groq
        from dotenv import load_dotenv
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        print("🚀 Khởi tạo Groq Engine (local test mode)")

    def generate(self, prompts: list, n: int = N_SAMPLES, temperature: float = TEMPERATURE) -> list:
        all_outputs = []
        for prompt in prompts:
            outputs = []
            for i in range(n):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                )
                outputs.append(response.choices[0].message.content)
                print(f"  ✅ Sinh output {i+1}/{n}")
            all_outputs.append(outputs)
        return all_outputs


def get_engine():
    if is_kaggle():
        return VLLMEngine()
    else:
        return GroqEngine()