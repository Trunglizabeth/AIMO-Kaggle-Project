# notebooks/test_prompt.py
import sys
import os
sys.path.append(".")

from dotenv import load_dotenv
from groq import Groq
from src.prompts import build_prompt

# Load API key từ file .env
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=API_KEY)

# Thử với 1 bài toán
problem = "Find the remainder when 2^100 is divided by 1000."

messages = build_prompt(problem)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",    
    messages=messages,
    temperature=0.7,
    max_tokens=1024,
)

output = response.choices[0].message.content
print("=" * 50)
print("OUTPUT TỪ MODEL:")
print("=" * 50)
print(output)