import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

try:
    response = client.models.generate_content(
        model = "gemini-3.8-flash",
        contents="Explain RAG in one sentence"
    )

    print("SUCCESS: Gemini API is working")
    print(response.text)

except Exception as e:
    print("ERROR:", e)