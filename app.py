# Test if everything is working
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ API Key loaded successfully!")
    print(f"Key starts with: {api_key[:10]}...")
else:
    print("❌ API Key not found! Check your .env file")