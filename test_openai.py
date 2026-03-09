# test_openai.py — Test your OpenAI API key
# Run this with: python3 test_openai.py

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 50)
print("OpenAI API Test")
print("=" * 50)

api_key = os.getenv("OPENAI_API_KEY")

# Check if key is loaded
if not api_key or api_key == "your_openai_key_here":
    print("❌ OPENAI_API_KEY not set in .env file")
    exit()

print(f"✅ API key loaded: {api_key[:12]}...{api_key[-4:]}")

try:
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # cheapest model — costs fractions of a cent
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10
    )

    reply = response.choices[0].message.content
    print(f"✅ OpenAI API works!")
    print(f"✅ Response: {reply}")
    print()
    print("=" * 50)
    print("🎉 OpenAI test complete — everything works!")
    print("=" * 50)

except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Common fixes:")
    print("1. Check OPENAI_API_KEY in your .env file")
    print("2. Make sure you added $10 credit at platform.openai.com")
    print("3. Make sure the key starts with sk-proj-")