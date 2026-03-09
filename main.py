import sys
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

print("=" * 50)
print("Dead Internet Detector — Day 1 Check")
print("=" * 50)

# 1. Check Python version
print(f"✅ Python version: {sys.version.split()[0]}")

# 2. Check libraries
print(f"✅ Pandas version: {pd.__version__}")
print(f"✅ NumPy version: {np.__version__}")
print(f"✅ Requests installed: OK")
print(f"✅ BeautifulSoup installed: OK")

# 3. Check .env file loads
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"✅ .env file loaded: OK (OPENAI_API_KEY found)")
else:
    print(f"⚠️  .env file: OPENAI_API_KEY not set yet (normal — fill in Day 2)")

# 4. Quick BeautifulSoup test
html = "<h1>Hello World</h1><p>This is a test page</p><a href='http://example.com'>link</a>"
soup = BeautifulSoup(html, 'html.parser')
text = soup.get_text()
links = [a['href'] for a in soup.find_all('a', href=True)]
print(f"✅ BeautifulSoup test: extracted text='{text.strip()}', links={links}")

# 5. Quick pandas test
df = pd.DataFrame({'domain': ['site1.com', 'site2.com'], 'score': [0.92, 0.45]})
print(f"✅ Pandas test: created DataFrame with {len(df)} rows")

print("=" * 50)
print("🎉 Day 1 complete! Your environment is ready.")
print("=" * 50)