
from google import genai
import os
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
r = client.models.generate_content(model='gemini-3-flash-preview', contents='say hi')
print(r.text)