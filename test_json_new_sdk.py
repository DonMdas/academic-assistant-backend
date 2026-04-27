from google import genai
from google.genai import types
import json
import os
from dotenv import load_dotenv

load_dotenv() 

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
print(client)

response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents="Extract the details: Alice is 28 years old and works as an engineer.",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"}
            }
        }
    )
)

print(response.text)
data = json.loads(response.text)

print(type(data))   # should print <class 'dict'>
print(data)         # should print {'name': 'Alice', 'age': 28, 'occupation': 'engineer'}





