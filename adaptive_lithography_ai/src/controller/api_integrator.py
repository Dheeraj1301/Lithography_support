import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_openai_api(prompt):
    if not openai.api_key:
        raise ValueError("OpenAI API key not found. Check your .env file.")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None
