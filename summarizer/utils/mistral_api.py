import os

from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()


def summarize_mistral(text, prompt, system_instructions=None, model="mistral-large-latest"):
    """
    Use Mistral API to summarize text with optional system instructions.
    """
    
    client = Mistral(
        api_key=os.environ.get("MISTRAL_API_KEY"),
    )
    messages = []
    if system_instructions:
        messages.append({"role": "system", "content": system_instructions})
    messages.append({"role": "user", "content": f"{prompt}\n\nText:\n{text}"})

    try:
        chat_completion = client.chat.complete(
            messages=messages,
            model=model,
            max_tokens=1024,
            temperature=0.2,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with Mistral API: {e}")
        return "Error generating summary"