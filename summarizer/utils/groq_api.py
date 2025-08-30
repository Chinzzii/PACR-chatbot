import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()



def summarize_groq(text, prompt, system_instructions=None, model="llama-3.3-70b-versatile"):
    """
    Use Groq API to summarize text with optional system instructions.
    """
    import os
    from groq import Groq

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    messages = []
    if system_instructions:
        messages.append({"role": "system", "content": system_instructions})
    messages.append({"role": "user", "content": f"{prompt}\n\nText:\n{text}"})

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=1024,
            temperature=0.2,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return "Error generating summary"