import os
import openai
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # 1536 dim

def embed_text(text: str) -> np.ndarray:
    resp = openai.Embedding.create(
        model=EMBED_MODEL,
        input=text
    )
    vec = np.array(resp["data"][0]["embedding"], dtype=np.float32)
    return vec  # shape (1536,)

def generate_response(user_input: str, context: str = "", history: List[Dict[str, str]] = None) -> str:
    """
    history: list of {"role": "user"/"assistant", "content": "..."}
    """
    messages = [{"role": "system", "content": "You are a helpful assistant. Use provided context when relevant."}]
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    try:
        resp = openai.responses.create(
            model=CHAT_MODEL,
            input=messages
        )
        return resp.output_text
    except Exception as e:
        return f"[LLM error: {e}]"
