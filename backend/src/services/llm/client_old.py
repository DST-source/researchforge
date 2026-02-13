"""
Unified LLM client: DeepInfra (LangChain) + OpenRouter (fallback).
FIXED: Uses model_kwargs correctly for DeepInfra.
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "deepinfra")
MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")


def chat(
    messages: List[Dict],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1200,
    **kwargs
) -> str:
    """Chat completion with automatic provider fallback."""
    m = model or MODEL
    
    try:
        if PROVIDER == "deepinfra":
            return _deepinfra_chat(messages, m, temperature, max_tokens, **kwargs)
        elif PROVIDER == "openrouter":
            return _openrouter_chat(messages, m, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")
    except Exception as e:
        print(f"⚠️ {PROVIDER} failed: {e}")
        print("🔄 Falling back to OpenRouter...")
        return _openrouter_chat(
            messages,
            os.getenv("OPENROUTER_MODEL_FALLBACK", "meta-llama/llama-3.2-3b-instruct:free"),
            temperature,
            max_tokens,
            **kwargs
        )


def _deepinfra_chat(
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs
) -> str:
    """DeepInfra via LangChain - FIXED model_kwargs pattern."""
    from langchain_community.llms import DeepInfra
    
    api_token = os.getenv("DEEPINFRA_API_TOKEN")
    if not api_token:
        raise ValueError("DEEPINFRA_API_TOKEN not found in .env")
    
    # Convert messages to single prompt (DeepInfra LLM expects string)
    prompt = _messages_to_prompt(messages)
    
    # Create LLM instance FIRST (no params in constructor)
    llm = DeepInfra(
        model_id=model,
        deepinfra_api_token=api_token,
    )
    
    # THEN set model_kwargs (this is the correct LangChain pattern)
    llm.model_kwargs = {
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "repetition_penalty": 1.1,
        "top_p": 0.9,
        **kwargs  # Allow additional params
    }
    
    return llm.invoke(prompt)


def _openrouter_chat(
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs
) -> str:
    """OpenRouter fallback."""
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")
    
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    
    return resp.choices[0].message.content


def _messages_to_prompt(messages: List[Dict]) -> str:
    """Convert chat messages to single prompt string."""
    parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


# Vision support (future)
def chat_vision(messages: List[Dict], images: List[str], model: Optional[str] = None, **kwargs) -> str:
    """Chat with DeepInfra vision models."""
    from openai import OpenAI
    
    api_token = os.getenv("DEEPINFRA_API_TOKEN")
    vision_model = model or os.getenv("VISION_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")
    
    content = []
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": img}})
    if messages:
        content.append({"type": "text", "text": messages[-1]["content"]})
    
    client = OpenAI(base_url="https://api.deepinfra.com/v1/openai", api_key=api_token)
    resp = client.chat.completions.create(model=vision_model, messages=[{"role": "user", "content": content}], **kwargs)
    return resp.choices[0].message.content
