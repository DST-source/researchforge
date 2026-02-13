"""
HYBRID LLM CLIENT - FIXED VERSION
- Increased Ollama timeout to 300 seconds (5 minutes)
- Updated to new Google GenAI package
- Fixed Gemini model name
"""
import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class HybridLLMClient:
    """
    Hybrid LLM client that prioritizes local Ollama and falls back to Gemini.
    """
    
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.google_key = os.getenv("GOOGLE_API_KEY")
        
        # Check what's available
        self.ollama_available = self._check_ollama()
        self.gemini_available = bool(self.google_key)
        
        if not self.ollama_available and not self.gemini_available:
            print("⚠️  NO LLM PROVIDERS AVAILABLE!")
            print("\nSetup at least ONE:")
            print("  1. Ollama (recommended): https://ollama.com/download")
            print("  2. Google Gemini: https://aistudio.google.com/app/apikey")
        else:
            providers = []
            if self.ollama_available:
                providers.append(f"Ollama ({self.ollama_model})")
            if self.gemini_available:
                providers.append("Google Gemini")
            
            print(f"🤖 Available LLM providers: {' + '.join(providers)}")
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if self.ollama_model in models:
                    return True
                else:
                    print(f"⚠️  Ollama running but model '{self.ollama_model}' not found")
                    print(f"   Run: ollama pull {self.ollama_model}")
                    return False
            return False
        except:
            return False
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1200,
        **kwargs
    ) -> str:
        """Chat completion with automatic provider selection."""
        
        # Try Ollama first (if available)
        if self.ollama_available:
            try:
                return self._ollama_chat(messages, temperature, max_tokens)
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️  Ollama failed: {error_msg[:100]}")
                
                # If timeout, suggest using Gemini
                if "timeout" in error_msg.lower():
                    print("💡 Ollama is too slow. Using Gemini instead...")
                
                if self.gemini_available:
                    print("🔄 Falling back to Google Gemini...")
                else:
                    raise
        
        # Fall back to Gemini
        if self.gemini_available:
            try:
                return self._gemini_chat(messages, temperature, max_tokens)
            except Exception as e:
                raise Exception(f"All LLM providers failed! Gemini error: {e}")
        
        raise Exception("No LLM providers available!")
    
    def _ollama_chat(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Ollama local inference with INCREASED timeout."""
        # Convert messages to prompt
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        prompt = "\n\n".join(parts)
        
        # Call Ollama API with LONGER timeout (5 minutes instead of 2)
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        # INCREASED TIMEOUT: 300 seconds (5 minutes)
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama error: {response.text}")
    
    def _gemini_chat(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Google Gemini API using NEW google-genai package."""
        try:
            # Try new package first
            import google.genai as genai
            from google.genai import types
            
            client = genai.Client(api_key=self.google_key)
            
            # Convert messages to prompt
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"Instructions: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            prompt = "\n\n".join(parts)
            
            # Use gemini-2.0-flash-exp (FREE, latest model)
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            return response.text
            
        except ImportError:
            # Fallback to old package (with warning suppression)
            import warnings
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            import google.generativeai as genai
            
            genai.configure(api_key=self.google_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')  # Use -latest suffix
            
            # Convert messages to prompt
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"Instructions: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            prompt = "\n\n".join(parts)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            return response.text


# ============================================================================
# SIMPLE INTERFACE
# ============================================================================

_client = None

def get_client():
    """Get singleton hybrid client."""
    global _client
    if _client is None:
        _client = HybridLLMClient()
    return _client


def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 1200,
    **kwargs
) -> str:
    """Simple chat interface with automatic provider selection."""
    return get_client().chat(messages, temperature, max_tokens, **kwargs)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTING HYBRID LLM CLIENT")
    print("=" * 70)
    
    test_messages = [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": "Say 'Hybrid LLM client works!' and tell me which provider answered."}
    ]
    
    try:
        print("\n⏳ Generating response...")
        response = chat(test_messages, temperature=0.1, max_tokens=100)
        
        print("\n✅ SUCCESS!")
        print(f"Response: {response}")
        print("\n🎉 Your hybrid LLM setup is ready for ResearchForge!")
        
        print("\n" + "=" * 70)
        print("CONFIGURATION:")
        print("=" * 70)
        client = get_client()
        print(f"Ollama available: {'✅ YES' if client.ollama_available else '❌ NO'}")
        if client.ollama_available:
            print(f"  Model: {client.ollama_model}")
            print(f"  URL: {client.ollama_url}")
            print(f"  Timeout: 300 seconds (5 minutes)")
        print(f"Gemini available: {'✅ YES' if client.gemini_available else '❌ NO'}")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        print("\nTROUBLESHOOTING:")
        print("1. Install new Gemini package: pip install google-genai")
        print("2. Make sure Ollama is running: ollama serve")
        print("3. Check .env has GOOGLE_API_KEY")