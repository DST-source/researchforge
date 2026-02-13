"""
Multi-Provider LLM Client: Groq (primary) + Gemini (fallback)
"""
import os
import time
from typing import List, Dict
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ResearchForge-Phase0")

# Initialize providers
GROQ_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# Groq client
groq_available = False
if GROQ_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_KEY)
        groq_available = True
        print("✅ Groq initialized (PRIMARY)")
    except ImportError:
        print("⚠️ Groq not installed. Run: pip install groq")

# Gemini client
gemini_available = False
if GOOGLE_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_KEY)
        gemini_available = True
        print("✅ Gemini initialized (FALLBACK)")
    except ImportError:
        print("⚠️ Gemini not installed")


@traceable(name="llm_chat", run_type="llm", tags=["multi-provider"])
def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 1200,
    **kwargs
) -> str:
    """
    Multi-provider chat with automatic fallback
    Tries: Groq → Gemini → Error
    """
    
    # Try Groq first (30 req/min, fast!)
    if groq_available:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # ✅ Correct free model
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"⚠️ Groq failed: {str(e)[:100]}... trying Gemini")
    
    # Fallback to Gemini
    if gemini_available:
        try:
            import google.generativeai as genai
            
            model = genai.GenerativeModel('gemini-1.5-flash')  # ✅ Stable, widely available
            
            # Convert messages to prompt
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    parts.append(f"Instructions: {content}")
                elif role == "user":
                    parts.append(content)
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            
            prompt = "\n\n".join(parts)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                safety_settings={
                    'HARASSMENT': 'block_none',
                    'HATE': 'block_none',
                    'SEXUAL': 'block_none',
                    'DANGEROUS': 'block_none'
                }
            )
            
            return response.text
        
        except Exception as e:
            error = str(e)
            if "429" in error:
                raise Exception("All providers rate limited. Please wait 1 minute.")
            raise Exception(f"Gemini failed: {error[:100]}")
    
    raise Exception("No LLM providers available!")


if __name__ == "__main__":
    print("🧪 Testing multi-provider client...\n")
    
    try:
        response = chat(
            [{"role": "user", "content": "Say 'Working!' in 2 words."}],
            temperature=0
        )
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Failed: {e}")
