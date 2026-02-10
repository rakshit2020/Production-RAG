# ============== ADD THIS IMPORT ==============
from langchain_openai import ChatOpenAI


# ============== UPDATE CONFIG ==============
class Config:
    # ... existing config ...
    
    # vLLM Settings
    VLLM_BASE_URL = "http://localhost:8000/v1"  # Your vLLM endpoint
    VLLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Model you loaded in vLLM
    VLLM_API_KEY = "EMPTY"  # vLLM doesn't need real key, but field is required
    
    # Set which LLM backend to use
    LLM_BACKEND = "vllm"  # Options: "vllm" or "nim"


# ============== UPDATE get_llm() FUNCTION ==============
def get_llm():
    """Initialize LLM (vLLM or NVIDIA NIM)."""
    
    if Config.LLM_BACKEND == "vllm":
        # vLLM (OpenAI-compatible endpoint)
        return ChatOpenAI(
            base_url=Config.VLLM_BASE_URL,
            api_key=Config.VLLM_API_KEY,
            model=Config.VLLM_MODEL_NAME,
            temperature=0.2,
            max_tokens=1024
        )
    else:
        # NVIDIA NIM
        return ChatNVIDIA(
            model=Config.LLM_MODEL,
            temperature=0.2,
            max_tokens=1024
        )


# Start vLLM server with OpenAI-compatible API
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --tensor-parallel-size 1