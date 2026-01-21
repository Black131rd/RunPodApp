from fastapi import FastAPI
from pydantic import BaseModel, Field
from llama_cpp import Llama
import threading

app = FastAPI(title="llama.cpp API", version="1.0")
llm_lock = threading.Lock()
llm = Llama(
    model_path="/workspace/models/Jinx-Qwen3-30B-A3B-Thinking-2507.Q4_K_M.gguf",
    n_ctx=16384,
    n_gpu_layers=-1,        
    temperature=0.8,
    top_p=0.95,
    repeat_penalty=1.05,
    use_mmap=True,
    use_mlock=True,
    logits_all=False,
    verbose=False,
)
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=2000, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
class GenerateResponse(BaseModel):
    text: str
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    params = {
        "max_tokens": req.max_tokens,
        "stop": ["</s>"],
    }
    if req.temperature is not None:
        params["temperature"] = req.temperature
    if req.top_p is not None:
        params["top_p"] = req.top_p

    # llama.cpp ist NICHT thread-safe → Lock
    with llm_lock:
        result = llm(req.prompt, **params)

    return GenerateResponse(
        text=result["choices"][0]["text"].strip()
    )
@app.get("/health")
def health():
    return {"status": "ok"}
