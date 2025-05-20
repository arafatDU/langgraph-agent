from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_runner import run_agent

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        messages = run_agent(request.message)
        # Convert messages to dicts for JSON serialization
        return {"messages": [m.dict() if hasattr(m, 'dict') else str(m) for m in messages]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
