from fastapi import FastAPI
from pydantic import BaseModel
from app.agent_impl import run_agent
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class RunRequest(BaseModel):
    input: str
    user_id: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/agent/run")
async def agent_run(req: RunRequest):
    result = await run_agent(req.input, req.user_id)
    return result