"""
FastAPI Backend — Wine Recommendation Chatbot
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from recommender import ContentBasedRecommender
from openai_agent import WineAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CSV_PATH = os.getenv(
    "WINE_CSV",
    str(Path(__file__).parent.parent / "XWines_Full_100K_wines (1).csv"),
)

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not set. Add it to chatbot/.env like OPENAI_API_KEY=sk-..."
    )

# ---------------------------------------------------------------------------
# Startup — load recommender once
# ---------------------------------------------------------------------------
recommender = ContentBasedRecommender(CSV_PATH)
agent = WineAgent(recommender=recommender, api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Wine Recommendation Chatbot", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    wines: list | None = None
    tool_called: str | None = None


class ResetRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Create a session ID if not provided
    session_id = req.session_id or str(uuid.uuid4())

    try:
        result = agent.chat(session_id=session_id, user_message=req.message)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit reached. Please wait a moment and try again.",
            )
        raise HTTPException(status_code=500, detail="An error occurred. Please try again.")

    return ChatResponse(
        session_id=session_id,
        reply=result["reply"],
        wines=result.get("wines"),
        tool_called=result.get("tool_called"),
    )


@app.post("/api/reset")
async def reset(req: ResetRequest):
    agent.reset_session(req.session_id)
    return {"status": "ok", "message": "Conversation reset."}


@app.get("/api/options")
async def get_options():
    """Return available filter values for the UI."""
    return recommender.get_available_options()


@app.get("/api/health")
async def health():
    return {"status": "ok", "wines_loaded": len(recommender.df)}
