from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
from rag_pipeline import RAGPipeline

app = FastAPI(
    title="YouTube RAG Chatbot API",
    description="Ask questions about any YouTube video using RAG + LLaMA",
    version="1.0.0"
)

# Allow React frontend (running on port 5173) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared RAGPipeline instance (embedding model + LLM loaded once)
pipeline = RAGPipeline()


# ----------- Request/Response Models -----------

class LoadVideoRequest(BaseModel):
    video_id: str

class LoadVideoResponse(BaseModel):
    status: str
    message: str
    video_id: str

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    video_id: str


# ----------- Endpoints -----------

@app.get("/")
def root():
    return {"message": "YouTube RAG Chatbot API is running."}


@app.post("/load-video", response_model=LoadVideoResponse)
def load_video(request: LoadVideoRequest):
    """
    Accepts a YouTube video ID, fetches its transcript,
    and builds the RAG pipeline (embeddings + vectorstore + chain).
    """
    video_id = request.video_id.strip()

    if not video_id:
        raise HTTPException(status_code=400, detail="Video ID cannot be empty.")

    try:
        pipeline.load_video(video_id)
        return LoadVideoResponse(
            status="success",
            message="Video loaded successfully. You can now ask questions.",
            video_id=video_id
        )

    except TranscriptsDisabled:
        raise HTTPException(
            status_code=422,
            detail="Transcripts are disabled for this video. Please try another video."
        )
    except NoTranscriptFound:
        raise HTTPException(
            status_code=404,
            detail="No English or Hindi transcript found for this video."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load video: {str(e)}"
        )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Accepts a question, runs the RAG chain, and returns the LLM's answer.
    Requires a video to have been loaded via /load-video first.
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = pipeline.ask(question)
        return AskResponse(
            answer=answer,
            video_id=pipeline.video_id
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)  # "No video loaded. Call load_video() first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get answer: {str(e)}"
        )
