from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import your modules
from voice.transcribe import transcribe_audio
from nlp.intent_parser import parse_intent
from calendar.scheduler import find_available_slots, create_event
from taste.qloo_client import get_recommendations

app = FastAPI(title="SyncMe API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceRequest(BaseModel):
    audio_file_path: str

class ScheduleRequest(BaseModel):
    text_input: str

@app.get("/")
async def root():
    return {"message": "SyncMe API is running"}

@app.post("/process-voice")
async def process_voice(request: VoiceRequest):
    """Process voice input and create calendar event"""
    try:
        # Step 1: Transcribe audio
        transcript = await transcribe_audio(request.audio_file_path)
        
        # Step 2: Parse intent
        intent = await parse_intent(transcript)
        
        # Step 3: Find available time slots
        slots = await find_available_slots(
            duration=intent.duration,
            preferred_time=intent.datetime_window
        )
        
        # Step 4: Get taste recommendations if needed
        recommendations = None
        if intent.vibe:
            recommendations = await get_recommendations(
                taste_token=intent.vibe,
                domain="music"
            )
        
        # Step 5: Create calendar event
        event = await create_event(
            title=intent.event_type,
            start_time=slots[0] if slots else None,
            duration=intent.duration,
            location=intent.location_pref,
            recommendations=recommendations
        )
        
        return {
            "transcript": transcript,
            "intent": intent,
            "available_slots": slots,
            "recommendations": recommendations,
            "created_event": event
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )