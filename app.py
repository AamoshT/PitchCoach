from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Form, Depends 
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import FileResponse, JSONResponse 
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient 
from bson import ObjectId 
from datetime import datetime, timezone
import uvicorn  
import tempfile 
import os
import json
import requests
from dotenv import load_dotenv 
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks 
import shutil
import re

load_dotenv()

app = FastAPI()
from auth import router as auth_router, verify_token, set_users_collection
app.include_router(auth_router)
app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_URI", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
DO_GRADIENT_API_KEY = os.getenv("DO_GRADIENT_API_KEY", "")
DO_GRADIENT_BASE_URL = os.getenv("DO_GRADIENT_BASE_URL", "https://inference.do-ai.run/v1")
DO_GRADIENT_MODEL = os.getenv("DO_GRADIENT_MODEL", "openai-gpt-oss-20b")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")


mongo_client = MongoClient(MONGO_URI) if MONGO_URI else None
database = mongo_client["pitchcoach"] if mongo_client is not None else None
sessions_collection = database["pitch_sessions"] if database is not None else None
users_collection = database["users"] if database is not None else None

set_users_collection(users_collection)

POSE_POINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

CONNECTIONS = [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
]

# Change your route to look inside the folder
@app.get("/")
async def serve_index_file():
    return FileResponse("templates/index.html")

@app.get("/app")
async def serve_app(token: str = Depends(verify_token)):
    return FileResponse("templates/recorder.html")

@app.get("/recorder")
async def serve_recorder(token: str = Depends(verify_token)):
    return FileResponse("templates/recorder.html")

def calculate_summary_scores(pose_frames: list):
    if not pose_frames:
        return {
            "gesture_score": 0,
            "posture_score": 0,
            "steadiness_score": 0,
            "overall_score": 0,
        }

    shoulder_tilt_values = []
    body_shift_values = []
    wrist_motion_values = []

    previous_frame = None

    for frame in pose_frames:
        landmarks = frame.get("landmarks", {})
        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")
        left_hip = landmarks.get("left_hip")
        right_hip = landmarks.get("right_hip")
        left_wrist = landmarks.get("left_wrist")
        right_wrist = landmarks.get("right_wrist")

        if left_shoulder and right_shoulder:
            shoulder_tilt_values.append(abs(left_shoulder["y"] - right_shoulder["y"]))

        if left_shoulder and right_shoulder and left_hip and right_hip:
            current_body_center_x = (
                left_shoulder["x"] + right_shoulder["x"] + left_hip["x"] + right_hip["x"]
            ) / 4.0

            if previous_frame:
                previous_landmarks = previous_frame.get("landmarks", {})
                previous_left_shoulder = previous_landmarks.get("left_shoulder")
                previous_right_shoulder = previous_landmarks.get("right_shoulder")
                previous_left_hip = previous_landmarks.get("left_hip")
                previous_right_hip = previous_landmarks.get("right_hip")

                if previous_left_shoulder and previous_right_shoulder and previous_left_hip and previous_right_hip:
                    previous_body_center_x = (
                        previous_left_shoulder["x"] + previous_right_shoulder["x"] +
                        previous_left_hip["x"] + previous_right_hip["x"]
                    ) / 4.0
                    body_shift_values.append(abs(current_body_center_x - previous_body_center_x))

        if previous_frame and left_wrist and right_wrist:
            previous_landmarks = previous_frame.get("landmarks", {})
            previous_left_wrist = previous_landmarks.get("left_wrist")
            previous_right_wrist = previous_landmarks.get("right_wrist")

            if previous_left_wrist:
                wrist_motion_values.append(
                    ((left_wrist["x"] - previous_left_wrist["x"]) ** 2 + (left_wrist["y"] - previous_left_wrist["y"]) ** 2) ** 0.5
                )
            if previous_right_wrist:
                wrist_motion_values.append(
                    ((right_wrist["x"] - previous_right_wrist["x"]) ** 2 + (right_wrist["y"] - previous_right_wrist["y"]) ** 2) ** 0.5
                )

        previous_frame = frame

    average_shoulder_tilt = sum(shoulder_tilt_values) / max(1, len(shoulder_tilt_values))
    average_body_shift = sum(body_shift_values) / max(1, len(body_shift_values))
    average_wrist_motion = sum(wrist_motion_values) / max(1, len(wrist_motion_values))

    def clamp_score(value):
        return max(0, min(100, round(value, 1)))

    gesture_score = clamp_score(100 - abs(average_wrist_motion - 0.018) * 3500)
    posture_score = clamp_score(100 - average_shoulder_tilt * 1200)
    steadiness_score = clamp_score(100 - average_body_shift * 2500)
    overall_score = clamp_score(
        gesture_score * 0.35 +
        posture_score * 0.35 +
        steadiness_score * 0.20 +
        100 * 0.10
    )

    return {
        "gesture_score": gesture_score,
        "posture_score": posture_score,
        "steadiness_score": steadiness_score,
        "overall_score": overall_score,
    }


def transcribe_with_elevenlabs(media_file_path: str):
    if not ELEVENLABS_API_KEY:
        return {
            "full_text": "",
            "segments": [],
            "warning": "ELEVENLABS_API_KEY not set"
        }

    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
    }

    with open(media_file_path, "rb") as media_file:
        files = {
            "file": media_file,
        }
        data = {
            "model_id": "scribe_v2",
        }
        response = requests.post(url, headers=headers, files=files, data=data, timeout=120)

    if response.status_code >= 400:
        return {
            "full_text": "",
            "segments": [],
            "warning": f"ElevenLabs STT failed: {response.text}"
        }

    result = response.json()

    # Adjust this mapping to match the exact response you receive.
    transcript_text = result.get("text", "")
    segments = result.get("segments", [])

    return {
        "full_text": transcript_text,
        "segments": segments,
    }


def analyze_with_gradient(transcript_result: dict, pose_frames: list, summary_scores: dict):
    if not DO_GRADIENT_API_KEY:
        return {
            "final_feedback": "Gradient API key not set yet.",
            "timeline_feedback": [],
        }

    compact_pose_frames = pose_frames[:120]

    prompt = f"""
You are a pitch coach.

Return ONLY valid JSON with this exact shape:
{{
  "final_feedback": "string",
  "timeline_feedback": [
    {{
      "start_ms": 0,
      "end_ms": 0,
      "category": "posture|gesture|delivery|clarity|energy",
      "feedback": "string"
    }}
  ]
}}

Rules:
1. final_feedback must be 2 to 4 sentences total.
2. timeline_feedback must contain AT MOST 4 items.
3. Each timeline item must describe an IMPORTANT moment or broader time range, not tiny repetitive slices.
4. Do NOT repeat the same advice in multiple timeline items.
5. Merge similar nearby moments into one larger interval.
6. Keep each timeline feedback sentence short: max 20 words.
7. Focus on the most useful coaching, not exhaustive commentary.
8. If evidence is weak, be conservative and avoid making claims like "hand covering mouth" unless clearly supported.
9. Prefer broad actionable advice like gesture openness, posture balance, steadiness, pacing, and emphasis.
10. If there are no strong moment-specific insights, return an empty timeline_feedback array.

Use these inputs:

Summary scores:
{json.dumps(summary_scores)}

Transcript:
{json.dumps(transcript_result)}

Pose frames:
{json.dumps(compact_pose_frames)}

Create constructive, specific feedback for a speaker practicing a pitch.
"""

    headers = {
        "Authorization": f"Bearer {DO_GRADIENT_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DO_GRADIENT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert public speaking coach. "
                    "Return valid JSON only. "
                    "Be concise, non-repetitive, and conservative about uncertain claims."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    response = requests.post(
        f"{DO_GRADIENT_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )

    if response.status_code >= 400:
        return {
            "final_feedback": f"Gradient request failed: {response.text}",
            "timeline_feedback": [],
        }

    response_json = response.json()
    content = response_json["choices"][0]["message"]["content"]

    try:
        parsed_content = json.loads(content)

        timeline_feedback = parsed_content.get("timeline_feedback", [])
        if not isinstance(timeline_feedback, list):
            timeline_feedback = []

        # Hard cap in case the model ignores instructions
        timeline_feedback = timeline_feedback[:4]

        # Light cleanup for overly long feedback lines
        cleaned_timeline_feedback = []
        for item in timeline_feedback:
            if not isinstance(item, dict):
                continue

            cleaned_timeline_feedback.append({
                "start_ms": int(item.get("start_ms", 0)),
                "end_ms": int(item.get("end_ms", 0)),
                "category": item.get("category", "delivery"),
                "feedback": str(item.get("feedback", "")).strip()[:140],
            })

        final_feedback = str(parsed_content.get("final_feedback", "")).strip()
        final_feedback = final_feedback[:400]

        return {
            "final_feedback": final_feedback,
            "timeline_feedback": cleaned_timeline_feedback,
        }

    except Exception:
        return {
            "final_feedback": content[:400],
            "timeline_feedback": [],
        }


@app.post("/api/pitch/upload")
async def upload_pitch_session(
    video: UploadFile = File(...),
    pose_frames_json: str = Form(...),
    duration_ms: int = Form(...),
):
    if sessions_collection is None:
        return JSONResponse(
            {"error": "MongoDB is not configured. Set MONGO_URI first."},
            status_code=500,
        )

    file_suffix = os.path.splitext(video.filename)[1] if video.filename else ".webm"
    if not file_suffix:
        file_suffix = ".webm"

    pose_frames = json.loads(pose_frames_json)

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temporary_media_file:
        temporary_media_file.write(await video.read())
        temporary_media_path = temporary_media_file.name

    try:
        summary_scores = calculate_summary_scores(pose_frames)

        initial_document = {
            "created_at": datetime.now(timezone.utc),
            "duration_ms": duration_ms,
            "media_filename": os.path.basename(temporary_media_path),
            "media_mime_type": video.content_type,
            "pose_frames": pose_frames,
            "connections": CONNECTIONS,
            "summary_scores": summary_scores,
            "transcript": {
                "full_text": "",
                "segments": [],
            },
            "timeline_feedback": [],
            "final_feedback": "",
        }

        insert_result = sessions_collection.insert_one(initial_document)
        session_id = str(insert_result.inserted_id)

        transcript_result = transcribe_with_elevenlabs(temporary_media_path)
        gradient_result = analyze_with_gradient(transcript_result, pose_frames, summary_scores)

        sessions_collection.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$set": {
                    "transcript": transcript_result,
                    "timeline_feedback": gradient_result.get("timeline_feedback", []),
                    "final_feedback": gradient_result.get("final_feedback", ""),
                }
            },
        )

        response_document = sessions_collection.find_one({"_id": ObjectId(session_id)})

        return JSONResponse({
            "session_id": session_id,
            "duration_ms": response_document["duration_ms"],
            "connections": response_document["connections"],
            "summary_scores": response_document["summary_scores"],
            "transcript": response_document["transcript"],
            "timeline_feedback": response_document["timeline_feedback"],
            "final_feedback": response_document["final_feedback"],
            "pose_frames": response_document["pose_frames"],
        })

    finally:
        if os.path.exists(temporary_media_path):
            os.remove(temporary_media_path)


@app.get("/api/pitch/{session_id}")
async def get_pitch_session(session_id: str):
    if sessions_collection is None:
        return JSONResponse({"error": "MongoDB is not configured."}, status_code=500)

    document = sessions_collection.find_one({"_id": ObjectId(session_id)})
    if not document:
        return JSONResponse({"error": "Session not found."}, status_code=404)

    document["_id"] = str(document["_id"])
    return JSONResponse(document)

@app.get("/api/tts")
async def text_to_speech(text: str):
    if not ELEVENLABS_API_KEY:
        return JSONResponse({"error": "ElevenLabs API key not set"}, status_code=500)

    if not text or not text.strip():
        return JSONResponse({"error": "No text provided"}, status_code=400)
    voice_id = ELEVENLABS_VOICE_ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    safe_text = text.strip() 

    data = {
        "text": safe_text, # This was failing before
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=data, headers=headers, timeout=60)

    print("TTS status:", response.status_code)
    print("TTS body:", response.text[:1000])

    if response.status_code != 200:
        return JSONResponse(
            {"error": f"TTS failed: {response.status_code} {response.text}"},
            status_code=500
        )

    from fastapi.responses import Response
    return Response(content=response.content, media_type="audio/mpeg")




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)