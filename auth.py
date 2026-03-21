from fastapi import APIRouter, Form, Cookie, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY         = os.getenv("SECRET_KEY", "changeme_supersecret_key")
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 8

# ── Password hashing ──────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()

# ── MongoDB users collection (injected from app.py) ───────────────────────────
users_collection = None

def set_users_collection(collection):
    """Called from app.py to inject the MongoDB users collection."""
    global users_collection
    users_collection = collection


# ── Pydantic model for signup ─────────────────────────────────────────────────
class SignupRequest(BaseModel):
    fullname: str
    username: str
    email:    str
    password: str


# ── Token helpers ─────────────────────────────────────────────────────────────
def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str = Cookie(default=None)):
    if not token:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    return token


def get_current_username(token: str = Cookie(default=None)) -> str:
    if not token:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub", "")
    except JWTError:
        raise HTTPException(status_code=302, headers={"Location": "/login"})


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_page():
    with open("templates/login.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.get("/signup", response_class=HTMLResponse)
async def signup_page():
    with open("templates/signup.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.post("/login")
async def do_login(
    username: str = Form(...),
    password: str = Form(...),
):
    if users_collection is None:
        return RedirectResponse(url="/login?error=db", status_code=302)

    user = users_collection.find_one({"username": username})
    if not user:
        return RedirectResponse(url="/login?error=1", status_code=302)

    if not pwd_context.verify(password[:72], user["hashed_password"]):
        return RedirectResponse(url="/login?error=1", status_code=302)

    token    = create_token(username)
    response = RedirectResponse(url="/app", status_code=302)
    response.set_cookie(
        key="token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=TOKEN_EXPIRE_HOURS * 3600,
    )
    return response


@router.post("/api/auth/signup")
async def do_signup(data: SignupRequest):
    if users_collection is None:
        raise HTTPException(status_code=500, detail="Database not configured.")

    if users_collection.find_one({"username": data.username}):
        raise HTTPException(status_code=400, detail="Username already taken.")

    if users_collection.find_one({"email": data.email}):
        raise HTTPException(status_code=400, detail="Email already registered.")

    if len(data.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")

    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    hashed = pwd_context.hash(data.password[:72])
    users_collection.insert_one({
        "fullname":        data.fullname,
        "username":        data.username,
        "email":           data.email,
        "hashed_password": hashed,
        "created_at":      datetime.now(timezone.utc),
    })

    return JSONResponse({"message": "Account created successfully!"})


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("token")
    return response