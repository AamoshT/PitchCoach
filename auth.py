from fastapi import APIRouter, Form, Cookie, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY      = os.getenv("SECRET_KEY", "changeme_supersecret_key")
ADMIN_USERNAME  = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD  = os.getenv("ADMIN_PASSWORD", "pitchcoach123")
ALGORITHM       = "HS256"
TOKEN_EXPIRE_HOURS = 8

# ── Helpers ───────────────────────────────────────────────────────────────────
pwd_context     = CryptContext(schemes=["bcrypt"], deprecated="auto")
HASHED_PASSWORD = pwd_context.hash(ADMIN_PASSWORD[:72])

router = APIRouter()


def create_token(username: str) -> str:
    """Create a signed JWT token valid for TOKEN_EXPIRE_HOURS hours."""
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str = Cookie(default=None)):
    """
    Dependency — call this with Depends(verify_token) on any protected route.
    Raises a redirect to /login if the token is missing or invalid.
    """
    if not token:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    return token


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_page():
    """Serve the login HTML page."""
    with open("login.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.post("/login")
async def do_login(
    username: str = Form(...),
    password: str = Form(...),
):
    """Validate credentials, set cookie, redirect to app."""
    valid_user = username == ADMIN_USERNAME
    valid_pass = pwd_context.verify(password, HASHED_PASSWORD)

    if not valid_user or not valid_pass:
        # Redirect back to login with an error flag
        return RedirectResponse(url="/login?error=1", status_code=302)

    token    = create_token(username)
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key="token",
        value=token,
        httponly=True,       # JS cannot read this cookie
        samesite="lax",      # CSRF protection
        max_age=TOKEN_EXPIRE_HOURS * 3600,
    )
    return response


@router.get("/logout")
async def logout():
    """Clear the session cookie and redirect to login."""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("token")
    return response