from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

import os
import re
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Literal

import bcrypt
import jwt
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, status, UploadFile, File, Form, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, Response
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr

from emergentintegrations.llm.chat import LlmChat, UserMessage

# ---------- Config ----------
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

JWT_SECRET = os.environ['JWT_SECRET']
JWT_ALGORITHM = "HS256"
ACCESS_EXPIRY_DAYS = 30  # Long-lived for mobile UX
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

# ---------- App ----------
app = FastAPI(title="Operator — Breath & Wellness API")
api_router = APIRouter(prefix="/api")
bearer_scheme = HTTPBearer(auto_error=False)

# Mount static files (video + poster). Prefixed with /api/static so K8s
# ingress routes the request to the backend pod.
# STATIC_DIR can be overridden via env (e.g. Render persistent disk at /var/data/static).
STATIC_DIR = Path(os.environ.get("STATIC_DIR", str(ROOT_DIR / "static")))
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/api/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------- Helpers ----------
def _gen_referral_code(length: int = 6) -> str:
    """Generate a short uppercase alphanumeric (no I/O/0/1 to avoid confusion)."""
    import secrets
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=ACCESS_EXPIRY_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await db.users.find_one({"user_id": payload["sub"]}, {"_id": 0, "password_hash": 0})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ---------- Models ----------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    role: Optional[str] = Field(default="operator", description="operator | civilian")
    referral_code: Optional[str] = Field(default=None, max_length=8)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    user_id: str
    email: str
    name: str
    role: str = "operator"
    created_at: datetime
    referral_code: Optional[str] = None
    subscription_tier: str = "free"


class AuthResponse(BaseModel):
    token: str
    user: UserOut


# ---- Wave 3 models ----
class CrewCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=48)


class CrewJoinIn(BaseModel):
    code: str = Field(min_length=4, max_length=12)


class CustomTimerPhase(BaseModel):
    phase: str = Field(min_length=1, max_length=12)  # inhale|hold|exhale|hold_out
    seconds: int = Field(ge=1, le=120)


class CustomTimerIn(BaseModel):
    name: str = Field(min_length=1, max_length=48)
    pattern: List[CustomTimerPhase] = Field(min_length=2, max_length=8)
    cycles: int = Field(ge=1, le=200, default=10)


class SessionIn(BaseModel):
    technique_id: str
    technique_name: str
    duration_seconds: int = Field(ge=1)
    cycles_completed: int = Field(ge=0, default=0)
    stress_before: Optional[int] = Field(default=None, ge=1, le=10)
    stress_after: Optional[int] = Field(default=None, ge=1, le=10)
    preset_id: Optional[str] = None
    notes: Optional[str] = None


class SessionOut(SessionIn):
    session_id: str
    user_id: str
    created_at: datetime


class StatsOut(BaseModel):
    total_sessions: int
    total_minutes: float
    current_streak: int
    longest_streak: int
    avg_stress_delta: float
    last_session_at: Optional[datetime]
    freeze_available: bool = False
    freezes_this_week: int = 0


class CoachMessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    state: Optional[dict] = None  # e.g. {stress:7, energy:3, sleep:4}


class CoachMessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


# ---------- Wall Models ----------
class PostIn(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    technique_id: Optional[str] = None
    image_base64: Optional[str] = Field(default=None, max_length=2_500_000)
    is_anonymous: bool = False
    # Admin-only video attachment. Either reference an existing library video by
    # id, OR paste any external URL (YouTube/Vimeo/direct .mp4). Both ignored
    # for non-admin users.
    video_library_id: Optional[str] = None
    video_url: Optional[str] = Field(default=None, max_length=600)


class PostOut(BaseModel):
    post_id: str
    user_id: str
    display_name: str
    role: str
    is_anonymous: bool
    content: str
    technique_id: Optional[str] = None
    image_base64: Optional[str] = None
    like_count: int
    me_too_count: int
    comment_count: int
    created_at: datetime
    liked_by_me: bool = False
    me_too_by_me: bool = False
    is_owner: bool = False
    # Optional admin-attached video — present only when admin attached one.
    video_url: Optional[str] = None
    video_source_type: Optional[str] = None  # 'youtube' | 'vimeo' | 'mp4' | 'library'
    video_thumbnail_url: Optional[str] = None
    video_title: Optional[str] = None


class CommentIn(BaseModel):
    content: str = Field(min_length=1, max_length=1000)
    is_anonymous: bool = False


class CommentOut(BaseModel):
    comment_id: str
    post_id: str
    user_id: str
    display_name: str
    is_anonymous: bool
    content: str
    created_at: datetime
    is_owner: bool = False


class ReactionIn(BaseModel):
    type: Literal["like", "me_too"]


class ReportIn(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


class FavoriteIn(BaseModel):
    technique_id: str = Field(min_length=1, max_length=60)


class UserPrefsIn(BaseModel):
    default_mood: Optional[str] = Field(default=None, max_length=32)
    voice_enabled: Optional[bool] = None
    reminder_enabled: Optional[bool] = None
    reminder_hour: Optional[int] = Field(default=None, ge=0, le=23)
    reminder_minute: Optional[int] = Field(default=None, ge=0, le=59)
    role_kind: Optional[str] = Field(default=None, max_length=24)  # firefighter|ems|leo|military|civilian
    pain_point: Optional[str] = Field(default=None, max_length=24)  # sleep|anger|focus|panic|stress
    shift_pattern: Optional[str] = Field(default=None, max_length=32)  # kelly|24-48|24-72|48-96|custom|none
    shift_anchor_date: Optional[str] = Field(default=None, max_length=10)  # ISO YYYY-MM-DD
    shift_custom: Optional[str] = Field(default=None, max_length=64)  # e.g. "1,1,0,0,1,0,0"
    audio_only_default: Optional[bool] = None
    health_kit_enabled: Optional[bool] = None


class UserPrefsOut(BaseModel):
    default_mood: str = "tanpura"
    voice_enabled: bool = True
    reminder_enabled: bool = False
    reminder_hour: int = 8
    reminder_minute: int = 0
    onboarded: bool = False
    has_watched_intro: bool = False
    role_kind: Optional[str] = None
    pain_point: Optional[str] = None
    shift_pattern: Optional[str] = None
    shift_anchor_date: Optional[str] = None
    shift_custom: Optional[str] = None
    audio_only_default: bool = False
    health_kit_enabled: bool = False


class JournalIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    mood_after: Optional[int] = Field(default=None, ge=1, le=10)


class HrvIn(BaseModel):
    rmssd_ms: float = Field(ge=1.0, le=400.0)
    measured_at: Optional[datetime] = None
    context: Optional[str] = Field(default=None, max_length=32)  # pre_session|post_session|baseline


# ---------- Content Safety (Claude Haiku classifier) ----------
SAFETY_SYSTEM = (
    "You are a content safety classifier for a breathwork community for high-pressure operators. "
    "Respond with ONE WORD only.\n"
    "Reply SAFE for: general breathwork/wellness/operator experience, emotional distress, trauma, "
    "anxiety, venting about stressful calls, questions about practices.\n"
    "Reply UNSAFE only for: credible imminent self-harm or suicide intent, direct threats of violence "
    "toward identifiable people, child sexual content, illegal drug sales, spam/advertising, or "
    "recommending practices likely to cause immediate physical harm (e.g. breath-hold in water)."
)

SAFETY_REJECT_MSG = (
    "This post can't be shared. If you're in crisis, you are not alone — "
    "US/CA: call or text 988. UK: 116 123. AU: 13 11 14. Your breath, your life, matters."
)


async def classify_content(text: str) -> bool:
    """Return True if SAFE. Fail-open if LLM unavailable."""
    if not EMERGENT_LLM_KEY or not text.strip():
        return True
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"safety_{uuid.uuid4().hex[:10]}",
            system_message=SAFETY_SYSTEM,
        ).with_model("anthropic", "claude-haiku-4-5-20251001")
        resp = await chat.send_message(UserMessage(text=text[:2000]))
        return "UNSAFE" not in (resp or "").upper()
    except Exception:
        logger.exception("Safety classifier failed — failing open")
        return True


def _post_to_out(p: dict, uid: str, liked_set: set, me_too_set: set) -> PostOut:
    display = "Anonymous Operator" if p.get("is_anonymous") else p.get("author_name", "Operator")
    return PostOut(
        post_id=p["post_id"],
        user_id=p["user_id"],
        display_name=display,
        role=p.get("author_role", "operator"),
        is_anonymous=p.get("is_anonymous", False),
        content=p["content"],
        technique_id=p.get("technique_id"),
        image_base64=p.get("image_base64"),
        like_count=p.get("like_count", 0),
        me_too_count=p.get("me_too_count", 0),
        comment_count=p.get("comment_count", 0),
        created_at=p["created_at"],
        liked_by_me=p["post_id"] in liked_set,
        me_too_by_me=p["post_id"] in me_too_set,
        is_owner=p["user_id"] == uid,
        video_url=p.get("video_url"),
        video_source_type=p.get("video_source_type"),
        video_thumbnail_url=p.get("video_thumbnail_url"),
        video_title=p.get("video_title"),
    )


# ---------- Auth Routes ----------
@api_router.post("/auth/register", response_model=AuthResponse)
async def register(body: RegisterIn):
    email = body.email.lower().strip()
    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    # Generate a unique referral code
    for _ in range(8):
        code = _gen_referral_code(6)
        if not await db.users.find_one({"referral_code": code}):
            break
    # Resolve referrer if a code was provided
    referred_by = None
    if body.referral_code:
        ref = (body.referral_code or "").strip().upper()
        if ref:
            ru = await db.users.find_one({"referral_code": ref})
            if ru:
                referred_by = ru["user_id"]
    doc = {
        "user_id": user_id,
        "email": email,
        "name": body.name.strip(),
        "password_hash": hash_password(body.password),
        "role": body.role or "operator",
        "referral_code": code,
        "referred_by": referred_by,
        "subscription_tier": "free",
        "created_at": datetime.now(timezone.utc),
    }
    await db.users.insert_one(doc)
    if referred_by:
        await db.referrals.insert_one({
            "referrer_user_id": referred_by,
            "new_user_id": user_id,
            "code": code,
            "created_at": datetime.now(timezone.utc),
        })
    token = create_token(user_id, email)
    user_out = UserOut(
        user_id=user_id,
        email=email,
        name=doc["name"],
        role=doc["role"],
        created_at=doc["created_at"],
        referral_code=doc["referral_code"],
        subscription_tier=doc["subscription_tier"],
    )
    return AuthResponse(token=token, user=user_out)


@api_router.post("/auth/login", response_model=AuthResponse)
async def login(body: LoginIn):
    email = body.email.lower().strip()
    user = await db.users.find_one({"email": email})
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user["user_id"], email)
    user_out = UserOut(
        user_id=user["user_id"],
        email=user["email"],
        name=user.get("name", ""),
        role=user.get("role", "operator"),
        created_at=user["created_at"],
        referral_code=user.get("referral_code"),
        subscription_tier=user.get("subscription_tier", "free"),
    )
    return AuthResponse(token=token, user=user_out)


@api_router.get("/auth/me", response_model=UserOut)
async def me(current_user: dict = Depends(get_current_user)):
    return UserOut(
        user_id=current_user["user_id"],
        email=current_user["email"],
        name=current_user.get("name", ""),
        role=current_user.get("role", "operator"),
        created_at=current_user["created_at"],
        referral_code=current_user.get("referral_code"),
        subscription_tier=current_user.get("subscription_tier", "free"),
    )


# ---------- Account Deletion (GDPR / Apple required) ----------
class AccountDeletePayload(BaseModel):
    confirm: str  # must equal "DELETE" exactly


@api_router.delete("/auth/me")
async def delete_my_account(
    payload: AccountDeletePayload,
    current_user: dict = Depends(get_current_user),
):
    """Permanently purges the calling user and ALL associated data.

    This satisfies Apple's account-deletion requirement (App Store guideline
    5.1.1(v)) and GDPR right-to-erasure. Operation is irreversible — the
    client should require an explicit "DELETE" confirmation string.

    Admin accounts cannot self-delete (would orphan the admin role); use
    the operator account for testing this flow.
    """
    if (payload.confirm or "").strip() != "DELETE":
        raise HTTPException(
            status_code=400,
            detail='Confirmation phrase must be exactly "DELETE" to proceed.',
        )
    if current_user.get("role") == "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin accounts cannot self-delete. Contact platform support.",
        )

    uid = current_user["user_id"]
    email = current_user.get("email", "")

    # Tally what will be removed for the audit log.
    deleted_summary = {}

    # 1) Per-user collections — full purge.
    # Collection names verified against actual writes in this file: posts (wall),
    # session_journals, hrv_readings, etc. Wall likes/me_too are stored as
    # arrays *inside* each post doc (likes: [uid], me_too: [uid]) — handled
    # separately below. intro_watched lives on user_prefs.has_watched_intro.
    for col_name in (
        "sessions",
        "session_journals",
        "hrv_readings",
        "favorites",
        "posts",
        "coach_messages",
        "achievements",
        "user_prefs",
        "state_check_log",
        "custom_timers",
        "library_video_views",
        "ai_drills",
    ):
        try:
            res = await db[col_name].delete_many({"user_id": uid})
            if res.deleted_count:
                deleted_summary[col_name] = res.deleted_count
        except Exception as e:
            logger.warning(f"[delete_account] collection={col_name} err={e}")

    # 1b) Wall posts store reactions as arrays on the post document. Pull this
    # user's id out of every post's likes / me_too arrays so they don't leave
    # ghost-reactions on other users' posts.
    try:
        await db.posts.update_many(
            {"$or": [{"likes": uid}, {"me_too": uid}]},
            {"$pull": {"likes": uid, "me_too": uid}},
        )
    except Exception as e:
        logger.warning(f"[delete_account] reaction cleanup err={e}")

    # 2) Crew membership — leave any crew the user is in.
    try:
        crew_member_doc = await db.crew_members.find_one({"user_id": uid}, {"_id": 0, "crew_id": 1})
        if crew_member_doc:
            await db.crew_members.delete_many({"user_id": uid})
            deleted_summary["crew_members"] = 1
            crew_id = crew_member_doc.get("crew_id")
            if crew_id:
                # If the user was the leader of the crew, transfer leadership
                # to the next-oldest member. If they were the only member,
                # disband the crew entirely.
                #
                # Older crew docs may not have leader_id set — fall back to
                # created_by so legacy crews still transfer correctly.
                crew = await db.crews.find_one({"crew_id": crew_id}, {"_id": 0})
                is_leader = bool(
                    crew and (
                        crew.get("leader_id") == uid
                        or crew.get("created_by") == uid
                    )
                )
                if is_leader:
                    next_member = await db.crew_members.find_one(
                        {"crew_id": crew_id},
                        {"_id": 0, "user_id": 1},
                        sort=[("joined_at", 1)],
                    )
                    if next_member:
                        await db.crews.update_one(
                            {"crew_id": crew_id},
                            {"$set": {"leader_id": next_member["user_id"]}},
                        )
                        await db.crew_members.update_one(
                            {"crew_id": crew_id, "user_id": next_member["user_id"]},
                            {"$set": {"role": "leader"}},
                        )
                    else:
                        await db.crews.delete_one({"crew_id": crew_id})
                        deleted_summary["crews_disbanded"] = 1
    except Exception as e:
        logger.warning(f"[delete_account] crew cleanup err={e}")

    # 3) Referrals collection — delete any historical referral docs that
    # reference this user as either side. Recruits' user records are NOT
    # touched (their progress is preserved); we only null out the referred_by
    # pointer below.
    try:
        ref_res = await db.referrals.delete_many({
            "$or": [
                {"referrer_user_id": uid},
                {"new_user_id": uid},
            ],
        })
        if ref_res.deleted_count:
            deleted_summary["referrals"] = ref_res.deleted_count
    except Exception as e:
        logger.warning(f"[delete_account] referrals cleanup err={e}")

    # 3b) Null out recruits' referrer pointer.
    try:
        await db.users.update_many(
            {"referred_by": uid},
            {"$unset": {"referred_by": ""}},
        )
    except Exception as e:
        logger.warning(f"[delete_account] recruit pointer cleanup err={e}")

    # 4) Finally: remove the user record itself.
    res = await db.users.delete_one({"user_id": uid})
    deleted_summary["users"] = res.deleted_count

    logger.info(
        f"[account_delete] purged user={uid} email={email} "
        f"summary={deleted_summary}"
    )
    return {
        "status": "deleted",
        "user_id": uid,
        "summary": deleted_summary,
    }


# ---------- Achievements ----------
# Operator-themed achievement catalog. Each entry has a deterministic predicate
# evaluated against (sessions, stats, prefs, posts, coach_msgs) snapshot.
ACHIEVEMENT_CATALOG: List[dict] = [
    # Volume tier
    {"id": "first_watch", "title": "First Watch",
     "description": "Complete your first session.",
     "icon": "ShieldCheck", "tier": "bronze", "category": "volume"},
    {"id": "boots_on_ground", "title": "Boots on the Ground",
     "description": "Complete 5 sessions.",
     "icon": "Footprints", "tier": "bronze", "category": "volume"},
    {"id": "iron_lung", "title": "Iron Lung",
     "description": "Complete 25 sessions.",
     "icon": "Wind", "tier": "silver", "category": "volume"},
    {"id": "centurion", "title": "Centurion",
     "description": "Complete 100 sessions.",
     "icon": "Award", "tier": "gold", "category": "volume"},
    # Streak tier
    {"id": "standby", "title": "Standby",
     "description": "Hold a 7-day streak.",
     "icon": "Flame", "tier": "bronze", "category": "streak"},
    {"id": "long_watch", "title": "Long Watch",
     "description": "Hold a 30-day streak.",
     "icon": "Sunrise", "tier": "silver", "category": "streak"},
    {"id": "unbreakable", "title": "Unbreakable",
     "description": "Reach a 60-day longest streak.",
     "icon": "Anchor", "tier": "gold", "category": "streak"},
    # Technique mastery tier
    {"id": "cold_steel", "title": "Cold Steel",
     "description": "10 Wim Hof sessions.",
     "icon": "Snowflake", "tier": "silver", "category": "mastery"},
    {"id": "inner_fire", "title": "Inner Fire",
     "description": "10 Breath of Fire sessions.",
     "icon": "Flame", "tier": "silver", "category": "mastery"},
    {"id": "still_water", "title": "Still Water",
     "description": "10 4-7-8 sessions.",
     "icon": "Moon", "tier": "silver", "category": "mastery"},
    {"id": "kumbhaka_keeper", "title": "Kumbhaka Keeper",
     "description": "5 Rhythmic Kumbhaka sessions.",
     "icon": "Timer", "tier": "gold", "category": "mastery"},
    # Behaviour / engagement
    {"id": "morning_watch", "title": "Morning Watch",
     "description": "5 sessions started before 8 AM.",
     "icon": "Sun", "tier": "bronze", "category": "habit"},
    {"id": "night_watch", "title": "Night Watch",
     "description": "5 sessions started after 10 PM.",
     "icon": "MoonStar", "tier": "bronze", "category": "habit"},
    {"id": "briefed_in", "title": "Briefed In",
     "description": "Watch the operator's briefing.",
     "icon": "Eye", "tier": "bronze", "category": "habit"},
    {"id": "crew_member", "title": "Crew Member",
     "description": "Post on the Wall.",
     "icon": "Users", "tier": "bronze", "category": "habit"},
    {"id": "coach_op", "title": "Coach Op",
     "description": "Talk to the AI Coach.",
     "icon": "MessageSquare", "tier": "bronze", "category": "habit"},
]

ACHIEVEMENT_INDEX: dict = {a["id"]: a for a in ACHIEVEMENT_CATALOG}


async def _check_achievements(user_id: str) -> List[dict]:
    """Recompute and persist any newly-unlocked achievements for this user.
    Returns the list of achievements unlocked DURING this call (for toast UI).
    Idempotent — safe to call after every session log.
    """
    # Snapshot
    sessions = await db.sessions.find(
        {"user_id": user_id}, {"_id": 0}
    ).sort("created_at", -1).to_list(length=10000)
    total = len(sessions)

    # Per-technique counts
    by_tech: dict = {}
    morning = night = 0
    for s in sessions:
        tid = s.get("technique_id") or ""
        by_tech[tid] = by_tech.get(tid, 0) + 1
        ca = s.get("created_at")
        if isinstance(ca, str):
            try:
                ca = datetime.fromisoformat(ca)
            except Exception:
                ca = None
        if ca:
            if ca.tzinfo is None:
                ca = ca.replace(tzinfo=timezone.utc)
            hour = ca.astimezone(timezone.utc).hour
            if hour < 8:
                morning += 1
            if hour >= 22:
                night += 1

    # Streak
    created_ats = []
    for s in sessions:
        ca = s.get("created_at")
        if isinstance(ca, str):
            try:
                ca = datetime.fromisoformat(ca)
            except Exception:
                continue
        if ca and ca.tzinfo is None:
            ca = ca.replace(tzinfo=timezone.utc)
        if ca:
            created_ats.append(ca)
    cur, longest = _compute_streaks(created_ats)

    # Engagement
    prefs = await db.user_prefs.find_one({"user_id": user_id}, {"_id": 0}) or {}
    has_intro = bool(prefs.get("has_watched_intro"))
    posts_count = await db.posts.count_documents({"user_id": user_id})
    coach_count = await db.coach_messages.count_documents(
        {"user_id": user_id, "role": "user"}
    )

    # Predicate map
    triggered = {
        "first_watch": total >= 1,
        "boots_on_ground": total >= 5,
        "iron_lung": total >= 25,
        "centurion": total >= 100,
        "standby": cur >= 7 or longest >= 7,
        "long_watch": cur >= 30 or longest >= 30,
        "unbreakable": longest >= 60,
        "cold_steel": by_tech.get("wimhof", 0) >= 10,
        "inner_fire": by_tech.get("breath_of_fire", 0) >= 10,
        "still_water": by_tech.get("478", 0) >= 10,
        "kumbhaka_keeper": by_tech.get("rhythmic_kumbhaka", 0) >= 5,
        "morning_watch": morning >= 5,
        "night_watch": night >= 5,
        "briefed_in": has_intro,
        "crew_member": posts_count >= 1,
        "coach_op": coach_count >= 1,
    }

    existing_docs = await db.achievements.find(
        {"user_id": user_id}, {"_id": 0}
    ).to_list(length=200)
    existing = {d["achievement_id"]: d for d in existing_docs}

    newly_unlocked: List[dict] = []
    now = datetime.now(timezone.utc)
    for ach in ACHIEVEMENT_CATALOG:
        aid = ach["id"]
        if not triggered.get(aid):
            continue
        if aid in existing:
            continue
        doc = {
            "user_id": user_id,
            "achievement_id": aid,
            "unlocked_at": now,
        }
        try:
            await db.achievements.insert_one(doc)
            newly_unlocked.append({**ach, "unlocked_at": now})
        except Exception:
            # unique-index race — ignore
            pass
    return newly_unlocked


@api_router.get("/achievements")
async def list_achievements(current_user: dict = Depends(get_current_user)):
    """Full catalog with per-user unlocked status. Recomputes on read so
    legacy users immediately see their backfilled badges."""
    await _check_achievements(current_user["user_id"])
    docs = await db.achievements.find(
        {"user_id": current_user["user_id"]}, {"_id": 0}
    ).to_list(length=200)
    by_id = {d["achievement_id"]: d for d in docs}
    items = []
    for ach in ACHIEVEMENT_CATALOG:
        rec = by_id.get(ach["id"])
        items.append({
            **ach,
            "unlocked": rec is not None,
            "unlocked_at": rec["unlocked_at"] if rec else None,
        })
    unlocked_count = sum(1 for it in items if it["unlocked"])
    return {
        "items": items,
        "unlocked_count": unlocked_count,
        "total": len(items),
    }


# ---------- Sessions ----------
@api_router.post("/sessions", response_model=SessionOut)
async def log_session(body: SessionIn, current_user: dict = Depends(get_current_user)):
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    doc = body.model_dump()
    doc.update({
        "session_id": session_id,
        "user_id": current_user["user_id"],
        "created_at": datetime.now(timezone.utc),
    })
    await db.sessions.insert_one(doc)
    doc.pop("_id", None)
    # Fire-and-forget achievement check (await briefly so client gets fresh badges next call)
    try:
        await _check_achievements(current_user["user_id"])
    except Exception as e:
        logger.warning("Achievement check failed: %s", e)
    return SessionOut(**doc)


@api_router.get("/sessions", response_model=List[SessionOut])
async def list_sessions(current_user: dict = Depends(get_current_user), limit: int = 50):
    cursor = db.sessions.find(
        {"user_id": current_user["user_id"]}, {"_id": 0}
    ).sort("created_at", -1).limit(limit)
    items = await cursor.to_list(length=limit)
    return [SessionOut(**i) for i in items]


def _compute_streaks(dates: List[datetime], freeze_dates: Optional[set] = None) -> tuple[int, int]:
    """Given ordered DESC datetimes, compute (current_streak, longest_streak) in days.
    freeze_dates: set of date objects where the user claimed a streak freeze; a freeze
    covers a missing day without breaking the streak.
    """
    if not dates:
        return 0, 0
    freeze_dates = freeze_dates or set()
    day_set = sorted({d.astimezone(timezone.utc).date() for d in dates}, reverse=True)
    today = datetime.now(timezone.utc).date()
    current = 0
    if day_set[0] == today or day_set[0] == today - timedelta(days=1):
        expected = day_set[0]
        i = 0
        while i < len(day_set):
            if day_set[i] == expected:
                current += 1
                expected = expected - timedelta(days=1)
                i += 1
            elif expected in freeze_dates:
                # skip the missing day thanks to freeze
                current += 1
                expected = expected - timedelta(days=1)
            else:
                break
    longest = 1
    run = 1
    for i in range(1, len(day_set)):
        gap = (day_set[i - 1] - day_set[i]).days
        if gap == 1:
            run += 1
            longest = max(longest, run)
        elif gap == 2 and (day_set[i - 1] - timedelta(days=1)) in freeze_dates:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    return current, longest


@api_router.get("/stats", response_model=StatsOut)
async def stats(current_user: dict = Depends(get_current_user)):
    cursor = db.sessions.find(
        {"user_id": current_user["user_id"]}, {"_id": 0}
    ).sort("created_at", -1)
    items = await cursor.to_list(length=10000)
    # Gather freezes from last 30 days
    since = datetime.now(timezone.utc) - timedelta(days=30)
    freezes = await db.streak_freezes.find(
        {"user_id": current_user["user_id"], "claimed_for": {"$gte": since}},
        {"_id": 0},
    ).to_list(length=200)
    freeze_dates = set()
    for f in freezes:
        d = f["claimed_for"]
        if isinstance(d, str):
            d = datetime.fromisoformat(d)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        freeze_dates.add(d.date())
    week_start = datetime.now(timezone.utc) - timedelta(days=7)
    freezes_week = sum(
        1 for f in freezes
        if (f["claimed_for"].replace(tzinfo=timezone.utc) if f["claimed_for"].tzinfo is None else f["claimed_for"]) >= week_start
    )
    freeze_available = freezes_week < 1
    if not items:
        return StatsOut(
            total_sessions=0, total_minutes=0.0, current_streak=0,
            longest_streak=0, avg_stress_delta=0.0, last_session_at=None,
            freeze_available=freeze_available, freezes_this_week=freezes_week,
        )
    total_sessions = len(items)
    total_seconds = sum(i.get("duration_seconds", 0) for i in items)
    deltas = [
        (i["stress_before"] - i["stress_after"])
        for i in items
        if i.get("stress_before") is not None and i.get("stress_after") is not None
    ]
    avg_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    created_ats = []
    for i in items:
        ca = i.get("created_at")
        if isinstance(ca, str):
            ca = datetime.fromisoformat(ca)
        if ca.tzinfo is None:
            ca = ca.replace(tzinfo=timezone.utc)
        created_ats.append(ca)
    current, longest = _compute_streaks(created_ats, freeze_dates)
    return StatsOut(
        total_sessions=total_sessions,
        total_minutes=round(total_seconds / 60.0, 1),
        current_streak=current,
        longest_streak=longest,
        avg_stress_delta=round(avg_delta, 2),
        last_session_at=created_ats[0] if created_ats else None,
        freeze_available=freeze_available,
        freezes_this_week=freezes_week,
    )


@api_router.get("/stats/extended")
async def stats_extended(current_user: dict = Depends(get_current_user)):
    """Per-day session counts for last 60 days, per-technique + per-category
    breakdown, and an achievement count. Drives the upgraded Progress dashboard
    + the home-screen heatmap."""
    sessions = await db.sessions.find(
        {"user_id": current_user["user_id"]}, {"_id": 0}
    ).sort("created_at", -1).to_list(length=10000)
    # Heatmap (last 60 days, UTC)
    today = datetime.now(timezone.utc).date()
    counts: dict = {}
    for s in sessions:
        ca = s.get("created_at")
        if isinstance(ca, str):
            try:
                ca = datetime.fromisoformat(ca)
            except Exception:
                continue
        if ca and ca.tzinfo is None:
            ca = ca.replace(tzinfo=timezone.utc)
        if not ca:
            continue
        d = ca.astimezone(timezone.utc).date()
        if (today - d).days <= 60:
            key = d.isoformat()
            counts[key] = counts.get(key, 0) + 1
    heatmap = []
    for i in range(59, -1, -1):
        d = today - timedelta(days=i)
        heatmap.append({"date": d.isoformat(), "count": counts.get(d.isoformat(), 0)})
    # Technique breakdown
    by_tech: dict = {}
    for s in sessions:
        tid = s.get("technique_id") or "unknown"
        tname = s.get("technique_name") or tid
        bucket = by_tech.setdefault(tid, {"technique_id": tid, "technique_name": tname,
                                           "count": 0, "minutes": 0.0})
        bucket["count"] += 1
        bucket["minutes"] += (s.get("duration_seconds") or 0) / 60.0
    technique_breakdown = sorted(
        [{**v, "minutes": round(v["minutes"], 1)} for v in by_tech.values()],
        key=lambda x: x["count"], reverse=True,
    )
    # Achievement progress
    ach_count = await db.achievements.count_documents(
        {"user_id": current_user["user_id"]}
    )
    return {
        "heatmap": heatmap,
        "technique_breakdown": technique_breakdown,
        "achievements_unlocked": ach_count,
        "achievements_total": len(ACHIEVEMENT_CATALOG),
    }


@api_router.post("/stats/freeze")
async def claim_streak_freeze(current_user: dict = Depends(get_current_user)):
    """Claim a 1-day streak freeze for yesterday (to cover a missed day).
    Max 1 per rolling 7 days.
    """
    now = datetime.now(timezone.utc)
    week_start = now - timedelta(days=7)
    recent = await db.streak_freezes.find(
        {"user_id": current_user["user_id"], "claimed_for": {"$gte": week_start}},
        {"_id": 0},
    ).to_list(length=10)
    if recent:
        raise HTTPException(409, "Streak freeze already used this week")
    claimed_for = (now - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    await db.streak_freezes.insert_one({
        "user_id": current_user["user_id"],
        "claimed_for": claimed_for,
        "created_at": now,
    })
    return {"status": "claimed", "claimed_for": claimed_for}


# ---------- Favorites ----------
@api_router.get("/favorites", response_model=List[str])
async def list_favorites(current_user: dict = Depends(get_current_user)):
    cursor = db.favorites.find({"user_id": current_user["user_id"]}, {"_id": 0})
    items = await cursor.to_list(length=200)
    return [i["technique_id"] for i in items]


@api_router.post("/favorites")
async def add_favorite(body: FavoriteIn, current_user: dict = Depends(get_current_user)):
    await db.favorites.update_one(
        {"user_id": current_user["user_id"], "technique_id": body.technique_id},
        {"$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return {"status": "ok"}


@api_router.delete("/favorites/{technique_id}")
async def remove_favorite(technique_id: str, current_user: dict = Depends(get_current_user)):
    await db.favorites.delete_one(
        {"user_id": current_user["user_id"], "technique_id": technique_id}
    )
    return {"status": "ok"}


# ---------- User Preferences ----------
@api_router.get("/prefs", response_model=UserPrefsOut)
async def get_prefs(current_user: dict = Depends(get_current_user)):
    doc = await db.user_prefs.find_one({"user_id": current_user["user_id"]}, {"_id": 0})
    if not doc:
        return UserPrefsOut()
    return UserPrefsOut(
        default_mood=doc.get("default_mood", "tanpura"),
        voice_enabled=doc.get("voice_enabled", True),
        reminder_enabled=doc.get("reminder_enabled", False),
        reminder_hour=doc.get("reminder_hour", 8),
        reminder_minute=doc.get("reminder_minute", 0),
        onboarded=doc.get("onboarded", False),
        has_watched_intro=doc.get("has_watched_intro", False),
        role_kind=doc.get("role_kind"),
        pain_point=doc.get("pain_point"),
        shift_pattern=doc.get("shift_pattern"),
        shift_anchor_date=doc.get("shift_anchor_date"),
        shift_custom=doc.get("shift_custom"),
        audio_only_default=doc.get("audio_only_default", False),
        health_kit_enabled=doc.get("health_kit_enabled", False),
    )


@api_router.put("/prefs", response_model=UserPrefsOut)
async def update_prefs(body: UserPrefsIn, current_user: dict = Depends(get_current_user)):
    update = {k: v for k, v in body.model_dump().items() if v is not None}
    if update:
        await db.user_prefs.update_one(
            {"user_id": current_user["user_id"]},
            {"$set": update, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
    return await get_prefs(current_user)


@api_router.post("/prefs/onboarded")
async def mark_onboarded(current_user: dict = Depends(get_current_user)):
    await db.user_prefs.update_one(
        {"user_id": current_user["user_id"]},
        {"$set": {"onboarded": True}},
        upsert=True,
    )
    return {"status": "ok"}


# ---------- Session Journal (post-session debrief) ----------
@api_router.post("/sessions/{session_id}/journal")
async def write_session_journal(
    session_id: str, body: JournalIn, current_user: dict = Depends(get_current_user),
):
    sess = await db.sessions.find_one(
        {"session_id": session_id, "user_id": current_user["user_id"]},
        {"_id": 0},
    )
    if not sess:
        raise HTTPException(404, "Session not found")
    doc = {
        "user_id": current_user["user_id"],
        "session_id": session_id,
        "text": body.text.strip(),
        "mood_after": body.mood_after,
        "created_at": datetime.now(timezone.utc),
    }
    await db.session_journals.insert_one(doc)
    doc.pop("_id", None)
    return {"status": "ok", "journal": doc}


@api_router.get("/sessions/{session_id}/journal")
async def get_session_journal(
    session_id: str, current_user: dict = Depends(get_current_user),
):
    items = await db.session_journals.find(
        {"session_id": session_id, "user_id": current_user["user_id"]},
        {"_id": 0},
    ).sort("created_at", -1).to_list(length=10)
    return {"items": items}


# ---------- HRV (Apple Health stub) ----------
@api_router.post("/health/hrv")
async def log_hrv(body: HrvIn, current_user: dict = Depends(get_current_user)):
    measured_at = body.measured_at or datetime.now(timezone.utc)
    if measured_at.tzinfo is None:
        measured_at = measured_at.replace(tzinfo=timezone.utc)
    await db.hrv_readings.insert_one({
        "user_id": current_user["user_id"],
        "rmssd_ms": body.rmssd_ms,
        "measured_at": measured_at,
        "context": body.context,
        "created_at": datetime.now(timezone.utc),
    })
    return {"status": "ok"}


@api_router.get("/health/hrv")
async def list_hrv(current_user: dict = Depends(get_current_user), limit: int = 30):
    items = await db.hrv_readings.find(
        {"user_id": current_user["user_id"]}, {"_id": 0},
    ).sort("measured_at", -1).limit(limit).to_list(length=limit)
    return {"items": items}


# ---------- Crew Mode ----------
async def _crew_member_summary(user_id: str) -> dict:
    """Return name + streak summary for a member (used in leaderboards)."""
    u = await db.users.find_one({"user_id": user_id}, {"_id": 0, "password_hash": 0}) or {}
    sessions = await db.sessions.find(
        {"user_id": user_id}, {"_id": 0, "created_at": 1},
    ).sort("created_at", -1).to_list(length=1000)
    cas = []
    for s in sessions:
        ca = s.get("created_at")
        if isinstance(ca, str):
            try:
                ca = datetime.fromisoformat(ca)
            except Exception:
                continue
        if ca and ca.tzinfo is None:
            ca = ca.replace(tzinfo=timezone.utc)
        if ca:
            cas.append(ca)
    cur, longest = _compute_streaks(cas)
    return {
        "user_id": user_id,
        "name": u.get("name", "Operator"),
        "current_streak": cur,
        "longest_streak": longest,
        "total_sessions": len(sessions),
    }


@api_router.post("/crews")
async def create_crew(body: CrewCreateIn, current_user: dict = Depends(get_current_user)):
    # If user already in a crew, reject (one crew per user for v1)
    existing = await db.crew_members.find_one({"user_id": current_user["user_id"]})
    if existing:
        raise HTTPException(409, "You're already in a crew. Leave it first.")
    # Generate unique crew code
    for _ in range(8):
        code = _gen_referral_code(6)
        if not await db.crews.find_one({"code": code}):
            break
    crew_id = f"crew_{uuid.uuid4().hex[:10]}"
    now = datetime.now(timezone.utc)
    await db.crews.insert_one({
        "crew_id": crew_id,
        "name": body.name.strip(),
        "code": code,
        "created_by": current_user["user_id"],
        "leader_id": current_user["user_id"],
        "created_at": now,
    })
    await db.crew_members.insert_one({
        "crew_id": crew_id,
        "user_id": current_user["user_id"],
        "role": "leader",
        "joined_at": now,
    })
    return {"crew_id": crew_id, "name": body.name.strip(), "code": code, "role": "leader"}


@api_router.post("/crews/join")
async def join_crew(body: CrewJoinIn, current_user: dict = Depends(get_current_user)):
    existing = await db.crew_members.find_one({"user_id": current_user["user_id"]})
    if existing:
        raise HTTPException(409, "Leave your current crew first.")
    code = body.code.strip().upper()
    crew = await db.crews.find_one({"code": code}, {"_id": 0})
    if not crew:
        raise HTTPException(404, "No crew with that code.")
    await db.crew_members.insert_one({
        "crew_id": crew["crew_id"],
        "user_id": current_user["user_id"],
        "role": "member",
        "joined_at": datetime.now(timezone.utc),
    })
    return {"crew_id": crew["crew_id"], "name": crew["name"], "code": crew["code"], "role": "member"}


@api_router.post("/crews/leave")
async def leave_crew(current_user: dict = Depends(get_current_user)):
    res = await db.crew_members.delete_one({"user_id": current_user["user_id"]})
    return {"status": "ok", "left": res.deleted_count > 0}


@api_router.get("/crews/me")
async def get_my_crew(current_user: dict = Depends(get_current_user)):
    member = await db.crew_members.find_one(
        {"user_id": current_user["user_id"]}, {"_id": 0},
    )
    if not member:
        return {"in_crew": False, "crew": None, "members": []}
    crew = await db.crews.find_one({"crew_id": member["crew_id"]}, {"_id": 0})
    if not crew:
        return {"in_crew": False, "crew": None, "members": []}
    member_docs = await db.crew_members.find(
        {"crew_id": crew["crew_id"]}, {"_id": 0},
    ).to_list(length=200)
    members = []
    for md in member_docs:
        s = await _crew_member_summary(md["user_id"])
        members.append({**s, "role": md.get("role", "member")})
    members.sort(key=lambda m: (m["current_streak"], m["total_sessions"]), reverse=True)
    return {
        "in_crew": True,
        "crew": {
            **crew,
            "member_count": len(members),
            "my_role": member.get("role", "member"),
        },
        "members": members,
    }


# ---------- Referrals ----------
@api_router.get("/referrals/me")
async def my_referrals(current_user: dict = Depends(get_current_user)):
    code = current_user.get("referral_code")
    if not code:
        # Backfill code for legacy accounts
        for _ in range(8):
            code = _gen_referral_code(6)
            if not await db.users.find_one({"referral_code": code}):
                break
        await db.users.update_one(
            {"user_id": current_user["user_id"]},
            {"$set": {"referral_code": code}},
        )
    referrals = await db.referrals.find(
        {"referrer_user_id": current_user["user_id"]}, {"_id": 0},
    ).sort("created_at", -1).to_list(length=200)
    # Hydrate names
    items = []
    for r in referrals:
        u = await db.users.find_one(
            {"user_id": r["new_user_id"]}, {"_id": 0, "name": 1, "created_at": 1},
        ) or {}
        items.append({
            "name": u.get("name", "Operator"),
            "joined_at": u.get("created_at", r.get("created_at")),
        })
    return {"code": code, "count": len(items), "items": items}


# ---------- Custom Timers (Interval Builder) ----------
@api_router.post("/custom-timers")
async def create_custom_timer(
    body: CustomTimerIn, current_user: dict = Depends(get_current_user),
):
    timer_id = f"ct_{uuid.uuid4().hex[:10]}"
    pattern = [p.model_dump() for p in body.pattern]
    one_cycle_secs = sum(p["seconds"] for p in pattern)
    total_seconds = one_cycle_secs * body.cycles
    doc = {
        "timer_id": timer_id,
        "user_id": current_user["user_id"],
        "name": body.name.strip(),
        "pattern": pattern,
        "cycles": body.cycles,
        "total_seconds": total_seconds,
        "created_at": datetime.now(timezone.utc),
    }
    await db.custom_timers.insert_one(doc)
    doc.pop("_id", None)
    return doc


@api_router.get("/custom-timers")
async def list_custom_timers(current_user: dict = Depends(get_current_user)):
    items = await db.custom_timers.find(
        {"user_id": current_user["user_id"]}, {"_id": 0},
    ).sort("created_at", -1).to_list(length=100)
    return {"items": items}


@api_router.delete("/custom-timers/{timer_id}")
async def delete_custom_timer(timer_id: str, current_user: dict = Depends(get_current_user)):
    res = await db.custom_timers.delete_one(
        {"timer_id": timer_id, "user_id": current_user["user_id"]},
    )
    if res.deleted_count == 0:
        raise HTTPException(404, "Not found")
    return {"status": "ok"}


# ---------- Subscription (scaffolding only — no payment integration yet) ----------
PRO_FEATURES = [
    "Unlimited AI Coach messages",
    "Apple Health HRV integration",
    "Crew Mode + Firehouse Leaderboard",
    "Custom Interval Builder (save unlimited)",
    "All Programs (multi-day series)",
    "Audio-only Eyes-Closed sessions",
    "Lock screen quick-start widget",
    "Streak Grace Day (2× per week vs 1×)",
]
FREE_FEATURES = [
    "All 13 breath techniques",
    "Daily 60-second reset",
    "Streak + Achievements",
    "Mission selection + State Check",
    "AI Coach (5 msgs/day)",
    "Community Wall",
]


@api_router.get("/subscription/status")
async def subscription_status(current_user: dict = Depends(get_current_user)):
    tier = current_user.get("subscription_tier", "free")
    return {
        "tier": tier,
        "is_pro": tier == "pro",
        "free_features": FREE_FEATURES,
        "pro_features": PRO_FEATURES,
    }


@api_router.post("/subscription/grant-pro")
async def grant_pro(current_user: dict = Depends(get_current_user)):
    """DEV/ADMIN scaffolding endpoint. Real upgrade flow will go through
    Apple/Google IAP or Stripe before launch."""
    await db.users.update_one(
        {"user_id": current_user["user_id"]},
        {"$set": {"subscription_tier": "pro",
                  "pro_granted_at": datetime.now(timezone.utc)}},
    )
    return {"status": "ok", "tier": "pro"}


@api_router.post("/subscription/revoke-pro")
async def revoke_pro(current_user: dict = Depends(get_current_user)):
    await db.users.update_one(
        {"user_id": current_user["user_id"]},
        {"$set": {"subscription_tier": "free"}},
    )
    return {"status": "ok", "tier": "free"}


# ---------- Wave 3 indexes (added to startup) ----------


# ---------- Intro Video ----------
INTRO_VIDEO_PATH = STATIC_DIR / "videos" / "intro.mp4"
INTRO_WEBM_PATH = STATIC_DIR / "videos" / "intro.webm"
INTRO_POSTER_PATH = STATIC_DIR / "videos" / "intro_poster.jpg"


def _range_video_response(file_path: Path, request: Request, media_type: str = "video/mp4") -> Response:
    """Serve a video file with HTTP Range support so <video> can seek/stream."""
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")
    ua = request.headers.get("user-agent", "")[:80]
    logger.info(f"[video] {file_path.name} Range={range_header!r} UA={ua!r}")

    # iOS native AVPlayer (CoreMedia) strictly validates Content-Range and
    # rejects responses smaller than what it requested. Web browsers (Safari,
    # Chrome) tolerate chunked sub-responses and re-issue Range requests. So
    # we detect the native player and skip our 4MB cap for it.
    is_native_avplayer = (
        "AppleCoreMedia" in ua
        or "CFNetwork" in ua
        or ua.startswith("stagefright")  # Android ExoPlayer fallback
    )

    # No range header → return full file with Accept-Ranges so browser can seek
    if not range_header:
        return FileResponse(
            str(file_path),
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=86400"},
        )

    # Parse "bytes=start-end"
    try:
        units, _, rng = range_header.partition("=")
        if units.strip().lower() != "bytes":
            raise ValueError("unsupported unit")
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        # Cap any open-ended or oversized range to MAX_RANGE_BYTES for WEB clients
        # only. Web Safari tends to request "bytes=N-" (open-ended) which makes a
        # single response try to push tens of MB through the proxy. With
        # cellular / lossy networks the connection dies mid-stream. Capping
        # forces the web player to issue a fresh Range request every few seconds.
        # iOS NATIVE AVPlayer (TestFlight) must NOT be capped — it rejects any
        # Content-Range that doesn't match exactly what it asked for.
        MAX_RANGE_BYTES = file_size if is_native_avplayer else (4 * 1024 * 1024)
        client_end = int(end_s) if end_s else file_size - 1
        client_end = min(client_end, file_size - 1)
        end = min(client_end, start + MAX_RANGE_BYTES - 1)
        if start > end or start < 0:
            raise ValueError("invalid range")
    except ValueError:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    length = end - start + 1
    chunk_size = 1024 * 256  # 256KB chunks

    def iterator():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                data = f.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    from starlette.responses import StreamingResponse
    return StreamingResponse(
        iterator(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            # Aggressive edge caching — videos are immutable assets so cache for
            # 7 days at Cloudflare edge. Survives backend restarts which makes
            # video playback resilient to platform pod cycling.
            "Cache-Control": "public, max-age=604800, s-maxage=604800, immutable",
            "CDN-Cache-Control": "public, max-age=604800",
            "Cloudflare-CDN-Cache-Control": "public, max-age=604800",
        },
    )


@api_router.api_route("/intro/video", methods=["GET", "HEAD"])
async def stream_intro_video(request: Request):
    if not INTRO_VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    file_size = INTRO_VIDEO_PATH.stat().st_size
    if request.method == "HEAD":
        return Response(
            status_code=200,
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "public, max-age=86400",
            },
        )
    return _range_video_response(INTRO_VIDEO_PATH, request, media_type="video/mp4")


@api_router.api_route("/intro/video.webm", methods=["GET", "HEAD"])
async def stream_intro_video_webm(request: Request):
    if not INTRO_WEBM_PATH.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    file_size = INTRO_WEBM_PATH.stat().st_size
    if request.method == "HEAD":
        return Response(
            status_code=200,
            media_type="video/webm",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "public, max-age=86400",
            },
        )
    return _range_video_response(INTRO_WEBM_PATH, request, media_type="video/webm")


@api_router.api_route("/intro/poster", methods=["GET", "HEAD"])
async def intro_poster(request: Request):
    if not INTRO_POSTER_PATH.exists():
        raise HTTPException(status_code=404, detail="Poster not found")
    if request.method == "HEAD":
        size = INTRO_POSTER_PATH.stat().st_size
        return Response(
            status_code=200,
            media_type="image/jpeg",
            headers={
                "Content-Length": str(size),
                "Cache-Control": "public, max-age=86400",
            },
        )
    return FileResponse(
        str(INTRO_POSTER_PATH),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@api_router.get("/intro/meta")
async def intro_meta(current_user: dict = Depends(get_current_user)):
    available = INTRO_VIDEO_PATH.exists()
    prefs_doc = await db.user_prefs.find_one(
        {"user_id": current_user["user_id"]}, {"_id": 0, "has_watched_intro": 1}
    )
    has_watched = bool(prefs_doc and prefs_doc.get("has_watched_intro", False))
    return {
        "available": available,
        "has_watched": has_watched,
        "video_url": "/api/intro/video" if available else None,
        "video_url_webm": "/api/intro/video.webm" if INTRO_WEBM_PATH.exists() else None,
        "poster_url": "/api/intro/poster" if INTRO_POSTER_PATH.exists() else None,
        "duration_seconds": 177,
        "title": "A Message from the Operator",
    }


@api_router.post("/intro/watched")
async def mark_intro_watched(current_user: dict = Depends(get_current_user)):
    await db.user_prefs.update_one(
        {"user_id": current_user["user_id"]},
        {"$set": {"has_watched_intro": True, "intro_watched_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return {"status": "ok"}


# ---------- State / Mission Videos (admin-managed) ----------
STATE_VIDEOS_DIR = STATIC_DIR / "videos" / "states"
STATE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_STATE_IDS = {
    "overwhelmed", "anxious", "low_energy", "scattered",
    "heart_racing", "unbalanced", "racing_thoughts", "good",
}
ALLOWED_MISSION_IDS = {
    "sleep", "calm", "lock_in", "energize", "recover", "perform",
}
ALLOWED_VIDEO_MIMES = {
    "video/mp4", "video/quicktime", "video/webm",
    "video/x-m4v", "video/mpeg",
}
MAX_STATE_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB hard cap


def _slot_key(state_id: Optional[str], mission_id: Optional[str]) -> str:
    """Stable key that identifies a video slot. Used as both Mongo _id and filename stem."""
    parts = []
    if state_id:
        parts.append(f"state-{state_id}")
    if mission_id:
        parts.append(f"mission-{mission_id}")
    if not parts:
        raise ValueError("must supply state_id and/or mission_id")
    return "__".join(parts)


def _ext_from_mime(mime: str) -> str:
    mapping = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/x-m4v": ".m4v",
        "video/mpeg": ".mpg",
    }
    return mapping.get(mime, ".mp4")


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if (current_user.get("role") or "").lower() != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return current_user


def _serialize_state_video(doc: dict) -> dict:
    return {
        "slot_id": doc.get("slot_id"),
        "state_id": doc.get("state_id"),
        "mission_id": doc.get("mission_id"),
        "title": doc.get("title"),
        "filename": doc.get("filename"),
        "content_type": doc.get("content_type"),
        "size": doc.get("size"),
        "video_url": f"/api/state-videos/file/{doc['filename']}" if doc.get("filename") else None,
        "uploaded_at": doc.get("uploaded_at").isoformat() if doc.get("uploaded_at") else None,
        "uploaded_by": doc.get("uploaded_by"),
    }


@api_router.post("/admin/state-videos/upload")
async def admin_upload_state_video(
    request: Request,
    file: UploadFile = File(...),
    state_id: Optional[str] = Form(None),
    mission_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    admin: dict = Depends(require_admin),
):
    # Validate identifiers
    if state_id and state_id not in ALLOWED_STATE_IDS:
        raise HTTPException(status_code=400, detail=f"Unknown state_id '{state_id}'")
    if mission_id and mission_id not in ALLOWED_MISSION_IDS:
        raise HTTPException(status_code=400, detail=f"Unknown mission_id '{mission_id}'")
    if not state_id and not mission_id:
        raise HTTPException(status_code=400, detail="Provide state_id and/or mission_id")

    # Validate mime
    mime = (file.content_type or "").lower()
    if mime not in ALLOWED_VIDEO_MIMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type: {mime}. Use mp4, mov, m4v or webm.",
        )

    slot_id = _slot_key(state_id, mission_id)
    ext = _ext_from_mime(mime)
    filename = f"{slot_id}{ext}"
    target_path = STATE_VIDEOS_DIR / filename

    # If a previous file existed under a different ext, remove it.
    for old in STATE_VIDEOS_DIR.glob(f"{slot_id}.*"):
        if old.name != filename:
            try:
                old.unlink()
            except OSError:
                pass

    # Stream-write to disk in chunks (cap memory usage on large uploads)
    total = 0
    chunk_size = 1024 * 1024  # 1 MB
    try:
        with open(target_path, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_STATE_VIDEO_BYTES:
                    out.close()
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File too large (max 500 MB)")
                out.write(chunk)
    finally:
        await file.close()

    doc = {
        "slot_id": slot_id,
        "state_id": state_id,
        "mission_id": mission_id,
        "title": title or None,
        "filename": filename,
        "content_type": mime,
        "size": total,
        "uploaded_at": datetime.now(timezone.utc),
        "uploaded_by": admin.get("email"),
    }
    await db.state_videos.update_one(
        {"slot_id": slot_id},
        {"$set": doc},
        upsert=True,
    )
    logger.info(f"[state-video] admin {admin.get('email')} uploaded slot={slot_id} bytes={total}")
    return _serialize_state_video(doc)


@api_router.get("/admin/state-videos")
async def admin_list_state_videos(admin: dict = Depends(require_admin)):
    """Return every configured slot AND empty placeholders for slots not yet filled."""
    cursor = db.state_videos.find({}, {"_id": 0})
    existing = {d["slot_id"]: d async for d in cursor}

    slots = []
    # State-only slots
    for sid in [
        "overwhelmed", "anxious", "low_energy", "scattered",
        "heart_racing", "unbalanced", "racing_thoughts", "good",
    ]:
        key = f"state-{sid}"
        if key in existing:
            slots.append(_serialize_state_video(existing[key]))
        else:
            slots.append({
                "slot_id": key, "state_id": sid, "mission_id": None,
                "filename": None, "video_url": None, "size": None,
                "content_type": None, "title": None, "uploaded_at": None,
                "uploaded_by": None,
            })
    # Mission-only slots
    for mid in ["sleep", "calm", "lock_in", "energize", "recover", "perform"]:
        key = f"mission-{mid}"
        if key in existing:
            slots.append(_serialize_state_video(existing[key]))
        else:
            slots.append({
                "slot_id": key, "state_id": None, "mission_id": mid,
                "filename": None, "video_url": None, "size": None,
                "content_type": None, "title": None, "uploaded_at": None,
                "uploaded_by": None,
            })
    # Any combo slots present
    for k, d in existing.items():
        if k.startswith("state-") and "__mission-" in k:
            slots.append(_serialize_state_video(d))

    return {"slots": slots}


@api_router.delete("/admin/state-videos/{slot_id}")
async def admin_delete_state_video(slot_id: str, admin: dict = Depends(require_admin)):
    doc = await db.state_videos.find_one({"slot_id": slot_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Slot not configured")
    fname = doc.get("filename")
    if fname:
        fpath = STATE_VIDEOS_DIR / fname
        try:
            fpath.unlink(missing_ok=True)
        except OSError:
            pass
    await db.state_videos.delete_one({"slot_id": slot_id})
    return {"status": "deleted", "slot_id": slot_id}


@api_router.get("/state-videos/lookup")
async def state_video_lookup(
    state: Optional[str] = None,
    mission: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """Best-match lookup: state+mission combo > state-only > mission-only > none."""
    candidates: List[str] = []
    if state and mission:
        candidates.append(f"state-{state}__mission-{mission}")
    if state:
        candidates.append(f"state-{state}")
    if mission:
        candidates.append(f"mission-{mission}")

    for slot_id in candidates:
        doc = await db.state_videos.find_one({"slot_id": slot_id}, {"_id": 0})
        if doc and doc.get("filename"):
            return {"matched": True, "slot": _serialize_state_video(doc)}

    return {"matched": False, "slot": None}


@api_router.api_route("/state-videos/file/{filename}", methods=["GET", "HEAD"])
async def stream_state_video(filename: str, request: Request):
    # Sanitize: filename must not contain path separators
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = STATE_VIDEOS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    ext = path.suffix.lower()
    media_type = {
        ".mp4": "video/mp4", ".m4v": "video/mp4",
        ".mov": "video/quicktime", ".webm": "video/webm",
        ".mpg": "video/mpeg",
    }.get(ext, "application/octet-stream")
    return _range_video_response(path, request, media_type=media_type)


# ---------- Reset Videos (Operator / Civilian 60-Second Reset) ----------
RESET_VIDEOS_DIR = STATIC_DIR / "videos" / "reset"
RESET_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

_RESET_VARIANTS = {
    "operator": "operator-reset.mp4",
    "civilian": "civilian-reset.mp4",
}


@api_router.api_route("/reset-videos/{variant}", methods=["GET", "HEAD"])
async def stream_reset_video(variant: str, request: Request):
    filename = _RESET_VARIANTS.get(variant.lower())
    if not filename:
        raise HTTPException(status_code=404, detail="Unknown reset variant")
    path = RESET_VIDEOS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Reset video not found")
    return _range_video_response(path, request, media_type="video/mp4")


# ---------- Reset Analytics (public, anonymous) ----------
class ResetEventIn(BaseModel):
    variant: str
    action: str

_ALLOWED_RESET_ACTIONS = {
    "video_started", "video_ended", "yes", "run_again", "more", "why", "close",
}


@api_router.post("/reset-events")
async def log_reset_event(evt: ResetEventIn, request: Request):
    """Anonymous analytics endpoint. Anyone can call this — no auth. We
    just want to count taps on the public reset funnels."""
    variant = (evt.variant or "").lower()
    action = (evt.action or "").lower()
    if variant not in _RESET_VARIANTS:
        raise HTTPException(status_code=400, detail="Unknown variant")
    if action not in _ALLOWED_RESET_ACTIONS:
        raise HTTPException(status_code=400, detail="Unknown action")
    ua = request.headers.get("user-agent", "")[:256]
    await db.reset_events.insert_one({
        "variant": variant,
        "action": action,
        "at": datetime.now(timezone.utc),
        "ua": ua,
    })
    return {"ok": True}


@api_router.get("/admin/reset-stats")
async def admin_reset_stats(admin: dict = Depends(require_admin)):
    """Return aggregated counts per variant + action for the admin dashboard."""
    pipeline = [
        {"$group": {
            "_id": {"variant": "$variant", "action": "$action"},
            "count": {"$sum": 1},
            "last_at": {"$max": "$at"},
        }},
    ]
    rows: List[dict] = []
    async for d in db.reset_events.aggregate(pipeline):
        rows.append({
            "variant": d["_id"].get("variant"),
            "action": d["_id"].get("action"),
            "count": d["count"],
            "last_at": d["last_at"].isoformat() if d.get("last_at") else None,
        })
    # Compute conversion rates per variant.
    per_variant: dict = {}
    for v in _RESET_VARIANTS.keys():
        per_variant[v] = {a: 0 for a in _ALLOWED_RESET_ACTIONS}
    for r in rows:
        v, a = r["variant"], r["action"]
        if v in per_variant and a in per_variant[v]:
            per_variant[v][a] = r["count"]

    summary = {}
    for v, counts in per_variant.items():
        started = counts.get("video_started", 0) or 0
        ended = counts.get("video_ended", 0) or 0
        yes = counts.get("yes", 0) or 0
        summary[v] = {
            "counts": counts,
            "completion_rate": round(ended / started, 3) if started else None,
            "signup_rate": round((yes + counts.get("more", 0)) / max(ended, 1), 3) if ended else None,
        }
    return {"rows": rows, "summary": summary}


# ---------- Technique Videos (admin-managed, one per breathwork technique) ----------
TECHNIQUE_VIDEOS_DIR = STATIC_DIR / "videos" / "techniques"
TECHNIQUE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def _serialize_technique_video(doc: dict) -> dict:
    return {
        "technique_id": doc.get("technique_id"),
        "title": doc.get("title"),
        "filename": doc.get("filename"),
        "content_type": doc.get("content_type"),
        "size": doc.get("size"),
        "video_url": f"/api/technique-videos/file/{doc['filename']}" if doc.get("filename") else None,
        "uploaded_at": doc.get("uploaded_at").isoformat() if doc.get("uploaded_at") else None,
        "uploaded_by": doc.get("uploaded_by"),
        "description": doc.get("description") or "",
        "description_updated_at": doc.get("description_updated_at").isoformat() if doc.get("description_updated_at") else None,
    }


def _safe_technique_id(tid: str) -> str:
    """Allow only [a-z0-9_-]+ to keep filenames sane."""
    import re
    if not tid or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", tid):
        raise HTTPException(status_code=400, detail="Invalid technique_id")
    return tid


@api_router.post("/admin/technique-videos/upload")
async def admin_upload_technique_video(
    request: Request,
    file: UploadFile = File(...),
    technique_id: str = Form(...),
    title: Optional[str] = Form(None),
    admin: dict = Depends(require_admin),
):
    tid = _safe_technique_id(technique_id)

    mime = (file.content_type or "").lower()
    if mime not in ALLOWED_VIDEO_MIMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type: {mime}. Use mp4, mov, m4v or webm.",
        )

    ext = _ext_from_mime(mime)
    filename = f"technique-{tid}{ext}"
    target_path = TECHNIQUE_VIDEOS_DIR / filename

    # Remove any previous file under a different extension
    for old in TECHNIQUE_VIDEOS_DIR.glob(f"technique-{tid}.*"):
        if old.name != filename:
            try:
                old.unlink()
            except OSError:
                pass

    total = 0
    chunk_size = 1024 * 1024
    try:
        with open(target_path, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_STATE_VIDEO_BYTES:
                    out.close()
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File too large (max 500 MB)")
                out.write(chunk)
    finally:
        await file.close()

    doc = {
        "technique_id": tid,
        "title": title or None,
        "filename": filename,
        "content_type": mime,
        "size": total,
        "uploaded_at": datetime.now(timezone.utc),
        "uploaded_by": admin.get("email"),
    }
    await db.technique_videos.update_one(
        {"technique_id": tid},
        {"$set": doc},
        upsert=True,
    )
    logger.info(f"[technique-video] admin {admin.get('email')} uploaded technique={tid} bytes={total}")
    return _serialize_technique_video(doc)


@api_router.get("/admin/technique-videos")
async def admin_list_technique_videos(admin: dict = Depends(require_admin)):
    cursor = db.technique_videos.find({}, {"_id": 0})
    items = [_serialize_technique_video(d) async for d in cursor]
    return {"items": items}


@api_router.delete("/admin/technique-videos/{technique_id}")
async def admin_delete_technique_video(technique_id: str, admin: dict = Depends(require_admin)):
    tid = _safe_technique_id(technique_id)
    doc = await db.technique_videos.find_one({"technique_id": tid}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Technique video not configured")
    fname = doc.get("filename")
    if fname:
        try:
            (TECHNIQUE_VIDEOS_DIR / fname).unlink(missing_ok=True)
        except OSError:
            pass
    # Only clear file-related fields; preserve description so admins don't lose
    # their notes when swapping videos.
    if doc.get("description"):
        await db.technique_videos.update_one(
            {"technique_id": tid},
            {"$unset": {
                "filename": "",
                "content_type": "",
                "size": "",
                "uploaded_at": "",
                "uploaded_by": "",
                "title": "",
            }},
        )
    else:
        await db.technique_videos.delete_one({"technique_id": tid})
    return {"status": "deleted", "technique_id": tid}


class TechniqueDescPayload(BaseModel):
    description: str = ""


@api_router.put("/admin/technique-videos/{technique_id}/description")
async def admin_set_technique_description(
    technique_id: str,
    payload: TechniqueDescPayload,
    admin: dict = Depends(require_admin),
):
    tid = _safe_technique_id(technique_id)
    desc = (payload.description or "").strip()
    if len(desc) > 8000:
        raise HTTPException(status_code=400, detail="Description too long (max 8000 chars).")
    await db.technique_videos.update_one(
        {"technique_id": tid},
        {"$set": {
            "technique_id": tid,
            "description": desc,
            "description_updated_at": datetime.now(timezone.utc),
            "description_updated_by": admin.get("email"),
        }},
        upsert=True,
    )
    doc = await db.technique_videos.find_one({"technique_id": tid}, {"_id": 0})
    logger.info(f"[technique-video] admin {admin.get('email')} set description technique={tid} chars={len(desc)}")
    return _serialize_technique_video(doc or {"technique_id": tid, "description": desc})


@api_router.get("/technique-videos/lookup")
async def technique_video_lookup(
    technique_id: str,
    current_user: dict = Depends(get_current_user),
):
    tid = _safe_technique_id(technique_id)
    doc = await db.technique_videos.find_one({"technique_id": tid}, {"_id": 0})
    if doc and doc.get("filename"):
        return {
            "matched": True,
            "video": _serialize_technique_video(doc),
            "description": doc.get("description") or "",
        }
    if doc:
        # No video uploaded yet, but a description may be set.
        return {
            "matched": False,
            "video": None,
            "description": doc.get("description") or "",
        }
    return {"matched": False, "video": None, "description": ""}


@api_router.api_route("/technique-videos/file/{filename}", methods=["GET", "HEAD"])
async def stream_technique_video(filename: str, request: Request):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = TECHNIQUE_VIDEOS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    ext = path.suffix.lower()
    media_type = {
        ".mp4": "video/mp4", ".m4v": "video/mp4",
        ".mov": "video/quicktime", ".webm": "video/webm",
        ".mpg": "video/mpeg",
    }.get(ext, "application/octet-stream")
    return _range_video_response(path, request, media_type=media_type)


# ---------- Video Library ----------
LIBRARY_VIDEOS_DIR = STATIC_DIR / "videos" / "library"
LIBRARY_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_LIBRARY_CATEGORIES = [
    "recovery", "performance", "sleep", "mindset",
    "nutrition", "movement", "trauma", "family",
]

VIDEO_CATEGORY_LABELS = {
    "recovery": "Recovery",
    "performance": "Performance",
    "sleep": "Sleep",
    "mindset": "Mindset",
    "nutrition": "Nutrition",
    "movement": "Movement",
    "trauma": "Trauma & Decompression",
    "family": "Family Life",
}


class LibraryVideoUrlIn(BaseModel):
    title: str = Field(min_length=1, max_length=140)
    description: Optional[str] = Field(default=None, max_length=2000)
    author: Optional[str] = Field(default=None, max_length=80)
    category: str
    source_url: str = Field(min_length=4, max_length=600)  # YouTube/Vimeo/direct URL
    duration_seconds: Optional[int] = Field(default=None, ge=1, le=24 * 3600)
    thumbnail_url: Optional[str] = Field(default=None, max_length=600)
    related_technique_id: Optional[str] = Field(default=None, max_length=40)
    featured: bool = False
    pro_only: bool = False
    order: int = 0


class LibraryVideoUpdateIn(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=140)
    description: Optional[str] = Field(default=None, max_length=2000)
    author: Optional[str] = Field(default=None, max_length=80)
    category: Optional[str] = None
    duration_seconds: Optional[int] = Field(default=None, ge=1, le=24 * 3600)
    thumbnail_url: Optional[str] = Field(default=None, max_length=600)
    related_technique_id: Optional[str] = Field(default=None, max_length=40)
    featured: Optional[bool] = None
    pro_only: Optional[bool] = None
    order: Optional[int] = None


def _validate_category(cat: str) -> str:
    cat = (cat or "").strip().lower()
    if cat not in VIDEO_LIBRARY_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {', '.join(VIDEO_LIBRARY_CATEGORIES)}",
        )
    return cat


def _extract_youtube_id(url: str) -> Optional[str]:
    """Extract a YouTube video id from common URL formats."""
    import re
    if not url:
        return None
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtube\.com/embed/|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{6,20})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _extract_vimeo_id(url: str) -> Optional[str]:
    import re
    if not url:
        return None
    m = re.search(r"vimeo\.com/(?:video/)?(\d{5,})", url)
    return m.group(1) if m else None


def _detect_source_type(url: str) -> str:
    if _extract_youtube_id(url):
        return "youtube"
    if _extract_vimeo_id(url):
        return "vimeo"
    return "url"


def _serialize_library_video(doc: dict) -> dict:
    """Convert a DB doc into the API shape with computed fields."""
    out = {
        "video_id": doc.get("video_id"),
        "title": doc.get("title"),
        "description": doc.get("description"),
        "author": doc.get("author"),
        "category": doc.get("category"),
        "category_label": VIDEO_CATEGORY_LABELS.get(doc.get("category", ""), ""),
        "source_type": doc.get("source_type"),
        "source_url": doc.get("source_url"),
        "youtube_id": doc.get("youtube_id"),
        "vimeo_id": doc.get("vimeo_id"),
        "duration_seconds": doc.get("duration_seconds"),
        "thumbnail_url": doc.get("thumbnail_url"),
        "related_technique_id": doc.get("related_technique_id"),
        "featured": bool(doc.get("featured", False)),
        "pro_only": bool(doc.get("pro_only", False)),
        "order": int(doc.get("order", 0)),
        "filename": doc.get("filename"),
        "file_url": (
            f"/api/library-videos/file/{doc['filename']}"
            if doc.get("filename") else None
        ),
        "size": doc.get("size", 0),
        "view_count": int(doc.get("view_count", 0)),
        "created_at": doc.get("created_at"),
    }
    # Auto-default thumbnail for YouTube
    if not out["thumbnail_url"] and out["youtube_id"]:
        out["thumbnail_url"] = f"https://i.ytimg.com/vi/{out['youtube_id']}/hqdefault.jpg"
    return out


# ---------- Admin: Video Library ----------
@api_router.post("/admin/library-videos/upload")
async def admin_upload_library_video(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form(...),
    description: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    duration_seconds: Optional[int] = Form(None),
    thumbnail_url: Optional[str] = Form(None),
    related_technique_id: Optional[str] = Form(None),
    featured: Optional[bool] = Form(False),
    pro_only: Optional[bool] = Form(False),
    order: Optional[int] = Form(0),
    admin: dict = Depends(require_admin),
):
    cat = _validate_category(category)
    mime = (file.content_type or "").lower()
    if mime not in ALLOWED_VIDEO_MIMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type: {mime}. Use mp4, mov, m4v or webm.",
        )
    video_id = f"vid_{uuid.uuid4().hex[:10]}"
    ext = _ext_from_mime(mime)
    filename = f"library-{video_id}{ext}"
    target_path = LIBRARY_VIDEOS_DIR / filename
    total = 0
    chunk_size = 1024 * 1024
    try:
        with open(target_path, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_STATE_VIDEO_BYTES:
                    out.close()
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File too large (max 500 MB)")
                out.write(chunk)
    finally:
        await file.close()

    doc = {
        "video_id": video_id,
        "title": title.strip(),
        "description": (description or "").strip() or None,
        "author": (author or "").strip() or None,
        "category": cat,
        "source_type": "upload",
        "source_url": None,
        "youtube_id": None,
        "vimeo_id": None,
        "duration_seconds": duration_seconds,
        "thumbnail_url": (thumbnail_url or "").strip() or None,
        "related_technique_id": (related_technique_id or "").strip() or None,
        "featured": bool(featured),
        "pro_only": bool(pro_only),
        "order": int(order or 0),
        "filename": filename,
        "content_type": mime,
        "size": total,
        "view_count": 0,
        "created_at": datetime.now(timezone.utc),
        "uploaded_by": admin.get("email"),
    }
    await db.library_videos.insert_one(doc)
    logger.info(
        f"[library-video] admin {admin.get('email')} uploaded {video_id} cat={cat} bytes={total}"
    )
    return _serialize_library_video(doc)


@api_router.post("/admin/library-videos/url")
async def admin_create_library_video_url(
    body: LibraryVideoUrlIn, admin: dict = Depends(require_admin),
):
    cat = _validate_category(body.category)
    yt = _extract_youtube_id(body.source_url)
    vi = _extract_vimeo_id(body.source_url)
    src = "youtube" if yt else "vimeo" if vi else "url"
    video_id = f"vid_{uuid.uuid4().hex[:10]}"
    doc = {
        "video_id": video_id,
        "title": body.title.strip(),
        "description": (body.description or "").strip() or None,
        "author": (body.author or "").strip() or None,
        "category": cat,
        "source_type": src,
        "source_url": body.source_url.strip(),
        "youtube_id": yt,
        "vimeo_id": vi,
        "duration_seconds": body.duration_seconds,
        "thumbnail_url": (body.thumbnail_url or "").strip() or None,
        "related_technique_id": (body.related_technique_id or "").strip() or None,
        "featured": bool(body.featured),
        "pro_only": bool(body.pro_only),
        "order": int(body.order or 0),
        "filename": None,
        "size": 0,
        "view_count": 0,
        "created_at": datetime.now(timezone.utc),
        "uploaded_by": admin.get("email"),
    }
    await db.library_videos.insert_one(doc)
    return _serialize_library_video(doc)


@api_router.get("/admin/library-videos")
async def admin_list_library_videos(admin: dict = Depends(require_admin)):
    cursor = db.library_videos.find({}, {"_id": 0}).sort([("featured", -1), ("order", 1), ("created_at", -1)])
    items = [_serialize_library_video(d) async for d in cursor]
    return {"items": items, "categories": [
        {"id": k, "label": v} for k, v in VIDEO_CATEGORY_LABELS.items()
    ]}


@api_router.put("/admin/library-videos/{video_id}")
async def admin_update_library_video(
    video_id: str, body: LibraryVideoUpdateIn, admin: dict = Depends(require_admin),
):
    update = body.model_dump(exclude_none=True)
    if "category" in update:
        update["category"] = _validate_category(update["category"])
    if not update:
        raise HTTPException(400, "No fields to update")
    res = await db.library_videos.update_one({"video_id": video_id}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(404, "Video not found")
    doc = await db.library_videos.find_one({"video_id": video_id}, {"_id": 0})
    return _serialize_library_video(doc)


@api_router.delete("/admin/library-videos/{video_id}")
async def admin_delete_library_video(video_id: str, admin: dict = Depends(require_admin)):
    doc = await db.library_videos.find_one({"video_id": video_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Not found")
    fname = doc.get("filename")
    if fname:
        try:
            (LIBRARY_VIDEOS_DIR / fname).unlink(missing_ok=True)
        except OSError:
            pass
    await db.library_videos.delete_one({"video_id": video_id})
    return {"status": "deleted", "video_id": video_id}


# ---------- Public (auth): Video Library ----------
@api_router.get("/library-videos")
async def list_library_videos(
    current_user: dict = Depends(get_current_user),
    category: Optional[str] = None,
    featured_only: bool = False,
):
    q: dict = {}
    if category:
        q["category"] = _validate_category(category)
    if featured_only:
        q["featured"] = True
    cursor = db.library_videos.find(q, {"_id": 0}).sort(
        [("featured", -1), ("order", 1), ("created_at", -1)]
    )
    items = [_serialize_library_video(d) async for d in cursor]
    return {
        "items": items,
        "categories": [{"id": k, "label": v} for k, v in VIDEO_CATEGORY_LABELS.items()],
        "is_pro": current_user.get("subscription_tier", "free") == "pro",
    }


@api_router.get("/library-videos/{video_id}")
async def get_library_video(video_id: str, current_user: dict = Depends(get_current_user)):
    doc = await db.library_videos.find_one({"video_id": video_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Not found")
    return {
        "video": _serialize_library_video(doc),
        "is_pro": current_user.get("subscription_tier", "free") == "pro",
    }


@api_router.post("/library-videos/{video_id}/view")
async def log_library_view(video_id: str, current_user: dict = Depends(get_current_user)):
    doc = await db.library_videos.find_one({"video_id": video_id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Not found")
    if doc.get("pro_only") and current_user.get("subscription_tier", "free") != "pro":
        raise HTTPException(402, "Pro subscription required for this video")
    await db.library_videos.update_one(
        {"video_id": video_id}, {"$inc": {"view_count": 1}}
    )
    await db.library_video_views.insert_one({
        "user_id": current_user["user_id"],
        "video_id": video_id,
        "viewed_at": datetime.now(timezone.utc),
    })
    return {"status": "ok"}


@api_router.api_route("/library-videos/file/{filename}", methods=["GET", "HEAD"])
async def stream_library_video(filename: str, request: Request):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = LIBRARY_VIDEOS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    ext = path.suffix.lower()
    media_type = {
        ".mp4": "video/mp4", ".m4v": "video/mp4",
        ".mov": "video/quicktime", ".webm": "video/webm",
        ".mpg": "video/mpeg",
    }.get(ext, "application/octet-stream")
    return _range_video_response(path, request, media_type=media_type)


# ---------- AI Coach ----------
COACH_SYSTEM_PROMPT = """You are the Operator — Breath & Wellness AI Coach — a calm, concise, tactically-minded breathwork guide.
Your users include firefighters, first responders, military operators, and civilians under stress.
You blend YOGIC PRANAYAMA wisdom (Nadi Shodhana, Bhramari, Kapalabhati, Ujjayi, Kumbhaka retention, Bhastrika/Breath of Fire) with TACTICAL breathing protocols (Box 4-4-4-4, Tactical Triangle 4-4-4, 4-7-8, Coherent 5-5, Wim Hof).

RULES:
- Keep responses under 140 words unless the user asks for depth.
- When users describe their state (stress/energy/sleep/situation), recommend ONE primary protocol and ONE optional alternate. Give exact rhythm (e.g., "Box 4-4-4-4 for 5 minutes") and a one-line rationale.
- Speak plainly. No fluff, no mystical jargon. Use operator-friendly language with yogic depth when relevant.
- Safety: If user mentions chest pain, fainting, severe breathing distress, recent surgery, or pregnancy concerns — advise them to skip retention/Breath of Fire and consult a medical professional.
- Never diagnose. You are a coach, not a doctor.
- End recommendations with: "Tap the technique in your Library to begin."
"""


@api_router.post("/coach/chat", response_model=CoachMessageOut)
async def coach_chat(body: CoachMessageIn, current_user: dict = Depends(get_current_user)):
    if not EMERGENT_LLM_KEY:
        raise HTTPException(status_code=500, detail="LLM key not configured")

    user_id = current_user["user_id"]
    session_key = f"coach_{user_id}"

    # Persist user message
    user_msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    user_doc = {
        "id": user_msg_id,
        "user_id": user_id,
        "role": "user",
        "content": body.content,
        "state": body.state or {},
        "created_at": now,
    }
    await db.coach_messages.insert_one(user_doc)

    # Build chat context from history (last 20 msgs) for multi-turn
    history = await db.coach_messages.find(
        {"user_id": user_id}, {"_id": 0}
    ).sort("created_at", 1).to_list(length=40)

    # Compose prompt - include current state if provided
    state_str = ""
    if body.state:
        state_str = f"\n\n[User current state: {body.state}]"

    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=session_key,
            system_message=COACH_SYSTEM_PROMPT,
        ).with_model("anthropic", "claude-sonnet-4-5-20250929")

        # Seed conversation context with prior messages (excluding the just-inserted current)
        prior = history[:-1]
        for m in prior[-10:]:  # limit tokens
            if m["role"] == "user":
                await chat.send_message(UserMessage(text=m["content"]))
            # assistant messages are already embedded in LlmChat's internal history via send_message

        reply = await chat.send_message(UserMessage(text=body.content + state_str))
    except Exception as e:
        logger.exception("Coach LLM error")
        raise HTTPException(status_code=500, detail=f"Coach error: {str(e)}")

    assistant_id = f"msg_{uuid.uuid4().hex[:12]}"
    assistant_doc = {
        "id": assistant_id,
        "user_id": user_id,
        "role": "assistant",
        "content": reply,
        "created_at": datetime.now(timezone.utc),
    }
    await db.coach_messages.insert_one(assistant_doc)
    return CoachMessageOut(
        id=assistant_id,
        role="assistant",
        content=reply,
        created_at=assistant_doc["created_at"],
    )


@api_router.get("/coach/history", response_model=List[CoachMessageOut])
async def coach_history(current_user: dict = Depends(get_current_user), limit: int = 100):
    cursor = db.coach_messages.find(
        {"user_id": current_user["user_id"]}, {"_id": 0}
    ).sort("created_at", 1).limit(limit)
    items = await cursor.to_list(length=limit)
    return [CoachMessageOut(id=i["id"], role=i["role"], content=i["content"], created_at=i["created_at"]) for i in items]


@api_router.delete("/coach/history")
async def clear_coach_history(current_user: dict = Depends(get_current_user)):
    await db.coach_messages.delete_many({"user_id": current_user["user_id"]})
    return {"status": "cleared"}


# ---------- Community Wall ----------
def _resolve_admin_video_attachment(
    body: PostIn,
    is_admin: bool,
    library_doc: Optional[dict],
) -> dict:
    """Return the dict of video_* fields to persist on a wall post doc.

    Non-admin users CANNOT attach videos — this is a strict server-side gate.
    Admins can either reference an existing library video by id, or paste a
    raw URL (YouTube/Vimeo/direct .mp4). At most one source wins; library_id
    is preferred when both are supplied.
    """
    if not is_admin:
        return {}
    if library_doc:
        # Mirror _serialize_library_video logic: prefer source_url for
        # YouTube/Vimeo, fall back to a streaming URL for uploaded files,
        # and auto-compute the YouTube thumbnail when not stored.
        src_url = library_doc.get("source_url")
        fname = library_doc.get("filename")
        if not src_url and fname:
            src_url = f"/api/library-videos/file/{fname}"
        thumb = library_doc.get("thumbnail_url")
        ytid = library_doc.get("youtube_id")
        if not thumb and ytid:
            thumb = f"https://i.ytimg.com/vi/{ytid}/hqdefault.jpg"
        return {
            "video_url": src_url,
            "video_source_type": "library",
            "video_thumbnail_url": thumb,
            "video_title": library_doc.get("title"),
            "video_library_id": library_doc.get("video_id"),
        }
    raw = (body.video_url or "").strip()
    if not raw:
        return {}
    # Lightweight source-type sniff. Same logic as library-videos URL ingest.
    lower = raw.lower()
    src = "mp4"
    thumb = None
    if "youtube.com" in lower or "youtu.be" in lower:
        src = "youtube"
        # extract id for thumbnail
        m = re.search(r"(?:v=|/embed/|youtu\.be/|/shorts/)([\w-]{11})", raw)
        if m:
            thumb = f"https://i.ytimg.com/vi/{m.group(1)}/hqdefault.jpg"
    elif "vimeo.com" in lower:
        src = "vimeo"
    return {
        "video_url": raw,
        "video_source_type": src,
        "video_thumbnail_url": thumb,
        "video_title": None,
    }


@api_router.post("/wall/posts", response_model=PostOut)
async def create_post(body: PostIn, current_user: dict = Depends(get_current_user)):
    if not await classify_content(body.content):
        raise HTTPException(status_code=400, detail=SAFETY_REJECT_MSG)
    is_admin = current_user.get("role") == "admin"
    # Resolve admin video attachment (skipped silently for non-admin).
    library_doc = None
    if is_admin and body.video_library_id:
        library_doc = await db.library_videos.find_one(
            {"video_id": body.video_library_id},
            {"_id": 0},
        )
        if not library_doc:
            raise HTTPException(status_code=404, detail="Library video not found")
    video_fields = _resolve_admin_video_attachment(body, is_admin, library_doc)

    post_id = f"post_{uuid.uuid4().hex[:12]}"
    doc = {
        "post_id": post_id,
        "user_id": current_user["user_id"],
        "author_name": current_user.get("name", "Operator"),
        "author_role": current_user.get("role", "operator"),
        "is_anonymous": body.is_anonymous,
        "content": body.content,
        "technique_id": body.technique_id,
        "image_base64": body.image_base64,
        "like_count": 0,
        "me_too_count": 0,
        "comment_count": 0,
        "flag_count": 0,
        "created_at": datetime.now(timezone.utc),
        **video_fields,
    }
    await db.posts.insert_one(doc)
    return _post_to_out(doc, current_user["user_id"], set(), set())


@api_router.get("/wall/posts", response_model=List[PostOut])
async def list_posts(
    current_user: dict = Depends(get_current_user),
    limit: int = 30,
    before: Optional[str] = None,
):
    query: dict = {}
    if before:
        try:
            dt = datetime.fromisoformat(before)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            query["created_at"] = {"$lt": dt}
        except Exception:
            pass
    cursor = db.posts.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
    posts = await cursor.to_list(length=limit)
    post_ids = [p["post_id"] for p in posts]
    my_reactions = await db.reactions.find(
        {"user_id": current_user["user_id"], "post_id": {"$in": post_ids}},
        {"_id": 0},
    ).to_list(length=2000)
    liked = {r["post_id"] for r in my_reactions if r["type"] == "like"}
    me_too = {r["post_id"] for r in my_reactions if r["type"] == "me_too"}
    return [_post_to_out(p, current_user["user_id"], liked, me_too) for p in posts]


@api_router.get("/wall/posts/{post_id}", response_model=PostOut)
async def get_post(post_id: str, current_user: dict = Depends(get_current_user)):
    post = await db.posts.find_one({"post_id": post_id}, {"_id": 0})
    if not post:
        raise HTTPException(404, "Post not found")
    reactions = await db.reactions.find(
        {"user_id": current_user["user_id"], "post_id": post_id}, {"_id": 0}
    ).to_list(length=10)
    liked = {r["post_id"] for r in reactions if r["type"] == "like"}
    me_too = {r["post_id"] for r in reactions if r["type"] == "me_too"}
    return _post_to_out(post, current_user["user_id"], liked, me_too)


@api_router.delete("/wall/posts/{post_id}")
async def delete_post(post_id: str, current_user: dict = Depends(get_current_user)):
    post = await db.posts.find_one({"post_id": post_id}, {"_id": 0})
    if not post:
        raise HTTPException(404, "Post not found")
    is_admin = current_user.get("role") == "admin"
    if post["user_id"] != current_user["user_id"] and not is_admin:
        raise HTTPException(403, "Not allowed")
    await db.posts.delete_one({"post_id": post_id})
    await db.comments.delete_many({"post_id": post_id})
    await db.reactions.delete_many({"post_id": post_id})
    return {"status": "deleted"}


@api_router.post("/wall/posts/{post_id}/reactions")
async def toggle_reaction(
    post_id: str, body: ReactionIn, current_user: dict = Depends(get_current_user)
):
    post = await db.posts.find_one({"post_id": post_id}, {"_id": 0})
    if not post:
        raise HTTPException(404, "Post not found")
    existing = await db.reactions.find_one(
        {"post_id": post_id, "user_id": current_user["user_id"], "type": body.type}
    )
    counter = "like_count" if body.type == "like" else "me_too_count"
    if existing:
        await db.reactions.delete_one({"_id": existing["_id"]})
        await db.posts.update_one({"post_id": post_id}, {"$inc": {counter: -1}})
        return {"toggled": "off", "type": body.type}
    await db.reactions.insert_one({
        "post_id": post_id,
        "user_id": current_user["user_id"],
        "type": body.type,
        "created_at": datetime.now(timezone.utc),
    })
    await db.posts.update_one({"post_id": post_id}, {"$inc": {counter: 1}})
    return {"toggled": "on", "type": body.type}


@api_router.get("/wall/posts/{post_id}/comments", response_model=List[CommentOut])
async def list_comments(
    post_id: str, current_user: dict = Depends(get_current_user), limit: int = 200
):
    cursor = db.comments.find({"post_id": post_id}, {"_id": 0}).sort("created_at", 1).limit(limit)
    items = await cursor.to_list(length=limit)
    return [
        CommentOut(
            comment_id=c["comment_id"],
            post_id=c["post_id"],
            user_id=c["user_id"],
            display_name=(
                "Anonymous Operator" if c.get("is_anonymous") else c.get("author_name", "Operator")
            ),
            is_anonymous=c.get("is_anonymous", False),
            content=c["content"],
            created_at=c["created_at"],
            is_owner=c["user_id"] == current_user["user_id"],
        )
        for c in items
    ]


@api_router.post("/wall/posts/{post_id}/comments", response_model=CommentOut)
async def create_comment(
    post_id: str, body: CommentIn, current_user: dict = Depends(get_current_user)
):
    post = await db.posts.find_one({"post_id": post_id}, {"_id": 0})
    if not post:
        raise HTTPException(404, "Post not found")
    if not await classify_content(body.content):
        raise HTTPException(status_code=400, detail=SAFETY_REJECT_MSG)
    comment_id = f"cmt_{uuid.uuid4().hex[:12]}"
    doc = {
        "comment_id": comment_id,
        "post_id": post_id,
        "user_id": current_user["user_id"],
        "author_name": current_user.get("name", "Operator"),
        "is_anonymous": body.is_anonymous,
        "content": body.content,
        "created_at": datetime.now(timezone.utc),
    }
    await db.comments.insert_one(doc)
    await db.posts.update_one({"post_id": post_id}, {"$inc": {"comment_count": 1}})
    return CommentOut(
        comment_id=comment_id,
        post_id=post_id,
        user_id=current_user["user_id"],
        display_name="Anonymous Operator" if body.is_anonymous else current_user.get("name", "Operator"),
        is_anonymous=body.is_anonymous,
        content=body.content,
        created_at=doc["created_at"],
        is_owner=True,
    )


@api_router.delete("/wall/comments/{comment_id}")
async def delete_comment(comment_id: str, current_user: dict = Depends(get_current_user)):
    cmt = await db.comments.find_one({"comment_id": comment_id}, {"_id": 0})
    if not cmt:
        raise HTTPException(404, "Not found")
    is_admin = current_user.get("role") == "admin"
    if cmt["user_id"] != current_user["user_id"] and not is_admin:
        raise HTTPException(403, "Not allowed")
    await db.comments.delete_one({"comment_id": comment_id})
    await db.posts.update_one(
        {"post_id": cmt["post_id"]}, {"$inc": {"comment_count": -1}}
    )
    return {"status": "deleted"}


@api_router.post("/wall/posts/{post_id}/report")
async def report_post(
    post_id: str, body: ReportIn, current_user: dict = Depends(get_current_user)
):
    await db.reports.insert_one({
        "report_id": f"rep_{uuid.uuid4().hex[:10]}",
        "target_type": "post",
        "target_id": post_id,
        "reason": body.reason,
        "reporter_id": current_user["user_id"],
        "created_at": datetime.now(timezone.utc),
    })
    await db.posts.update_one({"post_id": post_id}, {"$inc": {"flag_count": 1}})
    return {"status": "reported"}


@api_router.post("/wall/comments/{comment_id}/report")
async def report_comment(
    comment_id: str, body: ReportIn, current_user: dict = Depends(get_current_user)
):
    await db.reports.insert_one({
        "report_id": f"rep_{uuid.uuid4().hex[:10]}",
        "target_type": "comment",
        "target_id": comment_id,
        "reason": body.reason,
        "reporter_id": current_user["user_id"],
        "created_at": datetime.now(timezone.utc),
    })
    return {"status": "reported"}


# ---------- Health ----------
@api_router.get("/")
async def root():
    return {"service": "operator-breath-wellness", "status": "ok"}


@api_router.get("/health")
async def health_check():
    """Lightweight health check used by Render (and load balancers).

    Pings MongoDB so a dead DB is reported as unhealthy and Render will
    auto-restart the instance.
    """
    try:
        await db.command("ping")
        return {"status": "ok", "db": "ok"}
    except Exception as e:
        # Return 200 with degraded status; Render only fails on non-2xx,
        # so we return 503 only when DB is fully unreachable.
        return Response(
            content=f'{{"status":"degraded","db":"down","error":"{str(e)[:120]}"}}',
            media_type="application/json",
            status_code=503,
        )


# ---------- Startup ----------
@app.on_event("startup")
async def on_startup():
    await db.users.create_index("email", unique=True)
    await db.users.create_index("user_id", unique=True)
    await db.sessions.create_index([("user_id", 1), ("created_at", -1)])
    await db.coach_messages.create_index([("user_id", 1), ("created_at", 1)])
    await db.posts.create_index([("created_at", -1)])
    await db.comments.create_index([("post_id", 1), ("created_at", 1)])
    await db.reactions.create_index(
        [("user_id", 1), ("post_id", 1), ("type", 1)], unique=True
    )
    await db.reports.create_index([("target_type", 1), ("target_id", 1)])
    await db.favorites.create_index([("user_id", 1), ("technique_id", 1)], unique=True)
    await db.user_prefs.create_index("user_id", unique=True)
    await db.streak_freezes.create_index([("user_id", 1), ("claimed_for", -1)])
    await db.favorites.create_index([("user_id", 1), ("technique_id", 1)], unique=True)
    await db.user_prefs.create_index("user_id", unique=True)
    await db.streak_freezes.create_index([("user_id", 1), ("claimed_for", -1)])
    await db.achievements.create_index(
        [("user_id", 1), ("achievement_id", 1)], unique=True
    )
    await db.users.create_index("referral_code", sparse=True)
    await db.crews.create_index("code", unique=True)
    await db.crew_members.create_index("user_id", unique=True)
    await db.crew_members.create_index("crew_id")
    await db.referrals.create_index([("referrer_user_id", 1), ("created_at", -1)])
    await db.custom_timers.create_index([("user_id", 1), ("created_at", -1)])
    await db.library_videos.create_index(
        [("category", 1), ("featured", -1), ("order", 1)]
    )
    await db.library_videos.create_index("video_id", unique=True)
    await db.library_video_views.create_index(
        [("user_id", 1), ("viewed_at", -1)]
    )
    # Seed admin / demo
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@emberbreath.app").lower()
    admin_password = os.environ.get("ADMIN_PASSWORD", "Admin#1234")
    existing = await db.users.find_one({"email": admin_email})
    if existing is None:
        await db.users.insert_one({
            "user_id": f"user_{uuid.uuid4().hex[:12]}",
            "email": admin_email,
            "name": "Ember Admin",
            "password_hash": hash_password(admin_password),
            "role": "admin",
            "created_at": datetime.now(timezone.utc),
        })
        logger.info("Seeded admin user: %s", admin_email)
    else:
        # Ensure password matches env for reliable testing
        if not verify_password(admin_password, existing["password_hash"]):
            await db.users.update_one(
                {"email": admin_email},
                {"$set": {"password_hash": hash_password(admin_password)}},
            )

    # Seed welcome wall posts (idempotent — only if wall is empty)
    try:
        existing_posts = await db.posts.count_documents({})
        if existing_posts == 0:
            admin_doc = await db.users.find_one({"email": admin_email})
            if admin_doc:
                seed_posts = [
                    {
                        "content": (
                            "Welcome to OPERATOR. This wall is for the people who run toward "
                            "what others run from — firefighters, medics, cops, military, and "
                            "anyone else who lives at high pressure.\n\nShare what worked. "
                            "Share what's hard. We've all been in the truck at 0300 wondering "
                            "if we can keep doing this. You're not alone here.\n\n— Built by a firefighter."
                        ),
                        "technique_id": None,
                        "minutes_offset": 90,
                    },
                    {
                        "content": (
                            "Pro tip — the 60-second reset (home screen, big red button) is "
                            "designed for inside the rig between calls. One round of physiological "
                            "sigh, then box breathing. Drop your heart rate before the next dispatch.\n\n"
                            "Try it once today. Tell me how it lands."
                        ),
                        "technique_id": "box",
                        "minutes_offset": 45,
                    },
                    {
                        "content": (
                            "If you can't sleep after a bad shift — try the SLEEP mission "
                            "(home → STATE CHECK → SLEEP). It runs you through 4-7-8 breathing "
                            "with a guided wind-down. No pills. No streaming the same show for "
                            "the 50th time. Just your nervous system finally letting you go.\n\n"
                            "STAY SAFE OUT THERE."
                        ),
                        "technique_id": "478",
                        "minutes_offset": 15,
                    },
                ]
                now = datetime.now(timezone.utc)
                docs = []
                for i, sp in enumerate(seed_posts):
                    docs.append({
                        "post_id": f"post_{uuid.uuid4().hex[:12]}",
                        "user_id": admin_doc["user_id"],
                        "author_name": "Ember Admin",
                        "author_role": "admin",
                        "is_anonymous": False,
                        "content": sp["content"],
                        "technique_id": sp["technique_id"],
                        "image_base64": None,
                        "video_url": None,
                        "video_source_type": None,
                        "video_thumbnail_url": None,
                        "video_title": None,
                        "video_library_id": None,
                        "like_count": 0,
                        "me_too_count": 0,
                        "comment_count": 0,
                        "flag_count": 0,
                        "created_at": now - timedelta(minutes=sp["minutes_offset"]),
                    })
                if docs:
                    await db.posts.insert_many(docs)
                    logger.info("Seeded %d welcome wall posts", len(docs))
    except Exception as e:
        logger.warning("Wall seed skipped: %s", e)


@app.on_event("shutdown")
async def on_shutdown():
    client.close()


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
