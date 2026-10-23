"""
Ticket #1 — Voice I/O Integration
Maryam's AI Lesson Review Tutor (age 8)

Full voice loop:
    🎤 record  →  📝 Whisper transcription (Groq hosted API)
    →  🧠 Groq LLM tutor response  →  🔊 Web Speech API read-aloud

Locked design decisions (from TICKETS.md):
  * TTS uses the browser-native Web Speech API (`speechSynthesis`), NOT pyttsx3.
    pyttsx3 needs a server-side audio stack and fails on headless Linux
    containers (Streamlit Cloud). `speechSynthesis` runs on Maryam's own
    device, costs nothing, and works on any OS the browser supports.
  * Transcription uses Groq's HOSTED Whisper API, NOT local `openai-whisper`
    weights. Local weights would have to be downloaded + loaded into RAM on
    Streamlit Cloud → multi-minute cold starts. The hosted API has no cold
    start and is on the free tier.
  * Browsers gate audio playback behind a user gesture (autoplay policy).
    We therefore render a big "Play" button (explicit gesture) and only
    ATTEMPT autoplay once, right after a fresh answer is generated.
  * Secrets come from `st.secrets` (Streamlit Cloud) or the `GROQ_API_KEY`
    env var (local, via .env + python-dotenv). Never hardcoded.

Streamlit Cloud gotchas found/anticipated during testing:
  * `st.audio_input` re-returns the SAME audio bytes on every rerun, so we
    de-duplicate by hashing the bytes — otherwise every widget interaction
    would re-transcribe (and burn the Groq rate limit).
  * `st.components.v1.html` re-executes its JS on EVERY rerun. Autoplay is
    therefore armed for exactly one render (consumed flag), or the answer
    would re-speak on every unrelated click.
  * `speechSynthesis.getVoices()` loads asynchronously in Chrome; the speak
    call must wait for `onvoiceschanged` or no voice (esp. Arabic) is found.

Scope: ONLY the voice loop. Lesson picker UI = ticket #2/#3, quiz = #4,
deployment/secrets wiring = #5.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

load_dotenv()  # local dev: read .env if present (no-op on Streamlit Cloud)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [voice-loop] %(message)s",
)
log = logging.getLogger("voice_loop")

# Groq-hosted Whisper (no local weights → no Cloud cold start).
# large-v3 chosen over -turbo: child speech has higher WER, accuracy > speed.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")

# Fast, cheap open model on Groq free tier; plenty for 2–3 sentence answers.
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

REQUEST_TIMEOUT_S = 30          # per-request timeout for Groq calls
HISTORY_CONTEXT_TURNS = 6       # how much chat history the tutor sees
LATENCY_BUDGET_S = 10           # per-turn budget from TICKETS.md (log warning if over)

LESSONS_PATH = Path(__file__).with_name("lessons.json")

# Placeholder used ONLY until ticket #2 ships lessons.json. Keeping it
# in-code avoids touching ticket #2's deliverable.
PLACEHOLDER_LESSON = {
    "id": "placeholder_math",
    "title": "Addition Basics (placeholder)",
    "subject": "Math",
    "content": (
        "Adding numbers means putting groups together and counting how many "
        "there are in total. Example: 5 + 3 = 8. You can count on your "
        "fingers: start at 5, then count 6, 7, 8."
    ),
    "difficulty": "easy",
}

# --------------------------------------------------------------------------- #
# Secrets / clients                                                            #
# --------------------------------------------------------------------------- #


def get_api_key() -> str | None:
    """Streamlit Cloud: st.secrets. Local: GROQ_API_KEY env var (.env)."""
    try:
        key = st.secrets.get("GROQ_API_KEY")  # type: ignore[attr-defined]
        if key:
            return key
    except Exception:
        pass  # no secrets.toml locally — fall through to env var
    return os.getenv("GROQ_API_KEY")


@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str) -> Groq:
    """Cached so we don't rebuild the client on every Streamlit rerun."""
    return Groq(api_key=api_key)


# --------------------------------------------------------------------------- #
# Lessons (real loader is ticket #2 — this just unblocks the voice loop)       #
# --------------------------------------------------------------------------- #


def load_lesson() -> tuple[dict, str]:
    """Return (lesson, source_note). Uses lessons.json if ticket #2 landed."""
    if LESSONS_PATH.exists():
        try:
            data = json.loads(LESSONS_PATH.read_text(encoding="utf-8"))
            lessons = data.get("lessons", [])
            if lessons:
                return lessons[0], f"lessons.json ({len(lessons)} lesson(s))"
            log.warning("lessons.json has no lessons; using placeholder")
        except Exception as exc:  # malformed JSON etc.
            log.warning("Could not parse lessons.json (%s); using placeholder", exc)
    return PLACEHOLDER_LESSON, "placeholder (lessons.json arrives with ticket #2)"


# --------------------------------------------------------------------------- #
# Groq calls with latency timing + kid-friendly error handling                 #
# --------------------------------------------------------------------------- #

TUTOR_SYSTEM_PROMPT = """You are a warm, patient tutor helping Maryam, an 8-year-old girl, review her school lesson.

Rules:
- Answer in 2-3 short sentences. Use simple words an 8-year-old knows.
- Be encouraging and kind. Never say she is wrong in a mean way.
- Stay on the lesson below. If she asks about something else, answer very briefly and gently guide her back to the lesson.
- Reply in the SAME language Maryam used (English or Arabic).

Lesson title: {title}
Lesson content: {content}"""


def transcribe_audio(client: Groq, audio_bytes: bytes) -> tuple[str, float]:
    """Whisper via Groq hosted API. Returns (text, elapsed_ms).

    Language is auto-detected by Whisper (English/Arabic) — see WAYFINDER.md
    bilingual decision. Raises groq exceptions for the caller to handle.
    """
    t0 = time.perf_counter()
    resp = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=("question.wav", io.BytesIO(audio_bytes)),
        response_format="json",
        timeout=REQUEST_TIMEOUT_S,
    )
    return (resp.text or "").strip(), (time.perf_counter() - t0) * 1000


def ask_tutor(
    client: Groq, lesson: dict, history: list[dict], question: str
) -> tuple[str, float]:
    """Groq LLM tutor response. Returns (answer, elapsed_ms)."""
    messages = [
        {"role": "system", "content": TUTOR_SYSTEM_PROMPT.format(**lesson)},
    ]
    for m in history[-HISTORY_CONTEXT_TURNS:]:
        messages.append(
            {
                "role": "user" if m["role"] == "user" else "assistant",
                "content": m["text"],
            }
        )
    messages.append({"role": "user", "content": question})

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=160,  # ~2-3 sentences; keep it short for attention span
        temperature=0.6,
        timeout=REQUEST_TIMEOUT_S,
    )
    answer = (resp.choices[0].message.content or "").strip()
    return answer, (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------- #
# Session state                                                                #
# --------------------------------------------------------------------------- #


def init_state() -> None:
    """Per TICKETS.md: chat_history = [{"role", "text", "timestamp"}, ...]."""
    s = st.session_state
    s.setdefault("chat_history", [])
    s.setdefault("last_audio_hash", None)      # dedup across reruns
    s.setdefault("last_transcription", None)   # for the correction button
    s.setdefault("correction_mode", False)
    s.setdefault("pending_tts", None)          # {"text", "nonce"}
    s.setdefault("tts_autoplay", False)        # consumed on next render
    s.setdefault("latency_log", [])            # per-turn component timings
    s.setdefault("turn", 0)


def record_latency(stt_ms: float | None, llm_ms: float, total_ms: float) -> None:
    entry = {
        "turn": st.session_state.turn,
        "stt_ms": round(stt_ms) if stt_ms is not None else None,
        "llm_ms": round(llm_ms),
        "total_ms": round(total_ms),
        "over_budget": total_ms > LATENCY_BUDGET_S * 1000,
    }
    st.session_state.latency_log.append(entry)
    log.info(
        "latency turn=%d stt_ms=%s llm_ms=%.0f total_ms=%.0f over_budget=%s",
        entry["turn"],
        f"{stt_ms:.0f}" if stt_ms is not None else "n/a",
        llm_ms,
        total_ms,
        entry["over_budget"],
    )


# --------------------------------------------------------------------------- #
# Turn processing                                                              #
# --------------------------------------------------------------------------- #


def _kid_friendly_error(exc: Exception) -> str:
    """Map Groq errors to messages an 8-year-old (or her parent) can act on."""
    if isinstance(exc, RateLimitError):  # HTTP 429
        return (
            "Wow, lots of kids are learning right now! 😅 "
            "Wait a little bit, then press the microphone and ask again."
        )
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return "Hmm, I couldn't reach my brain in time. 🐢 Let's try again!"
    if isinstance(exc, AuthenticationError):  # bad/missing key — parent action needed
        return (
            "My tutor key doesn't work. 🔑 A grown-up should check that "
            "GROQ_API_KEY is set correctly (see .env.example)."
        )
    log.exception("Unexpected Groq error")
    return "Oops, something went wrong on my side. Try asking again!"


def _run_llm_turn(client: Groq, lesson: dict, question: str, stt_ms: float | None) -> None:
    """Shared tail of every turn: LLM answer → history → queue TTS → latency."""
    t0 = time.perf_counter()
    answer, llm_ms = ask_tutor(client, lesson, st.session_state.chat_history, question)
    total_ms = (time.perf_counter() - t0) * 1000 + (stt_ms or 0)

    now = time.time()
    st.session_state.chat_history.append(
        {"role": "user", "text": question, "timestamp": now}
    )
    st.session_state.chat_history.append(
        {"role": "tutor", "text": answer, "timestamp": time.time()}
    )
    st.session_state.turn += 1
    st.session_state.pending_tts = {
        "text": answer,
        "nonce": st.session_state.turn,
    }
    st.session_state.tts_autoplay = True  # one-shot; consumed by render_tts
    record_latency(stt_ms, llm_ms, total_ms)


def process_audio_turn(client: Groq, lesson: dict, audio_bytes: bytes) -> bool:
    """🎤→📝→🧠→🔊: transcribe, then delegate to the shared LLM turn.

    Returns True only when a full turn (question + answer) completed."""
    try:
        text, stt_ms = transcribe_audio(client, audio_bytes)
    except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
        st.warning(_kid_friendly_error(exc))
        return False

    if not text:
        st.info("I didn't hear anything. 👂 Can you press the microphone and try again?")
        return False

    st.session_state.last_transcription = text
    try:
        _run_llm_turn(client, lesson, text, stt_ms)
    except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
        st.warning(_kid_friendly_error(exc))
        return False
    return True


def process_text_turn(client: Groq, lesson: dict, question: str) -> bool:
    """Used by the correction button — skips Whisper, goes straight to the LLM."""
    try:
        _run_llm_turn(client, lesson, question, stt_ms=None)
    except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
        st.warning(_kid_friendly_error(exc))
        return False
    return True


# --------------------------------------------------------------------------- #
# Web Speech API TTS (browser-native, runs on Maryam's device)                 #
# --------------------------------------------------------------------------- #


def render_tts(text: str, autoplay: bool, nonce: int) -> None:
    """Render a Play button that speaks `text` via window.speechSynthesis.

    - Explicit button = the user gesture required by browser autoplay policy.
    - Autoplay is attempted ONCE per new answer (parent passes autoplay=True
      only on the render right after generation; the iframe re-runs its JS on
      every Streamlit rerun, so re-arming would re-speak constantly).
    - Voice pick: Arabic voice if the text has Arabic characters and an `ar*`
      voice exists; otherwise the browser default (documented limitation —
      Arabic voice availability varies by browser/OS).
    """
    payload = json.dumps(text)  # safe JS string literal
    html = f"""
    <div style="display:flex;align-items:center;gap:12px;font-family:sans-serif;">
      <button id="tts-play" style="
          font-size:22px;padding:14px 26px;border-radius:16px;border:none;
          background:#4CAF50;color:white;cursor:pointer;">
        🔊 Hear the answer
      </button>
      <span id="tts-status" style="font-size:14px;color:#666;"></span>
    </div>
    <script>
      const TEXT = {payload};
      const AUTOPLAY = {str(autoplay).lower()};
      const NONCE = "{nonce}";
      const hasArabic = /[\\u0600-\\u06FF]/.test(TEXT);
      const status = (m) => document.getElementById("tts-status").textContent = m;

      function pickVoice() {{
        const voices = window.speechSynthesis.getVoices() || [];
        if (!voices.length) return null;
        const pref = hasArabic ? "ar" : "en";
        return voices.find(v => (v.lang || "").toLowerCase().startsWith(pref))
               || voices.find(v => (v.lang || "").toLowerCase().startsWith("ar"))
               || null;  // fallback: browser default voice
      }}

      function speak() {{
        try {{
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance(TEXT);
          const v = pickVoice();
          if (v) {{ u.voice = v; u.lang = v.lang; }}
          else {{ u.lang = hasArabic ? "ar-SA" : "en-US"; }}
          u.rate = 0.95;  // slightly slower for an 8-year-old
          u.onend = () => status("");
          window.speechSynthesis.speak(u);
          status("Playing…");
        }} catch (e) {{
          status("Tap the green button to hear me.");
        }}
      }}

      document.getElementById("tts-play").onclick = speak;

      if (AUTOPLAY) {{
        // getVoices() is async in Chrome: wait for voices before speaking,
        // otherwise the Arabic voice may not be found on first load.
        if ((window.speechSynthesis.getVoices() || []).length) {{
          speak();
        }} else {{
          window.speechSynthesis.onvoiceschanged = () => {{ speak(); }};
        }}
      }}
    </script>
    """
    components.html(html, height=80)


# --------------------------------------------------------------------------- #
# UI                                                                           #
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title="Maryam's AI Tutor — Voice Loop", page_icon="🎤")
    st.title("🎤 Maryam's AI Tutor — Voice Loop (Ticket #1)")

    init_state()

    api_key = get_api_key()
    if not api_key:
        st.error(
            "No Groq API key found. 🔑\n\n"
            "- **Local:** copy `.env.example` to `.env` and set `GROQ_API_KEY`\n"
            "- **Streamlit Cloud:** add `GROQ_API_KEY` in the app's Secrets\n\n"
            "Get a free key at https://console.groq.com/keys"
        )
        st.stop()
    client = get_groq_client(api_key)

    lesson, source_note = load_lesson()
    st.caption(f"📖 Lesson: **{lesson['title']}** · source: {source_note}")

    # --- 1. Audio capture (processed BEFORE rendering history so the new ---
    # ---     turn shows up in this same run)                             ---
    audio = st.audio_input("🎤 Press the microphone and ask your question")
    if audio is not None:
        audio_bytes = audio.getvalue()
        if audio_bytes:
            digest = hashlib.sha256(audio_bytes).hexdigest()
            # st.audio_input re-returns the same clip on EVERY rerun.
            # Hash-dedup stops us re-transcribing (and 429-ing) on each click.
            if digest != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = digest
                process_audio_turn(client, lesson, audio_bytes)

    # --- 2. Chat history ----------------------------------------------------
    st.subheader("💬 Our chat")
    if not st.session_state.chat_history:
        st.write("No questions yet — press the microphone and ask me something! 😊")
    for msg in st.session_state.chat_history:
        who = "user" if msg["role"] == "user" else "assistant"
        name = "Maryam" if who == "user" else "Tutor"
        with st.chat_message(who):
            st.markdown(f"**{name}:** {msg['text']}")

    # --- 3. Transcription display + correction button -----------------------
    if st.session_state.last_transcription and not st.session_state.correction_mode:
        st.info(f'📝 You said: "{st.session_state.last_transcription}"')
        if st.button("🙋‍♀️ That's not what I said!"):
            st.session_state.correction_mode = True
            st.rerun()

    if st.session_state.correction_mode:
        with st.form("correction_form"):
            st.write("No problem! Type what you really said:")
            corrected = st.text_input("What you really said", label_visibility="collapsed")
            sent = st.form_submit_button("✅ Send my correction")
        if sent and corrected.strip():
            question = corrected.strip()
            # Replace the misheard turn (last user+tutor pair) with the fix.
            hist = st.session_state.chat_history
            if len(hist) >= 2 and hist[-1]["role"] == "tutor" and hist[-2]["role"] == "user":
                del hist[-2:]
            st.session_state.last_transcription = question
            st.session_state.correction_mode = False
            # Only rerun on success — otherwise the fresh error message
            # rendered above would be wiped before Maryam can read it.
            if process_text_turn(client, lesson, question):
                st.rerun()

    # --- 4. TTS playback (Web Speech API) ------------------------------------
    if st.session_state.pending_tts:
        # Autoplay exactly once: the flag is set when a new answer is
        # generated and consumed here (component JS re-runs every rerun).
        autoplay = st.session_state.tts_autoplay
        st.session_state.tts_autoplay = False
        render_tts(
            st.session_state.pending_tts["text"],
            autoplay=autoplay,
            nonce=st.session_state.pending_tts["nonce"],
        )

    # --- 5. Latency log (debug / post-mortem) --------------------------------
    with st.expander("⏱️ Latency log (target < 10 s per turn)"):
        if st.session_state.latency_log:
            st.table(st.session_state.latency_log[-10:])
        else:
            st.caption("No turns yet. Timings also go to the server console log.")


if __name__ == "__main__":
    main()
