"""
Ticket #1 — Voice I/O Integration  +  Ticket #3 — Streamlit Skeleton
Maryam's AI Lesson Review Tutor (age 8)

Full voice loop:
    🎤 record  →  📝 Whisper transcription (Groq hosted API)
    →  🧠 Groq LLM tutor response  →  🔊 Web Speech API read-aloud

Ticket #3 assembles the app around that loop: a lesson picker (populated
from lessons.json, shipped by ticket #2), a kid-friendly sectioned layout,
and per-lesson session state (switching lessons wipes the chat so contexts
never bleed across subjects).

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

Scope: voice loop (#1) + lesson picker & page skeleton (#3). Quiz = #4,
deployment/secrets wiring = #5.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import os
import re
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

# Arabic-focused model on Groq free tier; perfect for bilingual tutor (English + Arabic).
# Override via LLM_MODEL env var if needed.
LLM_MODEL = os.getenv("LLM_MODEL", "allam-2-7b")

REQUEST_TIMEOUT_S = 30          # per-request timeout for Groq calls
HISTORY_CONTEXT_TURNS = 6       # how much chat history the tutor sees
LATENCY_BUDGET_S = 10           # per-turn budget from TICKETS.md (log warning if over)

LESSONS_PATH = Path(__file__).with_name("lessons.json")

# Placeholder used ONLY if lessons.json is missing/malformed, so the voice
# loop stays testable on its own (required fallback from ticket #1).
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

# Kid-friendly visual cues for the picker dropdown and lesson caption.
SUBJECT_EMOJI = {"Math": "🔢", "Arabic": "🌙"}
DIFFICULTY_EMOJI = {"easy": "🌱", "medium": "🌿", "hard": "🌳"}

# Quiz (ticket #4): 5 MCQs per lesson, generator + independent validator.
QUIZ_QUESTION_COUNT = 5       # questions per quiz (locked by TICKETS.md)
QUIZ_OPTIONS_COUNT = 4        # A/B/C/D
QUIZ_MAX_TOKENS = 2048        # Arabic answers are token-heavy; leave headroom
VALIDATOR_MAX_TOKENS = 220    # short chain-of-thought + last-line answer

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
# Lessons (ticket #2 shipped lessons.json; ticket #3 loads the list for UI)    #
# --------------------------------------------------------------------------- #


def load_lesson_list() -> tuple[list[dict], str]:
    """Return (lessons, source_note) for the lesson picker.

    Reads lessons.json (ticket #2's deliverable). Falls back to a single
    placeholder lesson when the file is missing, empty, or malformed, so
    the app still boots and the voice loop stays testable on its own.
    Lessons without an id/title are dropped — the picker needs both.
    """
    if LESSONS_PATH.exists():
        try:
            data = json.loads(LESSONS_PATH.read_text(encoding="utf-8"))
            lessons = [
                l
                for l in data.get("lessons", [])
                if l.get("id") and l.get("title")
            ]
            if lessons:
                return lessons, f"lessons.json ({len(lessons)} lessons)"
            log.warning("lessons.json has no lessons; using placeholder")
        except Exception as exc:  # malformed JSON etc.
            log.warning("Could not parse lessons.json (%s); using placeholder", exc)
    return [PLACEHOLDER_LESSON], "placeholder (lessons.json missing)"


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
# Quiz generation + validation (ticket #4)                                     #
# --------------------------------------------------------------------------- #

QUIZ_GENERATOR_PROMPT = """You are a kind teacher writing a quiz for Maryam, an 8-year-old girl, about the lesson below.

{language_note}

Write {n} multiple-choice questions.

Rules:
- Use simple words an 8-year-old knows. No trick questions. Each question has exactly ONE correct answer.
- All {k} options must be plausible — none may be silly or obviously wrong.
- NEVER use "all of the above" or "none of the above" (or the Arabic equivalent) as an option.
- Questions must come ONLY from the lesson content below — nothing outside it.
- The correct answer must NOT simply repeat the question's own words (no "What are big numbers called? → Big numbers").
- Cover DIFFERENT parts of the lesson — never ask the same thing twice.
- If the lesson is about math: ask a calculation question ONLY when the exact calculation AND its final answer are both written in the lesson text. Never invent a new calculation.
- "correct" must be copied EXACTLY from one of the options.
- "explanation" is one short, kind sentence (15 words max) saying why the answer is right.

Reply with ONLY a JSON array (no markdown, no extra words), like:
[{{"question": "...", "options": ["...", "...", "...", "..."], "correct": "...", "explanation": "..."}}]

IMPORTANT for valid JSON: NEVER use the " character inside any text. If you need to quote something inside a question or explanation, use « » instead.

Here is ONE example of a good question (from a different lesson — just to show the style):
{{"question": "What is the place value of 4 in 3,456?", "options": ["4", "400", "40", "4000"], "correct": "400", "explanation": "The 4 is in the hundreds place, so it is worth 400."}}

Lesson title: {title}
Lesson content: {content}"""

# Hard language instruction: allam-2-7b is Arabic-dominant and drifts into
# Arabic unless told explicitly which language to stay in.
QUIZ_LANGUAGE_NOTES = {
    True: "LANGUAGE RULE: The lesson is in ARABIC. Write every question, option and explanation in ARABIC ONLY — لا تكتب أي كلمة بالإنجليزية.",
    False: "LANGUAGE RULE: The lesson is in ENGLISH. Write every question, option and explanation in ENGLISH ONLY — do not use a single Arabic word anywhere.",
}

QUIZ_VALIDATOR_PROMPT = """You are checking a quiz question for an 8-year-old. Read the lesson, then solve the question BY YOURSELF, using ONLY the lesson.

Lesson title: {title}
Lesson content: {content}

Question: {question}
Options:
{options}

Think step by step in 1-3 short sentences, using ONLY the lesson. Then, on the LAST line of your reply, write ONLY ONE of these:
- the exact text of the correct option (no option numbers, no quotes), or
- the word UNANSWERABLE — if the question cannot be answered from the lesson, is unclear, has no correct option, has MORE THAN ONE correct option (ambiguous), or asks for a calculation whose final answer is not written in the lesson.
If the options are in Arabic, write the Arabic option exactly as written on the last line."""

_ARABIC_RE = re.compile(r"[؀-ۿ]")

# Detects calculation questions ("389 + 271", "sum of 389 and 271", جمع/زائد).
# A 7B generator happily invents sums and marks WRONG answers, and a 7B
# validator estimates the same wrong answer — so we don't trust prompts here:
# any calculation whose operands or answer are not literally written in the
# lesson text is dropped by _clean_question, deterministically.
_CALC_RE = re.compile(
    r"([0-9][0-9,]*)\s*(?:\+|plus|and|زائد)\s*([0-9][0-9,]*)"
    r"|(?:sum of|total of|add|مجموع|ناتج جمع)\s+([0-9][0-9,]*)\s+(?:and|و)\s+([0-9][0-9,]*)",
    re.IGNORECASE,
)


def _calc_is_grounded(question_text: str, correct: str, lesson_content: str) -> bool:
    """True unless the question asks for a calculation that the lesson does
    not fully work out (operands AND the numeric answer must all appear in
    the lesson text)."""
    m = _CALC_RE.search(question_text)
    if not m:
        return True
    haystack = lesson_content.replace(",", "")
    for operand in (g for g in m.groups() if g):
        if operand.replace(",", "") not in haystack:
            return False
    numeric_correct = correct.replace(",", "")
    if numeric_correct.isdigit() and numeric_correct not in haystack:
        return False
    return True


def _too_similar(a: dict, b: dict) -> bool:
    """Semantic-duplicate detection the exact-match dedup misses
    ('How many digits...' vs 'How many places...'). High word overlap alone
    is enough; moderate overlap is only redundant if the ANSWER also matches
    (reworded same question, same takeaway)."""
    wa = set(_normalize_answer(a["question"]).split())
    wb = set(_normalize_answer(b["question"]).split())
    if not wa or not wb:
        return False
    jaccard = len(wa & wb) / len(wa | wb)
    if jaccard >= 0.6:
        return True
    if jaccard >= 0.4 and _normalize_answer(a["correct"]) == _normalize_answer(b["correct"]):
        return True
    return False


def _lesson_is_arabic(lesson: dict) -> bool:
    """Arabic lessons get Arabic quizzes (subject tag first, content as backup)."""
    if lesson.get("subject") == "Arabic":
        return True
    return bool(_ARABIC_RE.search(lesson.get("content", "")))


_ARABIC_DIACRITICS_RE = re.compile(r"[ً-ْٰـ]")


def _norm_light(text: str) -> str:
    """Whitespace/case-only normalization. Used for duplicate-option detection,
    where diacritics MUST stay: بَ / بُ / بِ are different options (they are
    the whole point of the harakat lesson), not duplicates."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison: strip quotes, a leading
    option number ('2.' / '2)'), Arabic diacritics/tatweel, extra whitespace,
    case and trailing punctuation (English + Arabic)."""
    t = _ARABIC_DIACRITICS_RE.sub("", text)
    t = t.strip().strip('"\'`').strip()
    t = re.sub(r"^[0-9]+[.)\-:]\s*", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.rstrip(".!؟?،,;:").strip().lower()


def _answers_match(validator_answer: str, question: dict) -> bool:
    """True if the validator's independent answer agrees with the marked one."""
    norm = _normalize_answer(validator_answer)
    correct = _normalize_answer(question["correct"])
    if norm == correct:
        return True
    # Validator replied with just an option number ("2" / "option 2")?
    digits = re.sub(r"[^0-9]", "", validator_answer)
    if digits and not any(ch.isdigit() for ch in question["correct"]):
        idx = int(digits) - 1
        if 0 <= idx < len(question["options"]):
            return _normalize_answer(question["options"][idx]) == correct
    return False


def _extract_json_array(text: str) -> list:
    """Pull the JSON array of questions out of an LLM reply (tolerates
    markdown fences and chatter). If the array is broken — Arabic output
    sometimes contains an unescaped quote — salvage whatever individual
    question objects still parse (they contain no nested braces)."""
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            log.info("quiz: array parse failed, salvaging individual objects")
    items = [
        obj
        for chunk in re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if isinstance((obj := _try_json(chunk)), dict)
    ]
    if items:
        return items
    raise ValueError("no parseable questions found in generator reply")


def _try_json(chunk: str):
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None


def _clean_question(raw: dict, lesson_arabic: bool, lesson_content: str) -> dict | None:
    """Structural check + fix-ups for one generated question.

    Returns a normalized question dict, or None if it is unusable (missing
    keys, wrong option count, duplicate/empty options, correct not in options,
    wrong language for the lesson, or an invented calculation).
    """
    if not isinstance(raw, dict):
        return None
    q_text = str(raw.get("question") or "").strip()
    options = [str(o).strip() for o in raw.get("options") or [] if str(o).strip()]
    correct = str(raw.get("correct") or "").strip()
    explanation = str(raw.get("explanation") or "").strip()
    if not q_text or len(options) != QUIZ_OPTIONS_COUNT or not correct:
        return None
    if len({_norm_light(o) for o in options}) != len(options):
        return None  # duplicate options
    # Format noise the model leaks in, e.g. "What is X? (4 points)"
    q_text = re.sub(r"\s*\(\d+\s*points?\)", "", q_text).strip()
    # Language gate: the model loves mixing Arabic into English quizzes.
    # Digits are language-neutral, so only check texts that contain letters.
    if lesson_arabic:
        if not _ARABIC_RE.search(q_text):
            return None  # Arabic lesson needs an Arabic question
    else:
        if any(_ARABIC_RE.search(t) for t in [q_text, *options, explanation]):
            return None  # English lesson must be 100% Arabic-free
    if not _calc_is_grounded(q_text, correct, lesson_content):
        return None  # invented calculation → untrustworthy marked answer
    # 'correct' must be one of the options (tolerate case/whitespace drift:
    # snap it to the option's exact spelling so radio-value comparison works).
    for opt in options:
        if _normalize_answer(opt) == _normalize_answer(correct):
            correct = opt
            break
    else:
        return None
    if not explanation:
        explanation = f"The right answer is {correct}."
    return {
        "question": q_text,
        "options": options,
        "correct": correct,
        "explanation": explanation,
    }


def _call_with_rate_limit_retry(fn, retries: int = 2, sleep_s: float = 8.0):
    """Run a Groq call, absorbing transient 429s with a short wait.

    The free tier caps allam-2-7b at 6000 tokens/min; a quiz round (1
    generation + 5 validations) can brush that limit, so instead of failing
    the whole quiz on a momentary 429 we wait out the window and retry.
    Persistent failures are re-raised for the UI's kid-friendly mapping.
    """
    for i in range(retries + 1):
        try:
            return fn()
        except RateLimitError:
            if i == retries:
                raise
            log.info("quiz: hit 429, waiting %.0fs before retry", sleep_s)
            time.sleep(sleep_s)


def validate_question(client: Groq, question_dict: dict, lesson: dict) -> bool:
    """Have a second LLM solve the question independently.

    Returns True only if the validator's answer matches the marked correct
    answer (ticket #4 validation pass: ship only questions both LLMs agree on).
    """
    options_block = "\n".join(
        f"{i + 1}. {opt}" for i, opt in enumerate(question_dict["options"])
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": QUIZ_VALIDATOR_PROMPT.format(
                    title=lesson.get("title", ""),
                    content=lesson.get("content", ""),
                    question=question_dict["question"],
                    options=options_block,
                ),
            },
            {"role": "user", "content": "What is the correct option? Reply with its exact text only."},
        ],
        max_tokens=VALIDATOR_MAX_TOKENS,
        temperature=0.0,  # deterministic: we want its best single answer
        timeout=REQUEST_TIMEOUT_S,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Chain-of-thought reply: the answer is on the LAST non-empty line.
    answer = next((l.strip() for l in reversed(raw.splitlines()) if l.strip()), "")
    if "unanswerable" in answer.lower():
        log.info("quiz validation GROUNDING fail — q=%r", question_dict["question"][:80])
        return False
    agreed = _answers_match(answer, question_dict)
    if not agreed:
        log.info(
            "quiz validation DISAGREE — q=%r marked=%r validator=%r",
            question_dict["question"][:80],
            question_dict["correct"][:40],
            answer[:40],
        )
    return agreed


def generate_quiz_questions(
    client: Groq, lesson: dict, max_attempts: int = 5
) -> list:
    """Generate 5 MCQ questions from the lesson with a validation pass.

    Each attempt asks the generator LLM for the questions still needed, keeps
    only structurally sound ones, then has the validator LLM independently
    solve each; disagreements are silently dropped and retried on the next
    attempt. Returns up to QUIZ_QUESTION_COUNT validated questions (possibly
    fewer — or an empty list — if attempts run out). Groq API errors are
    raised for the caller to map to a kid-friendly message.
    """
    validated: list[dict] = []
    seen: set[str] = set()
    lesson_arabic = _lesson_is_arabic(lesson)
    for attempt in range(1, max_attempts + 1):
        needed = QUIZ_QUESTION_COUNT - len(validated)
        if needed <= 0:
            break
        # Overgenerate a little: the validator rejects weak questions, and
        # spares keep the yield at 5 without burning extra attempts.
        ask = needed + 3
        resp = _call_with_rate_limit_retry(
            lambda: client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": QUIZ_GENERATOR_PROMPT.format(
                        language_note=QUIZ_LANGUAGE_NOTES[lesson_arabic],
                        n=ask,
                        k=QUIZ_OPTIONS_COUNT,
                        title=lesson.get("title", ""),
                        content=lesson.get("content", ""),
                    ),
                },
                {
                    "role": "user",
                    "content": f"Write the {ask} quiz questions now. JSON array only.",
                },
            ],
            max_tokens=QUIZ_MAX_TOKENS,
            temperature=0.4,  # conservative + lesson-grounded beats creative here
            timeout=REQUEST_TIMEOUT_S,
            )
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            batch = _extract_json_array(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning("quiz attempt %d: unparseable reply (%s)", attempt, exc)
            continue
        candidates: list[dict] = []
        for item in batch:
            question = _clean_question(item, lesson_arabic, lesson.get("content", ""))
            if question is None:
                log.info("quiz attempt %d: dropped malformed/wrong-language/ungrounded question", attempt)
                continue
            key = _normalize_answer(question["question"])
            if key in seen:
                continue
            # Check validated questions AND this batch's candidates: intra-batch
            # semantic duplicates must not double-ship either.
            if any(
                _too_similar(question, other)
                for other in (*validated, *candidates)
            ):
                log.info("quiz attempt %d: dropped near-duplicate question", attempt)
                continue
            seen.add(key)  # mark NOW: intra-batch duplicates must not double-ship
            candidates.append(question)
        # Validate the batch concurrently: 5 sequential LLM calls made the
        # spinner crawl (~85 s); threads bring it back under ~30 s.
        def _check(q: dict) -> bool:
            try:
                return _call_with_rate_limit_retry(
                    lambda: validate_question(client, q, lesson)
                )
            except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError):
                raise  # real API problem → let the UI show a friendly error
            except Exception:
                log.exception("quiz validator blew up; dropping question")
                return False

        with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as pool:
            results = list(pool.map(_check, candidates))
        for question, ok in zip(candidates, results):
            if ok:
                validated.append(question)
            if len(validated) >= QUIZ_QUESTION_COUNT:
                break
        log.info(
            "quiz attempt %d for %s: %d/%d valid so far",
            attempt, lesson.get("id"), len(validated), QUIZ_QUESTION_COUNT,
        )
    return validated[:QUIZ_QUESTION_COUNT]


def init_state() -> None:
    """Per TICKETS.md: chat_history = [{"role", "text", "timestamp"}, ...]."""
    s = st.session_state
    s.setdefault("chat_history", [])
    s.setdefault("last_lesson_id", None)       # picker change detection (#3)
    s.setdefault("last_audio_hash", None)      # dedup across reruns
    s.setdefault("last_transcription", None)   # for the correction button
    s.setdefault("correction_mode", False)
    s.setdefault("pending_tts", None)          # {"text", "nonce"}
    s.setdefault("tts_autoplay", False)        # consumed on next render
    s.setdefault("latency_log", [])            # per-turn component timings
    s.setdefault("turn", 0)
    # --- Quiz (ticket #4) ---------------------------------------------------
    s.setdefault("quiz_cache", {})             # {lesson_id: [validated questions]}
    s.setdefault("quiz_in_progress", False)    # user is taking the quiz now
    s.setdefault("quiz_answers", {})           # {q_index: selected_option}
    s.setdefault("quiz_score", None)           # {"correct": n, "total": n} after submit
    s.setdefault("quiz_nonce", 0)              # bumps per quiz → fresh radio keys


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


def start_quiz(client: Groq, lesson: dict, force_new: bool = False) -> None:
    """Begin a quiz for the picked lesson: reuse the cached questions unless
    `force_new` (Retake button) — otherwise generate + validate a fresh set,
    cache them per lesson, reset per-quiz state, and rerun into the quiz view.
    Generation/validation failures surface as kid-friendly messages; the
    validator's silent retries never reach the UI (ticket #4 error handling).
    """
    lesson_id = lesson["id"]
    questions = None if force_new else st.session_state.quiz_cache.get(lesson_id)
    if questions is None:
        with st.spinner("Let me make a quiz just for you... 🎨"):
            try:
                questions = generate_quiz_questions(client, lesson)
            except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
                st.warning(_kid_friendly_error(exc))
                return
            except Exception:
                log.exception("quiz generation failed")
                st.warning("Hmm, my quiz maker got stuck. 😅 Let's try again in a little bit!")
                return
        if not questions:
            st.warning("Hmm, my quiz maker got stuck. 😅 Let's try again in a little bit!")
            return
        st.session_state.quiz_cache[lesson_id] = questions
    st.session_state.quiz_nonce += 1  # fresh radio keys → no stale selections
    st.session_state.quiz_answers = {}
    st.session_state.quiz_score = None
    st.session_state.quiz_in_progress = True
    st.rerun()


def format_quiz_for_display(questions: list) -> None:
    """Render the quiz UI in Streamlit (ticket #4).

    Two views, driven by session state:
    - Not yet submitted: one form with a radio per question (a form so picking
      answers doesn't trigger reruns) + a single Submit button.
    - After submission: score banner, then per-question feedback — green with
      the explanation if correct, the right answer + explanation if wrong.
    Radio keys carry the quiz nonce so a regenerated quiz never inherits
    stale selections from the previous one.
    """
    nonce = st.session_state.quiz_nonce
    score = st.session_state.quiz_score

    if score is None:  # --- answering view -----------------------------------
        with st.form(f"quiz_form_{nonce}"):
            for i, q in enumerate(questions):
                st.markdown(f"**{i + 1}. {q['question']}**")
                st.radio(
                    f"Options for question {i + 1}",
                    q["options"],
                    key=f"quiz_q_{nonce}_{i}",
                    index=None,  # nothing pre-selected
                    label_visibility="collapsed",
                )
                if i < len(questions) - 1:
                    st.write("")
            submitted = st.form_submit_button("✅ Submit my answers!")
        if submitted:
            answers = {
                i: st.session_state.get(f"quiz_q_{nonce}_{i}")
                for i in range(len(questions))
            }
            if any(a is None for a in answers.values()):
                st.warning("Oops! Answer every question first, then press submit. 😊")
            else:
                correct_n = sum(
                    1 for i, q in enumerate(questions) if answers[i] == q["correct"]
                )
                st.session_state.quiz_answers = answers
                st.session_state.quiz_score = {
                    "correct": correct_n,
                    "total": len(questions),
                }
                st.session_state.quiz_in_progress = False
                st.rerun()
        return

    # --- results view --------------------------------------------------------
    answers = st.session_state.quiz_answers
    n_ok, n_all = score["correct"], score["total"]
    if n_ok == n_all:
        st.success(f"🌟 PERFECT, Maryam! You got {n_ok} out of {n_all}! You're a star!")
    elif n_ok >= n_all / 2:
        st.success(f"🎉 Great job! You got {n_ok} out of {n_all}!")
    else:
        st.info(f"💪 Good try! You got {n_ok} out of {n_all}. Let's peek at the tricky ones!")

    for i, q in enumerate(questions):
        picked = answers.get(i)
        st.markdown(f"**{i + 1}. {q['question']}**")
        if picked == q["correct"]:
            st.success(f"✅ Your answer: {picked} — right! {q['explanation']}")
        else:
            st.error(f"❌ Your answer: {picked}")
            st.info(f"💡 The right answer is **{q['correct']}**. {q['explanation']}")


# --------------------------------------------------------------------------- #
# UI                                                                           #
# --------------------------------------------------------------------------- #


def inject_kid_friendly_css() -> None:
    """Ticket #3 polish: big readable text (16px+), chunky tappable buttons,
    high contrast, and bidi-aware paragraphs so Arabic chat/lessons render
    right-to-left automatically while English stays left-to-right."""
    st.markdown(
        """
        <style>
          /* Larger base text for young readers. */
          html, body, .stApp,
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stChatMessageContent"] p,
          [data-testid="stCaptionContainer"] {
              font-size: 18px !important;
              line-height: 1.6 !important;
          }
          /* Browser picks text direction per paragraph → Arabic renders RTL,
             English stays LTR, mixed chats look right. */
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stChatMessageContent"] p {
              unicode-bidi: plaintext;
          }
          /* Big, bright, rounded buttons: one clear action per tap. */
          .stButton > button, .stFormSubmitButton > button {
              font-size: 18px !important;
              padding: 0.8rem 1.6rem !important;
              border-radius: 14px !important;
              background-color: #FFD54F !important;  /* bright yellow */
              color: #4E342E !important;             /* dark brown text */
              border: none !important;
          }
          /* Roomier labels on the picker, mic widget and inputs. */
          [data-testid="stSelectbox"] label p,
          [data-testid="stAudioInput"] label p,
          [data-testid="stSelectbox"] [data-baseweb="select"] * {
              font-size: 18px !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    # Centered column reads well on both phone and tablet (Ticket #3).
    st.set_page_config(
        page_title="Maryam's AI Tutor",
        page_icon="📚",
        layout="centered",
    )
    inject_kid_friendly_css()

    st.title("📚 Maryam's AI Tutor")
    st.caption("Pick a lesson, press the microphone, and ask me anything about it! 🎉")

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

    # --- 1. Lesson picker (Ticket #3) ----------------------------------------
    lessons_list, source_note = load_lesson_list()
    selected_lesson = st.selectbox(
        "📖 Pick a lesson:",
        options=lessons_list,
        format_func=lambda l: (
            f"{SUBJECT_EMOJI.get(l.get('subject'), '📘')} "
            f"{l['title']} ({l.get('subject', '?')})"
        ),
        key="lesson_selector",
    )

    # New lesson = fresh context. Wipe the chat AND the per-turn UI state
    # (transcription, correction form, queued TTS) so nothing from the old
    # lesson bleeds into the new one — TICKETS.md: "Clear session history
    # when lesson changes (avoid cross-lesson confusion)".
    if st.session_state.last_lesson_id != selected_lesson["id"]:
        st.session_state.chat_history = []
        st.session_state.last_transcription = None
        st.session_state.correction_mode = False
        st.session_state.pending_tts = None
        st.session_state.tts_autoplay = False
        # Reset the active quiz too — but NOT quiz_cache: a quiz already made
        # for lesson A must still be there when Maryam comes back to it (#4).
        st.session_state.quiz_in_progress = False
        st.session_state.quiz_answers = {}
        st.session_state.quiz_score = None
        # last_audio_hash is kept ON PURPOSE: the mic widget still holds the
        # previous clip after a lesson switch, and re-hashing it must NOT
        # re-transcribe that clip in the new lesson's context.
        st.session_state.last_lesson_id = selected_lesson["id"]

    lesson = selected_lesson  # from here on, the whole app uses the picked lesson
    difficulty = lesson.get("difficulty", "easy")
    st.caption(
        f"{SUBJECT_EMOJI.get(lesson.get('subject'), '📘')} **{lesson.get('subject', '?')}** · "
        f"{DIFFICULTY_EMOJI.get(difficulty, '🌱')} {difficulty} · "
        f"source: {source_note}"
    )

    st.divider()

    # --- 2. Audio section: mic → transcription → correction -----------------
    # Audio is processed BEFORE rendering the chat so the new turn shows up
    # in this same run (Streamlit reruns top-to-bottom on every interaction).
    st.subheader("🎤 Ask your question")
    audio = st.audio_input("Press the microphone and ask your question")
    if audio is not None:
        audio_bytes = audio.getvalue()
        if audio_bytes:
            digest = hashlib.sha256(audio_bytes).hexdigest()
            # st.audio_input re-returns the same clip on EVERY rerun.
            # Hash-dedup stops us re-transcribing (and 429-ing) on each click.
            if digest != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = digest
                process_audio_turn(client, lesson, audio_bytes)

    # Transcription display + correction button (part of the audio flow).
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

    st.divider()

    # --- 3. Chat section ------------------------------------------------------
    st.subheader("💬 Our chat")
    if not st.session_state.chat_history:
        st.write("No questions yet — press the microphone and ask me something! 😊")
    for msg in st.session_state.chat_history:
        who = "user" if msg["role"] == "user" else "assistant"
        name = "Maryam" if who == "user" else "Tutor"
        with st.chat_message(who):
            st.markdown(f"**{name}:** {msg['text']}")

    st.divider()

    # --- 4. Quiz section (Ticket #4) ------------------------------------------
    # Fallback (gate 6): the placeholder lesson means lessons.json is missing,
    # so quiz generation is disabled rather than quizzing on fake content.
    is_placeholder = lesson["id"] == PLACEHOLDER_LESSON["id"]
    st.subheader("📋 Quiz time!")
    if is_placeholder:
        st.caption("Quizzes need the real lessons (lessons.json) — ask a grown-up! 🌱")
    elif st.session_state.quiz_in_progress or st.session_state.quiz_score is not None:
        questions = st.session_state.quiz_cache.get(lesson["id"], [])
        if questions:
            format_quiz_for_display(questions)
            if st.session_state.quiz_score is not None:  # results → next actions
                col_retake, col_done = st.columns(2)
                if col_retake.button("🔁 Retake Quiz (new questions)"):
                    start_quiz(client, lesson, force_new=True)
                if col_done.button("✅ Done"):
                    st.session_state.quiz_score = None
                    st.session_state.quiz_answers = {}
                    st.rerun()
        else:  # cache lost (shouldn't happen) — bail out cleanly
            st.session_state.quiz_in_progress = False
            st.session_state.quiz_score = None
            st.rerun()
    else:
        if st.session_state.quiz_cache.get(lesson["id"]):
            st.caption("Your quiz for this lesson is ready — same questions as before!")
        if st.button("📋 Take a Quiz"):
            start_quiz(client, lesson)

    st.divider()

    # --- 5. TTS section (Web Speech API) --------------------------------------
    if st.session_state.pending_tts:
        st.subheader("🔊 Listen to the answer")
        # Autoplay exactly once: the flag is set when a new answer is
        # generated and consumed here (component JS re-runs every rerun).
        autoplay = st.session_state.tts_autoplay
        st.session_state.tts_autoplay = False
        render_tts(
            st.session_state.pending_tts["text"],
            autoplay=autoplay,
            nonce=st.session_state.pending_tts["nonce"],
        )

    st.divider()

    # --- 6. Footer: latency log (debug / post-mortem) --------------------------
    with st.expander("⏱️ Latency log (target < 10 s per turn)"):
        if st.session_state.latency_log:
            st.table(st.session_state.latency_log[-10:])
        else:
            st.caption("No turns yet. Timings also go to the server console log.")


if __name__ == "__main__":
    main()
