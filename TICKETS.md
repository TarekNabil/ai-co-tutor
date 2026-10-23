# Maryam's AI Lesson Review Tutor — Detailed Tickets

**Project:** Hacktoberfest 2026 — Build for a Friend with Open-Source AI
**User:** Maryam, age 8
**Theme:** Voice-driven homework tutor (Math + Arabic lessons)
**Stack:** Groq (open LLM) + Whisper (open STT) + Web Speech API (browser TTS) + Streamlit (open framework)

---

## Ticket #1: Voice I/O Integration

**Type:** Prototype  
**Effort:** 4–6 hours  
**Blocked by:** None  
**Blocks:** #3 (Streamlit Skeleton)  
**Priority:** Critical — this is the core product

### Description
Implement the full voice loop: user speaks → Whisper transcription → Groq LLM response → Web Speech API reads aloud. Test specifically on Streamlit Cloud (not just local).

### Technical Spec

**Transcription (Whisper via Groq):**
- Use Groq's hosted Whisper API (not local weights; avoids cold-start on Cloud)
- Groq API key via `GROQ_API_KEY` env var (secrets, never in repo)
- Handle 429 rate-limit errors gracefully ("Let's try again!")
- Capture audio via `st.audio_input()` or `streamlit-mic-recorder` component
- **Test both on Streamlit Cloud specifically** — st.audio_input is flaky; custom recorders break in deployment environments
- Display transcribed text to Maryam: "You said: [text]" with a "that's not what I said!" correction button

**LLM (Groq):**
- System prompt: tutor persona, stay on-lesson, explain clearly for age 8
- Input: lesson context + Maryam's question
- Output: 2–3 sentence response (keep attention span in mind)
- Handle Groq timeouts gracefully

**Text-to-Speech (Web Speech API):**
- Browser-native `speechSynthesis` API (zero server cost, works on device)
- Select Arabic voice if available (`ar-SA` or similar)
- Add explicit user gesture requirement (big "Play" button) to work around browser autoplay policy
- Fallback: if no Arabic voice, use available TTS (document limitation)

**Session State:**
- Use `st.session_state` to maintain chat history within a session
- Structure: `chat_history` list of `{"role": "user"|"tutor", "text": "...", "audio_url": "..."}`
- Handle reruns correctly (Streamlit reruns entire script on each interaction)

**Latency Budget:**
- Target: < 10 seconds per turn (record + upload + STT + Groq + TTS + playback)
- Log latency per component (useful for post-mortem)

### Acceptance Criteria
- [ ] Whisper transcription works on Cloud without local weights
- [ ] Groq LLM responds to lesson questions (tutor mode confirmed)
- [ ] Web Speech API reads responses aloud in browser
- [ ] Transcription text displayed with correction button
- [ ] Session state preserves chat history across reruns
- [ ] 429 rate limits handled gracefully
- [ ] Tested on Streamlit Cloud (not just local)
- [ ] Latency logged per component

### Notes
- **Biggest risk:** Custom recorder components break on Cloud. Test `st.audio_input()` first (safer), then explore `streamlit-mic-recorder` if needed.
- **Arabic TTS:** Browser might not have good Arabic voice. Document fallback and note for v2 improvement.

---

## Ticket #2: Lesson Setup

**Type:** Task  
**Effort:** 2 hours  
**Blocked by:** None  
**Blocks:** #3 (Streamlit Skeleton)  
**Priority:** High — removes PDF extraction from critical path

### Description
Pre-extract Math and Arabic lessons from source PDFs at build time. Create a `lessons.json` file that the app loads on startup. Build a simple UI dropdown for Maryam to pick a lesson.

### Technical Spec

**Lesson Extraction:**
- Source files: Math PDF and Arabic PDF (assumed available in repo or specified path)
- Extract lesson names + page ranges using Table of Contents
- For Math: straightforward text extraction (likely clean)
- For Arabic: be prepared for garbled text; may need manual review or semi-automated fix
- Output format: `lessons.json`

```json
{
  "lessons": [
    {
      "id": "math_001",
      "title": "Addition Basics",
      "subject": "Math",
      "content": "Two-digit addition without regrouping. Examples: 23 + 14 = 37...",
      "difficulty": "easy"
    },
    {
      "id": "arabic_001",
      "title": "حروف الهجاء (Arabic Letters)",
      "subject": "Arabic",
      "content": "الألف، الباء، التاء... [Arabic lesson content]",
      "difficulty": "easy"
    }
  ]
}
```

**Lesson Content:**
- Extract or transcribe the core learning material (200–400 words per lesson, age-appropriate)
- Include key terms, definitions, examples
- Content will be passed to Groq for tutoring context

**UI Dropdown:**
- Streamlit `st.selectbox()` for lesson selection
- Show lesson title + subject + difficulty
- Selected lesson populates the chat context
- Clear session history when lesson changes (avoid cross-lesson confusion)

**Validation:**
- All lessons have required fields
- Arabic text renders correctly (check RTL handling in Streamlit if needed)
- No missing or broken references

### Acceptance Criteria
- [ ] `lessons.json` created with at least 2 Math and 2 Arabic lessons
- [ ] Lesson content is clear, age-appropriate, and at least 200 words
- [ ] UI dropdown renders all lessons correctly
- [ ] Arabic text displays properly (no garbled/reversed characters)
- [ ] Session history clears when lesson changes
- [ ] File deployed with app (no runtime extraction)

### Notes
- **PDF extraction tool:** Use `pdfplumber` or `PyPDF2` for extraction; if Arabic is garbled, manually review + fix.
- **v2 scope:** Dynamic PDF upload + extraction. v1 is pre-baked only.

---

## Ticket #3: Streamlit Skeleton

**Type:** Task  
**Effort:** 2–3 hours  
**Blocked by:** #1 (Voice I/O Integration), #2 (Lesson Setup)  
**Blocks:** #5 (Cloud Deploy), #6 (Test with Maryam)  
**Priority:** High — assembles the app

### Description
Create the main Streamlit app structure: page layout, lesson picker, chat interface, session state management, and integration of Voice I/O (#1) and Lesson Setup (#2).

### Technical Spec

**Page Layout:**
```
┌─────────────────────────┐
│  Maryam's AI Tutor      │
│  📖 Pick a lesson:      │
│  [Dropdown: Math/Arabic]│
├─────────────────────────┤
│ Chat History            │
│ ┌─────────────────────┐ │
│ │ Tutor: Hi Maryam!   │ │
│ │ You: What is 5 + 3? │ │
│ │ Tutor: 5 + 3 = 8... │ │
│ └─────────────────────┘ │
├─────────────────────────┤
│ 🎤 [Record]             │
│ [Play] (response audio) │
│                         │
│ You said: "what is 5+3" │
│ [That's not what I said!]
├─────────────────────────┤
│ [Quiz] [New Lesson]     │
└─────────────────────────┘
```

**Key Components:**
1. **Header:** App title + kid-friendly emoji/styling
2. **Lesson Picker:** Dropdown to select lesson (populated from `lessons.json`)
3. **Chat Display:** Conversation history (tutor + Maryam messages)
4. **Voice Input:** Mic button, recorder, transcription display
5. **Correction Button:** "That's not what I said!" to fix transcription
6. **Action Buttons:** Quiz, New Lesson, Clear

**Session State Structure:**
```python
st.session_state.current_lesson = "math_001"
st.session_state.chat_history = [
  {"role": "tutor", "text": "...", "timestamp": ...},
  {"role": "user", "text": "...", "timestamp": ...}
]
st.session_state.lesson_loaded = True
```

**Chat Loop Logic:**
1. Display lesson picker
2. Load lesson from `lessons.json` on selection
3. Initialize tutor greeting
4. Repeat:
   - User records audio
   - Transcribe (Whisper via Groq)
   - Show transcribed text
   - Get tutor response (Groq)
   - Play response (Web Speech API)
   - Add to chat history
   - Rerun

**Styling:**
- Kid-friendly colors, large text (16px+), high contrast
- Emoji for visual cues (🎤 mic, 📖 lesson, 🎉 progress)
- Responsive layout (works on tablet + mobile)
- RTL support for Arabic lesson display

**Error Handling:**
- Missing lesson: show fallback message
- Transcription empty: "I didn't hear that. Try again?"
- Groq timeout: "Let me think... (retry)"
- Audio playback fail: display tutor text fallback

### Acceptance Criteria
- [ ] App starts without errors
- [ ] Lesson picker works and loads lesson context
- [ ] Chat history displays correctly
- [ ] Session state preserved across reruns
- [ ] Voice I/O integration works (from #1)
- [ ] Lesson content displayed with context for tutor
- [ ] Error messages are kid-friendly
- [ ] Page responsive and readable on mobile/tablet
- [ ] Arabic lesson renders with correct text direction (RTL check)

### Notes
- **State management:** Streamlit reruns on every interaction. Be explicit about dependencies to avoid cascading bugs.
- **Testing:** Manually test the happy path (pick lesson → ask question → hear response) before moving to #5.

---

## Ticket #4: Quiz Authoring

**Type:** Task  
**Effort:** 1 hour  
**Blocked by:** #2 (Lesson Setup) — needs lesson content  
**Blocks:** None (can ship v1 with hand-reviewed quizzes)  
**Priority:** Medium — ships with app, but auto-gen is v2

### Description
Hand-author and validate 3–5 multiple-choice questions per lesson. Add questions to `lessons.json` or a separate `quizzes.json`. These are static, hand-reviewed questions (not auto-generated).

### Technical Spec

**Quiz Format (embed in lessons.json):**
```json
{
  "lessons": [
    {
      "id": "math_001",
      "title": "Addition Basics",
      "content": "...",
      "quiz": {
        "questions": [
          {
            "id": "q1",
            "text": "What is 5 + 3?",
            "options": ["7", "8", "9", "10"],
            "correct": "8",
            "explanation": "5 + 3 = 8. Count on your fingers!"
          }
        ]
      }
    }
  ]
}
```

**Authoring Guidelines:**
- **Difficulty:** Match lesson content (age 8, not advanced)
- **Question clarity:** Short, simple sentences
- **Distractors:** Plausible wrong answers (not absurd)
- **Explanation:** Always explain the correct answer (Maryam learns from feedback)
- **Coverage:** Mix recall + comprehension (not just "what did I say?")

**Validation:**
- [ ] Each question has exactly 4 options
- [ ] Correct answer is one of the 4 options
- [ ] Explanation is clear and reinforces the lesson
- [ ] At least 3 questions per lesson
- [ ] No typos or garbled text (especially Arabic)

**v1 Scope:**
- Hand-authored, verified, shipped with app
- No auto-generation (that's v2 after Maryam feedback)
- Questions are deterministic (same questions every session)

### Acceptance Criteria
- [ ] At least 3 questions per lesson in `lessons.json`
- [ ] All questions properly formatted
- [ ] Explanations are kid-friendly
- [ ] Arabic questions render correctly
- [ ] Correct answer marked clearly in JSON

### Notes
- **v2 scope:** Auto-generate questions with Groq + validation pass (model generates → second model solves → ship only on agreement).
- **Emotional safety:** Wrong answers should feel safe ("Try again!", not "Wrong! You're bad at math!").

---

## Ticket #5: Streamlit Cloud Deploy

**Type:** Task  
**Effort:** 1 hour  
**Blocked by:** #3 (Streamlit Skeleton)  
**Blocks:** None (can test with Maryam locally first)  
**Priority:** High — ships to public URL

### Description
Deploy the app to Streamlit Cloud. Handle secrets (Groq API key), set up dependencies, and verify the app works on public URL.

### Technical Spec

**Secrets Management:**
- Create `.streamlit/secrets.toml` (never in repo):
  ```toml
  GROQ_API_KEY = "your-key-here"
  ```
- Deploy via Streamlit Cloud UI: add secret in dashboard
- Code accesses via `st.secrets["GROQ_API_KEY"]`
- `.gitignore` includes `secrets.toml` and `.env`

**Repo Structure for Deploy:**
```
Hacktoberfest-Weekend-Challenge/
├── app.py                    # Main Streamlit app
├── lessons.json              # Pre-baked lessons
├── quizzes.json              # Or embedded in lessons.json
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit config
├── .gitignore                # Exclude secrets
└── README.md                 # Usage + attribution
```

**requirements.txt:**
```
streamlit>=1.28.0
groq>=0.4.0
pydub>=0.25.0
python-dotenv>=1.0.0
```

**Deployment Steps:**
1. Push code to GitHub (with `.gitignore` excluding secrets)
2. Log in to Streamlit Cloud (streamlit.io/cloud)
3. Create new app from repo
4. Add `GROQ_API_KEY` in Secrets section
5. Wait for deploy (2–3 min)
6. Test public URL end-to-end

**Testing on Deployed App:**
- [ ] Pick a lesson
- [ ] Record audio (test mic permissions)
- [ ] Verify transcription displays
- [ ] Tutor responds with audio
- [ ] Chat history persists
- [ ] No errors in Streamlit logs

**Cold Start Optimization:**
- Streamlit Cloud spins down after 1 hour idle
- First load after idle may be slow (acceptable for now)
- v2: consider session pooling or caching

### Acceptance Criteria
- [ ] App deployed to public Streamlit Cloud URL
- [ ] Groq API key stored in secrets (not in repo)
- [ ] Full voice loop works on deployed app
- [ ] Mic permissions flow works
- [ ] No console errors
- [ ] Repo includes `.gitignore` for secrets
- [ ] README documents how to run locally + deploy

### Notes
- **Quota risk:** Public URL + free Groq tier = anyone can use it. Monitor usage and set warnings if needed (v2).
- **Test before sharing:** Verify everything works on public URL before giving link to Maryam.

---

## Ticket #6: Test with Maryam

**Type:** Grilling / User Feedback  
**Effort:** 1–2 sessions (~1 hour total)  
**Blocked by:** #3 (Streamlit Skeleton) — can test locally over LAN  
**Blocks:** #7 (Hacktoberfest Post)  
**Priority:** Critical — the whole point

### Description
Have Maryam actually use the app. Observe and collect feedback on UX, voice quality, lesson clarity, and fun factor. Iterate based on what she says.

### Testing Plan

**Setup:**
- Deploy to Streamlit Cloud (or run locally on shared network)
- Give Maryam the URL / laptop
- Let her explore with minimal guidance
- Sit nearby (don't hover; observe passively)

**Test Scenarios:**
1. **First use:** Does she understand how to start? Does the mic button make sense?
2. **Lesson pick:** Can she select a lesson? Does the dropdown feel natural?
3. **Asking questions:** Does she understand she should speak? What does she ask about?
4. **Comprehension:** Does the tutor's explanation help? Too fast/slow/complex?
5. **Quiz:** Does she attempt the quiz? Does she understand the questions?
6. **Errors:** If something breaks (no mic, transcription fail), what does she do?

**Observation Notes:**
- Time to first interaction
- Questions she asks (reveals what she finds interesting/confusing)
- Spontaneous reactions (bored, engaged, frustrated)
- What she tries to do that the app doesn't support
- Error recovery (does she retry or give up?)

**Feedback Questions (if she's willing to talk):**
- Did the tutor help you learn?
- Was the voice easy to understand?
- Did you like the quiz?
- What was confusing?
- What would you like it to do differently?

**Post-Test Iteration:**
- Document observations
- Identify top 3 issues to fix
- Prioritize by emotional impact (e.g., if she felt dumb, fix that first)
- Minor fixes: do immediately
- Major changes: tag for v2

### Acceptance Criteria
- [ ] Maryam uses app for at least 5 minutes
- [ ] Feedback documented (what worked, what confused her)
- [ ] At least 1 iteration based on feedback (if time allows)
- [ ] Session note captured for Hacktoberfest post

### Notes
- **No pressure:** This is user research, not a grade. If she's not interested, that's valuable data too.
- **Safety first:** Ensure app is kid-safe (no inappropriate content, reasonable latency, clear error messages).
- **Real feedback > polish:** Honest "she got bored" beats "she would love it" (imagined).

---

## Ticket #7: Hacktoberfest Post

**Type:** Writing / Retrospective  
**Effort:** 1–2 hours  
**Blocked by:** #6 (Test with Maryam)  
**Blocks:** None  
**Priority:** High — ships the story

### Description
Write a Hacktoberfest challenge post explaining why open-source AI was essential for this project and what Maryam said.

### Content Structure

**Title & Hook:**
- "Building an AI Tutor for My 8-Year-Old Sister — Why Open-Source AI Was Essential"

**The Problem:**
- Maryam was struggling with math and Arabic homework
- Needed a patient, multilingual tutor
- Parents couldn't afford subscription tutoring services
- Had to be safe (no data extraction, no ads, runs anywhere)

**Why Open-Source AI Mattered (Core Narrative):**

1. **Cost:** Groq free tier + open-source stack = $0/month (closed APIs would bill per request)
2. **Privacy:** Whisper is local-runnable; voice transcription stays on device (no Maryam profiling)
3. **Hackability:** Anyone can fork, swap models, run on their own infra (teacher could self-host in a classroom)
4. **Reliability:** Not dependent on a company's API stability (teacher in offline-prone region could cache responses)

**The Build (Brief):**
- Tech stack: Groq + Whisper + Web Speech API + Streamlit
- Pre-baked lessons (vs. dynamic extraction — simpler, safer for child)
- Voice loop: listen → transcribe → respond → speak
- Deployed to Streamlit Cloud (free tier)

**Real Feedback from Maryam:**
- What she liked
- What confused her
- What she asked for
- Moments of engagement or frustration
- Quote something she actually said (if quotable)

**Lessons Learned:**
- Whisper WER on child speech is higher than adults (noted for v2)
- pyttsx3 fails on Cloud; Web Speech API is better
- Pre-baking lessons was the right call (simpler, safer, faster)
- 8-year-olds need visible text fallback (voice-only is too hard)

**What's Next (v2 Roadmap):**
- PDF upload + dynamic extraction
- Quiz auto-generation with validation
- Session persistence
- Teacher admin view (classroom deployment)

**Close:**
- Open-source AI isn't just cheaper — it's more responsible
- Building for a real person, not a hypothetical user, changes everything
- Link to repo (GitHub)
- Call to action: "If you're building AI for kids, test with real kids"

### Acceptance Criteria
- [ ] Post explains why open-source > closed for this use case
- [ ] Includes real feedback from Maryam (not imagined)
- [ ] Tech stack clearly listed (Groq, Whisper, Web Speech API, Streamlit)
- [ ] Honest about constraints (pre-baked lessons, hand-reviewed quizzes, no persistence v1)
- [ ] Includes a personal moment / quote from Maryam
- [ ] Links to repo + deployment
- [ ] 500–800 words (readable, not a novel)
- [ ] No hype; honest retrospective tone

### Notes
- **Authenticity matters:** Judges notice if you've actually built vs. hypothesized. Real Maryam feedback is gold.
- **Humble tone:** "Here's what worked, here's what broke" reads better than "Look how clever I am."
- **Open-source win:** Emphasize where open enabled something closed couldn't (cost, privacy, hackability).

---

## Execution Order

**Phase 1 (Parallel):** #1 + #2
- Voice I/O and Lesson Setup can happen in parallel (no dependencies)

**Phase 2:** #3
- Assemble skeleton once #1 + #2 done

**Phase 3 (Parallel):** #4 + #5
- Quiz authoring and deploy can happen in parallel

**Phase 4:** #6
- Test with Maryam (local or deployed)

**Phase 5:** #7
- Write post with real feedback

---

## Definition of Done (MVP)

- [ ] Voice I/O works end-to-end on Cloud
- [ ] Maryam can pick a lesson and ask questions
- [ ] Tutor responds with audio
- [ ] Quiz is attempted (at least one question)
- [ ] Maryam has tested it and given feedback
- [ ] Post published (Hacktoberfest submission)

**Not in MVP (v2):**
- PDF upload / dynamic extraction
- Quiz auto-generation
- Session persistence
- Teacher admin view
- Analytics / usage tracking

---

## Attribution

Each commit should reference this project and include:
```
Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
```

PRs should end with:
```
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

---

**Ready to start? Which ticket first?**
