# streamlit_app.py
# CFA-style Fair Value Analyst voice chatbot (Streamlit):
# - OpenRouter chat (text + optional vision)
# - URL ingestion: HTML + PDF text + PDF slide images fallback
# - Output controls: higher max_tokens, trimmed history, auto-continue for cutoffs
# - TTS: edge-tts (MP3) with timeout + toggle + longer audio char cap
# - STT: Vosk + streamlit-mic-recorder
#
# requirements.txt (minimum):
# streamlit
# requests
# beautifulsoup4
# lxml
# edge-tts
# streamlit-mic-recorder
# vosk
# numpy
# soundfile
# scipy
# pymupdf
# Pillow
#
# IMPORTANT:
# - Do NOT install "fitz" from pip. Use "pymupdf".
# - Streamlit Secrets:
#   OPENROUTER_API_KEY="..."
#   (optional) OPENROUTER_MODEL="deepseek/deepseek-v3.2"
#   (optional) OPENROUTER_VISION_MODEL="..."   # vision-capable model for slide images
#   (optional) EDGE_VOICE="en-US-JennyNeural"
#   (optional) EDGE_RATE="-10%"
#   (optional) EDGE_VOLUME="+0%"
#   (optional) VOSK_MODEL_PATH="models/vosk-model-small-en-us-0.15"
#   (optional) OPENROUTER_TIMEOUT_READ=240
#   (optional) TTS_TIMEOUT_SECONDS=25
#   (optional) MAX_TOKENS=2200
#   (optional) HISTORY_TURNS=14
#   (optional) CONTINUE_PASSES=2
#   (optional) EXTRACT_MAX_CHARS=20000
#   (optional) TTS_MAX_CHARS=1200

import os
import re
import io
import json
import asyncio
import base64
from typing import List, Dict, Any

import requests
import streamlit as st
from bs4 import BeautifulSoup

import edge_tts
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from vosk import Model, KaldiRecognizer
from streamlit_mic_recorder import mic_recorder

# PyMuPDF import guard (prevents wrong "fitz" package issue)
try:
    import fitz  # PyMuPDF provides module name 'fitz'
    if not hasattr(fitz, "open"):
        raise ImportError("fitz imported but missing fitz.open; likely wrong pip package 'fitz'. Use 'pymupdf'.")
except Exception as e:
    raise ImportError(
        "PDF support requires PyMuPDF.\n"
        "Fix requirements.txt:\n"
        "  - REMOVE: fitz\n"
        "  - ADD:    pymupdf\n"
        f"Details: {e}"
    )

# --------------------
# Page config
# --------------------
st.set_page_config(page_title="Fair Value Analyst", page_icon="📈", layout="centered")

# --------------------
# Secrets / env (safe)
# --------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)

OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = get_secret("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
OPENROUTER_VISION_MODEL = get_secret("OPENROUTER_VISION_MODEL", "")

EDGE_VOICE = get_secret("EDGE_VOICE", "en-US-JennyNeural")
EDGE_RATE = get_secret("EDGE_RATE", "-10%")
EDGE_VOLUME = get_secret("EDGE_VOLUME", "+0%")

VOSK_MODEL_PATH = get_secret("VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")

OPENROUTER_TIMEOUT_READ = int(get_secret("OPENROUTER_TIMEOUT_READ", "240"))
TTS_TIMEOUT_SECONDS = int(get_secret("TTS_TIMEOUT_SECONDS", "25"))

# Output controls (updated limits)
MAX_TOKENS = int(get_secret("MAX_TOKENS", "2200"))            # higher output limit
HISTORY_TURNS = int(get_secret("HISTORY_TURNS", "14"))        # keep last N messages (plus system)
CONTINUE_PASSES = int(get_secret("CONTINUE_PASSES", "2"))     # auto-continue attempts
EXTRACT_MAX_CHARS = int(get_secret("EXTRACT_MAX_CHARS", "20000"))
TTS_MAX_CHARS = int(get_secret("TTS_MAX_CHARS", "1200"))

# --------------------
# Endpoints
# --------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --------------------
# CFA Persona
# --------------------
PERSONA = """
ROLE (STRICT):
You are a Chartered Financial Analyst (CFA) acting as a buy-side valuation analyst.
Your job is to perform deep fair value assessment of financial assets and communicate like a professional investment memo writer.

SCOPE:
- Public equities, ETFs, bonds/credit, commodities, FX, crypto (with explicit caveats).
- You DO NOT give personalized financial advice. You provide educational, analytical fair value estimates based on stated assumptions.

STYLE:
- Clear, structured investment-memo format.
- Ask for missing inputs only if truly necessary; otherwise proceed with reasonable assumptions and state them explicitly.
- Separate facts vs assumptions vs outputs.

DISCIPLINE RULES:
- Never fabricate “latest numbers.” If the user doesn’t provide them or link/slides don’t contain them, do NOT state specific historical revenue/EPS figures.
- If data is missing, proceed with transparent placeholders and clearly label illustrative assumptions.
- Provide ranges and confidence, not certainty.
"""

# --------------------
# URL detection
# --------------------
URL_RE = re.compile(r"(https?://[^\s]+)")
def extract_urls(text: str) -> List[str]:
    return URL_RE.findall(text or "")

# --------------------
# Asset intake helper
# --------------------
TICKER_RE = re.compile(r"\b[A-Z]{1,6}(\.[A-Z]{1,3})?\b")
ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")
CUSIP_RE = re.compile(r"\b[0-9A-Z]{9}\b")

def detect_asset_type(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["bond", "ytm", "coupon", "duration", "spread", "oas", "cds"]):
        return "credit"
    if any(k in t for k in ["etf", "fund", "ucits"]):
        return "fund"
    if any(k in t for k in ["fx", "forex", "exchange rate", "spot", "forward"]):
        return "fx"
    if any(k in t for k in ["crypto", "bitcoin", "btc", "ethereum", "eth", "token"]):
        return "crypto"
    if any(k in t for k in ["oil", "gold", "silver", "commodity", "futures"]):
        return "commodity"
    if any(k in t for k in ["stock", "equity", "dcf", "wacc", "ev/ebitda", "eps", "revenue", "margin"]):
        return "equity"
    return "unknown"

def detect_currency(text: str) -> str:
    t = (text or "").upper()
    if "SGD" in t or "S$" in t:
        return "SGD"
    if "HKD" in t or "HK$" in t:
        return "HKD"
    if "USD" in t or "$" in t:
        return "USD"
    if "EUR" in t or "€" in t:
        return "EUR"
    if "GBP" in t or "£" in t:
        return "GBP"
    if "JPY" in t or "¥" in t:
        return "JPY"
    if "CNY" in t or "RMB" in t:
        return "CNY"
    return "unspecified"

def extract_identifiers(text: str) -> Dict[str, List[str]]:
    raw = text or ""
    isin = ISIN_RE.findall(raw)
    cusip = CUSIP_RE.findall(raw)
    blacklist = {"DCF", "WACC", "FCF", "FCFF", "FCFE", "EBITDA", "EPS", "PE", "EV", "IRR", "NPV", "OAS", "CDS", "YTM", "NAV"}
    tickers = [m.group(0) for m in TICKER_RE.finditer(raw)]
    tickers = [t for t in tickers if t not in blacklist]
    seen = set()
    def uniq(seq):
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    return {"tickers": uniq(tickers)[:5], "isin": uniq(isin)[:3], "cusip": uniq(cusip)[:3]}

def build_intake_preamble(user_text: str) -> str:
    asset_type = detect_asset_type(user_text)
    currency = detect_currency(user_text)
    ids = extract_identifiers(user_text)
    return f"""
[INTAKE (auto-detected, may be wrong)]
- Asset type guess: {asset_type}
- Currency mentioned: {currency}
- Identifiers found: tickers={ids.get("tickers")}, ISIN={ids.get("isin")}, CUSIP={ids.get("cusip")}
[END INTAKE]
"""

# --------------------
# PDF handling
# --------------------
def is_probably_pdf(url: str, content_type: str = "") -> bool:
    u = (url or "").lower()
    ct = (content_type or "").lower()
    return u.endswith(".pdf") or "application/pdf" in ct or "pdf" in ct

def text_quality_ok(text: str, min_chars: int = 800) -> bool:
    if not text:
        return False
    s = text.strip()
    if len(s) < min_chars:
        return False
    printable = sum(ch.isprintable() for ch in s)
    return (printable / max(1, len(s))) > 0.92

def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 12) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    pages = min(max_pages, doc.page_count)
    for i in range(pages):
        page = doc.load_page(i)
        chunks.append(page.get_text("text"))
    doc.close()
    return "\n".join(c.strip() for c in chunks if c and c.strip())

def render_pdf_pages_to_png_bytes(pdf_bytes: bytes, max_pages: int = 4, zoom: float = 2.0) -> List[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    pages = min(max_pages, doc.page_count)
    mat = fitz.Matrix(zoom, zoom)
    for i in range(pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out.append(pix.tobytes("png"))
    doc.close()
    return out

def fetch_and_extract(url: str, max_chars: int = 20000) -> Dict[str, Any]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "")
    raw = r.content
    final_url = r.url

    if is_probably_pdf(final_url, content_type) or raw[:4] == b"%PDF":
        text = extract_text_from_pdf_bytes(raw, max_pages=12)
        if text_quality_ok(text):
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...(truncated)"
            return {"type": "pdf_text", "text": text, "pdf_images": [], "meta": {"content_type": content_type, "final_url": final_url}}
        imgs = render_pdf_pages_to_png_bytes(raw, max_pages=4, zoom=2.0)
        return {"type": "pdf_images", "text": "", "pdf_images": imgs, "meta": {"content_type": content_type, "final_url": final_url}}

    # HTML path
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body
    text = main.get_text("\n") if main else soup.get_text("\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cleaned = "\n".join(lines)

    title = soup.title.get_text(strip=True) if soup.title else ""
    if title:
        cleaned = f"Title: {title}\n\n{cleaned}"

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n...(truncated)"
    return {"type": "html", "text": cleaned, "pdf_images": [], "meta": {"content_type": content_type, "final_url": final_url}}

# --------------------
# OpenRouter (trim history + higher max_tokens + auto-continue)
# --------------------
def _trimmed_messages() -> List[Dict[str, Any]]:
    msgs = st.session_state.messages
    if not msgs:
        return []
    system = msgs[0] if msgs[0].get("role") == "system" else None
    tail = msgs[-HISTORY_TURNS:] if HISTORY_TURNS > 0 else msgs
    if system and (not tail or tail[0] is not system):
        return [system] + tail
    return tail

def _openrouter_post(messages: List[Dict[str, Any]], model_name: str) -> requests.Response:
    return requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": messages,
            "temperature": 0.35,
            "max_tokens": MAX_TOKENS,  # updated
        },
        timeout=(15, OPENROUTER_TIMEOUT_READ),
    )

def _parse_reply(resp: requests.Response) -> str:
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def _is_likely_cutoff(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    # Heuristics:
    # - Very close to cap length implies cutoff
    # - Ends without strong sentence termination
    if len(t) > 0 and not t.endswith((".", "!", "?", "…", ")", "]")):
        return True
    return False

def ask_openrouter_text(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY (set Streamlit Cloud Secrets).")

    # append user prompt to conversation
    st.session_state.messages.append({"role": "user", "content": prompt})

    # call with trimmed history
    r = _openrouter_post(_trimmed_messages(), OPENROUTER_MODEL)

    # fallback on common failures
    if r.status_code in (403, 404, 429, 500, 502, 503, 504):
        fallback = "deepseek/deepseek-v3.2"
        if OPENROUTER_MODEL != fallback:
            st.warning(f"Model '{OPENROUTER_MODEL}' failed ({r.status_code}). Falling back to {fallback}.")
            r = _openrouter_post(_trimmed_messages(), fallback)

    if r.status_code == 401:
        raise RuntimeError("OpenRouter 401: API key rejected (check Secrets).")
    if r.status_code == 402:
        raise RuntimeError("OpenRouter 402: insufficient credits/quota.")
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:600]}")

    reply = _parse_reply(r)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    return reply

def ask_openrouter_vision(prompt: str, png_images: List[bytes]) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY (set Streamlit Cloud Secrets).")

    model_name = OPENROUTER_VISION_MODEL.strip() or OPENROUTER_MODEL

    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b in png_images:
        b64 = base64.b64encode(b).decode("utf-8")
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    st.session_state.messages.append({"role": "user", "content": parts})

    r = _openrouter_post(_trimmed_messages(), model_name)

    if r.status_code >= 400:
        raise RuntimeError(
            "Vision request failed. Ensure OPENROUTER_VISION_MODEL is set to a vision-capable model. "
            f"Status {r.status_code}: {r.text[:600]}"
        )

    reply = _parse_reply(r)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    return reply

def get_full_reply_text(prompt: str) -> str:
    """Auto-continue for cutoffs."""
    full = ask_openrouter_text(prompt)
    passes = 0
    while passes < CONTINUE_PASSES and _is_likely_cutoff(full):
        passes += 1
        cont = ask_openrouter_text(
            "Continue exactly where you left off. Do NOT repeat earlier text. "
            "Finish any incomplete sections and include the conclusion."
        )
        full = full + "\n\n" + cont
        if not _is_likely_cutoff(cont):
            break
    return full

def get_full_reply_vision(prompt: str, images: List[bytes]) -> str:
    full = ask_openrouter_vision(prompt, images)
    passes = 0
    while passes < CONTINUE_PASSES and _is_likely_cutoff(full):
        passes += 1
        cont = ask_openrouter_text(
            "Continue exactly where you left off. Do NOT repeat earlier text. "
            "Finish any incomplete sections and include the conclusion."
        )
        full = full + "\n\n" + cont
        if not _is_likely_cutoff(cont):
            break
    return full

def openrouter_ping() -> (int, str):
    if not OPENROUTER_API_KEY:
        return 0, "Missing OPENROUTER_API_KEY"
    r = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": "Say OK"}],
            "temperature": 0.0,
            "max_tokens": 16,
        },
        timeout=(10, 25),
    )
    return r.status_code, r.text[:600]

# --------------------
# TTS (longer cap + timeout)
# --------------------
def clean_for_tts(text: str) -> str:
    repl = {
        "—": ", ",
        "–": "-",
        "•": "-",
        "→": " to ",
        "≥": " greater than or equal to ",
        "≤": " less than or equal to ",
        "≈": " approximately ",
    }
    out = (text or "")
    for k, v in repl.items():
        out = out.replace(k, v)
    return out.strip()

def speak_edge_tts_bytes(text: str, timeout_s: int = 25) -> bytes:
    async def _gen_audio():
        communicate = edge_tts.Communicate(
            text=text, voice=EDGE_VOICE, rate=EDGE_RATE, volume=EDGE_VOLUME
        )
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        return audio_bytes

    async def _run():
        return await asyncio.wait_for(_gen_audio(), timeout=timeout_s)

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_run())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

# --------------------
# Vosk STT
# --------------------
@st.cache_resource
def load_vosk_model():
    if not os.path.isdir(VOSK_MODEL_PATH):
        raise RuntimeError(f"Vosk model folder not found: {VOSK_MODEL_PATH}")
    for req in ["am", "conf", "graph"]:
        if not os.path.exists(os.path.join(VOSK_MODEL_PATH, req)):
            raise RuntimeError(f"Vosk model incomplete: missing '{req}' in {VOSK_MODEL_PATH}")
    return Model(VOSK_MODEL_PATH)

def wav_bytes_to_pcm16k_mono(wav_bytes: bytes, target_sr: int = 16000):
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        mono = resample_poly(mono, target_sr, sr)
        sr = target_sr
    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm16.tobytes(), sr

def stt_vosk_from_wav_bytes(wav_bytes: bytes) -> str:
    model = load_vosk_model()
    pcm_bytes, sr = wav_bytes_to_pcm16k_mono(wav_bytes)
    rec = KaldiRecognizer(model, sr)
    rec.SetWords(False)
    chunk_size = 4000
    for i in range(0, len(pcm_bytes), chunk_size):
        rec.AcceptWaveform(pcm_bytes[i:i + chunk_size])
    result = json.loads(rec.FinalResult())
    return (result.get("text") or "").strip()

# --------------------
# Session state init
# --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": PERSONA}]
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "status" not in st.session_state:
    st.session_state.status = ""
if "debug_last_step" not in st.session_state:
    st.session_state.debug_last_step = ""

# --------------------
# UI
# --------------------
st.title("📈 Fair Value Analyst｜CFA-style Valuation Chatbot")
st.caption("Type or 🎙️ speak. I’ll build a fair value range with assumptions, scenarios, sensitivities, and risks.")

with st.sidebar:
    st.subheader("Tools")

    ENABLE_TTS = st.toggle("Enable TTS", value=True)
    SHOW_DEBUG = st.toggle("Show debug info", value=False)

    if st.button("🧹 Reset conversation", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": PERSONA}]
        st.session_state.chat = []
        st.session_state.last_audio = None
        st.session_state.status = ""
        st.session_state.debug_last_step = ""
        st.rerun()

    with st.expander("Limits / Settings"):
        st.write("MAX_TOKENS:", MAX_TOKENS)
        st.write("HISTORY_TURNS:", HISTORY_TURNS)
        st.write("CONTINUE_PASSES:", CONTINUE_PASSES)
        st.write("EXTRACT_MAX_CHARS:", EXTRACT_MAX_CHARS)
        st.write("TTS_MAX_CHARS:", TTS_MAX_CHARS)
        st.write("OpenRouter timeout:", OPENROUTER_TIMEOUT_READ)

    with st.expander("Diagnostics (optional)"):
        st.write("OpenRouter key loaded:", bool(OPENROUTER_API_KEY))
        st.write("Text model:", OPENROUTER_MODEL)
        st.write("Vision model:", OPENROUTER_VISION_MODEL or "(not set)")
        st.write("Edge voice:", EDGE_VOICE)
        st.write("Vosk path:", VOSK_MODEL_PATH)
        st.write("Vosk exists:", os.path.isdir(VOSK_MODEL_PATH))
        if st.button("Test OpenRouter (text)"):
            code, body = openrouter_ping()
            st.write("Status:", code)
            st.code(body)

# Chat history
for m in st.session_state.chat:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

# Status + debug + audio
if st.session_state.status:
    st.info(st.session_state.status)

if SHOW_DEBUG and st.session_state.debug_last_step:
    st.caption(f"Debug: {st.session_state.debug_last_step}")

if st.session_state.last_audio:
    st.audio(st.session_state.last_audio, format="audio/mpeg", autoplay=True)

# Voice input
st.markdown("### 🎙️ Voice input (press to record, press again to stop)")
mic = mic_recorder(
    start_prompt="🎙️ Start recording",
    stop_prompt="⏹️ Stop",
    just_once=True,
    use_container_width=True,
    format="wav",
)

def _prompt_from_extracted_text(intake: str, user_text: str, extracted: str) -> str:
    return f"""
{intake}

You are a CFA-style valuation analyst.

User message:
{user_text}

Use the extracted source text below. Summarize investable points and provide fair value assessment.

Output format:
1) Source summary
2) Asset / thesis framing
3) Valuation approach
4) Key assumptions (ranges)
5) Base/Bull/Bear intrinsic value + probability-weighted fair value
6) Sensitivities (2 drivers)
7) Risks + monitoring checklist
8) Conclusion

Rules:
- Use only numbers explicitly present in the extracted text (do not guess).
- If key numbers are missing, use placeholders and label assumptions.

[BEGIN EXTRACTED TEXT]
{extracted}
[END EXTRACTED TEXT]
"""

def _prompt_from_pdf_images(intake: str, user_text: str) -> str:
    return f"""
{intake}

You are a CFA-style valuation analyst.

User message:
{user_text}

This is an image-heavy PDF earnings/IR slide deck.
Extract key financial figures ONLY if visible (do not guess), then perform a fair value assessment.

Output:
1) Slide-extracted figures (include page numbers)
2) Key takeaways
3) Valuation approach
4) Assumptions (ranges)
5) Base/Bull/Bear intrinsic value + probability-weighted fair value
6) Sensitivities
7) Risks + monitoring checklist
8) Conclusion
"""

def _prompt_no_url(intake: str, user_text: str) -> str:
    return f"""
{intake}

You are a CFA-style valuation analyst.

User message:
{user_text}

Task: Provide an intrinsic value (fair value) framework and estimate with Base/Bull/Bear scenarios.
Do not fabricate financial statement numbers. Use ranges and clearly label assumptions.

Output format:
1) Asset summary
2) Key value drivers
3) Valuation methods
4) Assumptions (ranges)
5) Base/Bull/Bear intrinsic value + probability-weighted fair value
6) Sensitivities
7) Risks + monitoring checklist
8) Next data needed
"""

def handle_user_message(text: str):
    st.session_state.chat.append({"role": "user", "content": text})
    st.session_state.status = "Analyzing…"
    st.session_state.last_audio = None
    st.session_state.debug_last_step = "start"

    try:
        urls = extract_urls(text)
        intake = build_intake_preamble(text)

        if urls:
            st.session_state.status = "Reading link content…"
            st.session_state.debug_last_step = "fetch_link"
            result = fetch_and_extract(urls[0], max_chars=EXTRACT_MAX_CHARS)

            if result["type"] in ("html", "pdf_text"):
                st.session_state.status = "Building valuation model…"
                st.session_state.debug_last_step = "model_text"
                prompt = _prompt_from_extracted_text(intake, text, result["text"])
                reply = get_full_reply_text(prompt)

            else:  # pdf_images
                st.session_state.status = "Reading PDF slides…"
                st.session_state.debug_last_step = "model_vision"
                prompt = _prompt_from_pdf_images(intake, text)
                reply = get_full_reply_vision(prompt, result["pdf_images"])
        else:
            st.session_state.status = "Building valuation model…"
            st.session_state.debug_last_step = "model_text"
            prompt = _prompt_no_url(intake, text)
            reply = get_full_reply_text(prompt)

        # Show reply first
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.session_state.status = ""

        # Optional TTS (longer cap)
        if ENABLE_TTS:
            try:
                st.session_state.status = "Generating audio…"
                st.session_state.debug_last_step = "tts"
                tts_text = clean_for_tts(reply[:TTS_MAX_CHARS])
                audio = speak_edge_tts_bytes(tts_text, timeout_s=TTS_TIMEOUT_SECONDS)
                st.session_state.last_audio = audio
            except Exception as e:
                st.warning(f"TTS failed (text reply shown): {e}")
            finally:
                st.session_state.status = ""

        st.session_state.debug_last_step = "done"

    except Exception as e:
        st.session_state.status = ""
        st.session_state.debug_last_step = f"error: {type(e).__name__}"
        st.error(f"Error: {e}")

# Mic processing
if mic and mic.get("bytes"):
    st.session_state.status = "Transcribing audio…"
    st.session_state.debug_last_step = "stt"
    try:
        spoken_text = stt_vosk_from_wav_bytes(mic["bytes"])
        st.session_state.status = ""
        if spoken_text:
            st.info(f"🗣️ You said: {spoken_text}")
            handle_user_message(spoken_text)
            st.rerun()
        else:
            st.warning("I couldn't catch that. Try again?")
    except Exception as e:
        st.session_state.status = ""
        st.error(f"Transcription failed: {e}")

# Text input
user_text = st.chat_input("Ask for a valuation (e.g., 'Value AAPL with a 3-scenario DCF') or paste a link…")
if user_text:
    handle_user_message(user_text)
    st.rerun()

# First greeting
if len(st.session_state.chat) == 0:
    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": (
                "Hi — I’m your CFA-style valuation analyst.\n\n"
                "Tell me the asset (ticker/ISIN), asset type (stock/bond/ETF/crypto), currency, and any key inputs you have.\n"
                "You can paste a link (HTML or PDF). For image-heavy PDF slide decks, I can render pages and read the slides.\n\n"
                "Example:\n"
                "👉 “Value AAPL using a 3-scenario DCF and show sensitivities.”"
            ),
        }
    )
    st.rerun()