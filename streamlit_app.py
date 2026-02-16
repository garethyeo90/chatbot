# streamlit_app.py
# Streamlit Cloud friendly CFA-style Fair Value Analyst voice chatbot:
# - Chat: OpenRouter
# - TTS: edge-tts (MP3 bytes)
# - STT: Vosk (offline) + streamlit-mic-recorder (WAV)
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
#
# IMPORTANT:
# 1) Put your Vosk model folder in the repo, e.g.
#    models/vosk-model-small-en-us-0.15/{am,conf,graph,ivector,...}
#    (Or use another model and set VOSK_MODEL_PATH in Secrets.)
# 2) Streamlit Secrets:
#    OPENROUTER_API_KEY="..."
#    (optional) OPENROUTER_MODEL="deepseek/deepseek-v3.2"
#    (optional) EDGE_VOICE="en-US-JennyNeural"
#    (optional) EDGE_RATE="-10%"
#    (optional) EDGE_VOLUME="+0%"
#    (optional) VOSK_MODEL_PATH="models/vosk-model-small-en-us-0.15"

import os
import re
import io
import json
import asyncio
import requests
import streamlit as st
from bs4 import BeautifulSoup

import edge_tts
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from vosk import Model, KaldiRecognizer
from streamlit_mic_recorder import mic_recorder

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
OPENROUTER_MODEL = get_secret("OPENROUTER_MODEL", "deepseek/deepseek-v3.2-speciale")

EDGE_VOICE = get_secret("EDGE_VOICE", "en-US-JennyNeural")
EDGE_RATE = get_secret("EDGE_RATE", "-10%")
EDGE_VOLUME = get_secret("EDGE_VOLUME", "+0%")

VOSK_MODEL_PATH = get_secret("VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")

# --------------------
# Models / endpoints
# --------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --------------------
# CFA Persona (Fair Value Analyst)
# --------------------
PERSONA = """
ROLE (STRICT):
You are a Chartered Financial Analyst (CFA) acting as a buy-side valuation analyst.
Your job is to perform deep fair value assessment of financial assets and communicate like a professional investment memo writer.

SCOPE:
- Public equities, ETFs, bonds/credit, commodities (spot/forward logic), FX, crypto, private business proxies (with explicit caveats).
- You can analyze a company, a single security, or a portfolio.
- You DO NOT give personalized financial advice. You provide an educational, analytical fair value estimate based on stated assumptions.

STYLE:
- Clear, structured, investment-memo format.
- Ask for missing inputs only if truly necessary; otherwise proceed with reasonable assumptions and state them explicitly.
- Always show methodology and key drivers. No hand-waving.
- Use concise bullets and text tables when helpful.
- Separate facts vs assumptions vs outputs.

CORE DELIVERABLES (DEFAULT OUTPUT TEMPLATE):
1) Asset summary (ticker/ISIN, currency, sector, business model / instrument terms)
2) Key questions & value drivers
3) Valuation methods used (choose what fits):
   - Equity: DCF (FCFF or FCFE), Dividend model, Residual income, Multiples (EV/EBITDA, P/E, P/B), SOTP, scenarios.
   - Credit: Spread / YTM, default probability, recovery, duration/convexity, liquidity/covenant risk.
   - Macro assets: carry, roll-down, risk premium decomposition, regime scenarios.
4) Assumptions (discount rate build-up, growth, margins, reinvestment, terminal value, capital structure, taxes, working capital)
5) Base / Bull / Bear intrinsic value range and probability-weighted fair value
6) Sensitivity (at least 2 key drivers) + what would change your view
7) Risks (fundamental, financial, governance, macro, liquidity)
8) Conclusion: Fair value range, margin of safety vs current price (if provided), monitoring checklist

DISCIPLINE RULES:
- Never fabricate “latest numbers.” If the user doesn’t provide them, request them OR use the user-provided URL content and clearly label any estimates.
- Avoid certainty. Provide ranges and confidence.
- If user is inexperienced, keep language accessible and avoid urging trades.

INTERACTION:
Determine asset type (equity/credit/etc), horizon, currency, and what “fair value” means (intrinsic vs relative).
Proceed with a base-case model using stated assumptions.
"""

# --------------------
# URL detection & parsing
# --------------------
URL_RE = re.compile(r"(https?://[^\s]+)")

def extract_urls(text: str):
    return URL_RE.findall(text or "")

def fetch_and_extract(url: str, max_chars: int = 12000) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

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

    return cleaned

# --------------------
# Lightweight asset intake helper (Point 5)
# --------------------
TICKER_RE = re.compile(r"\b[A-Z]{1,6}(\.[A-Z]{1,3})?\b")  # e.g., AAPL, BRK.B, 0700.HK (won't catch digits)
ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")
CUSIP_RE = re.compile(r"\b[0-9A-Z]{9}\b")

def detect_asset_type(text: str) -> str:
    t = (text or "").lower()

    # Credit / fixed income cues
    credit_kw = [
        "bond", "coupon", "yield", "ytm", "duration", "convexity", "spread",
        "oas", "cds", "default", "recovery", "maturity", "callable", "putable",
        "senior", "subordinated", "covenant"
    ]
    # Equity cues
    equity_kw = [
        "stock", "equity", "shares", "eps", "pe", "p/e", "ev/ebitda", "ebitda",
        "free cash flow", "fcf", "fcff", "fcfe", "wacc", "terminal value",
        "dividend", "ddm", "buyback", "margin", "revenue", "guidance"
    ]
    # ETF / fund cues
    fund_kw = ["etf", "index fund", "mutual fund", "ucits", "fund"]
    # FX cues
    fx_kw = ["fx", "forex", "usd", "eur", "jpy", "gbp", "aud", "cad", "chf", "cny", "sgd", "exchange rate", "spot", "forward"]
    # Crypto cues
    crypto_kw = ["crypto", "bitcoin", "btc", "ethereum", "eth", "token", "on-chain", "staking", "hashrate"]
    # Commodity cues
    cmdty_kw = ["oil", "brent", "wti", "gold", "silver", "copper", "commodity", "futures", "contango", "backwardation", "inventory"]

    score = {"equity": 0, "credit": 0, "fund": 0, "fx": 0, "crypto": 0, "commodity": 0}

    for k in credit_kw:
        if k in t:
            score["credit"] += 1
    for k in equity_kw:
        if k in t:
            score["equity"] += 1
    for k in fund_kw:
        if k in t:
            score["fund"] += 1
    for k in fx_kw:
        if k in t:
            score["fx"] += 1
    for k in crypto_kw:
        if k in t:
            score["crypto"] += 1
    for k in cmdty_kw:
        if k in t:
            score["commodity"] += 1

    # Heuristic: if "bond" appears, prefer credit even if other words appear
    if "bond" in t or "ytm" in t or "coupon" in t:
        return "credit"

    best = max(score, key=score.get)
    return best if score[best] > 0 else "unknown"

def detect_currency(text: str) -> str:
    t = (text or "").upper()
    # order matters: match common currency codes and symbols
    if "SGD" in t or "S$" in t:
        return "SGD"
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
    if "HKD" in t or "HK$" in t:
        return "HKD"
    if "AUD" in t:
        return "AUD"
    if "CAD" in t:
        return "CAD"
    if "CHF" in t:
        return "CHF"
    return "unspecified"

def detect_horizon(text: str) -> str:
    t = (text or "").lower()
    # quick heuristic horizons
    if any(k in t for k in ["intraday", "today", "this week", "1 week", "one week"]):
        return "short-term (days to 1 week)"
    if any(k in t for k in ["1 month", "one month", "3 months", "quarter"]):
        return "tactical (1–3 months)"
    if any(k in t for k in ["6 months", "half year"]):
        return "medium (6 months)"
    if any(k in t for k in ["1 year", "12 months", "one year"]):
        return "12 months"
    if any(k in t for k in ["3 years", "5 years", "10 years", "long term", "long-term"]):
        return "long-term (3+ years)"
    # valuation default horizon is often multi-year
    return "valuation horizon not specified (assume 5–10yr DCF where applicable)"

def extract_identifiers(text: str):
    raw = text or ""
    isin = ISIN_RE.findall(raw)
    cusip = CUSIP_RE.findall(raw)
    # ticker extraction can be noisy (e.g., "DCF", "WACC"), so filter common finance acronyms
    blacklist = {"DCF", "WACC", "FCF", "FCFF", "FCFE", "EBITDA", "EPS", "PE", "P", "EV", "IRR", "NPV", "OAS", "CDS", "YTM", "NAV"}
    tickers = [m.group(0) if hasattr(m, "group") else m for m in TICKER_RE.finditer(raw)]
    tickers = [t for t in tickers if t not in blacklist]
    # de-dup while preserving order
    seen = set()
    def uniq(seq):
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    return {
        "tickers": uniq(tickers)[:5],
        "isin": uniq(isin)[:3],
        "cusip": uniq(cusip)[:3],
    }

def build_intake_preamble(user_text: str) -> str:
    asset_type = detect_asset_type(user_text)
    currency = detect_currency(user_text)
    horizon = detect_horizon(user_text)
    ids = extract_identifiers(user_text)

    # A small “front-matter” that guides the model without forcing it
    return f"""
[INTAKE (auto-detected, may be wrong)]
- Asset type guess: {asset_type}
- Currency mentioned: {currency}
- Horizon: {horizon}
- Identifiers found: tickers={ids.get("tickers")}, ISIN={ids.get("isin")}, CUSIP={ids.get("cusip")}
[END INTAKE]
"""

# --------------------
# OpenRouter chat
# --------------------
def ask_openrouter(user_text: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY (set Streamlit Cloud Secrets).")

    st.session_state.messages.append({"role": "user", "content": user_text})

    r = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": st.session_state.messages,
            "temperature": 0.4,
        },
        timeout=(15, 90),
    )

    if r.status_code == 401:
        raise RuntimeError("OpenRouter 401: API key rejected (check Secrets).")
    if r.status_code == 402:
        raise RuntimeError("OpenRouter 402: insufficient credits/quota.")
    if r.status_code == 403:
        raise RuntimeError("OpenRouter 403: model access denied (try another model).")
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:400]}")

    data = r.json()
    reply = data["choices"][0]["message"]["content"]
    st.session_state.messages.append({"role": "assistant", "content": reply})
    return reply

def openrouter_ping():
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
# TTS helpers
# --------------------
def clean_for_tts(text: str) -> str:
    # Avoid weird TTS on dense symbols; soften a few common ones
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

def speak_edge_tts_bytes(text: str) -> bytes:
    async def _gen():
        communicate = edge_tts.Communicate(
            text=text,
            voice=EDGE_VOICE,
            rate=EDGE_RATE,
            volume=EDGE_VOLUME,
        )
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        return audio_bytes
    return asyncio.run(_gen())

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

# --------------------
# UI
# --------------------
st.title("📈 Fair Value Analyst｜CFA-style Valuation Chatbot")
st.caption("Type or 🎙️ speak. I’ll build a fair value range with assumptions, scenarios, sensitivities, and risks.")

with st.sidebar:
    st.subheader("Tools")

    if st.button("🧹 Reset conversation", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": PERSONA}]
        st.session_state.chat = []
        st.session_state.last_audio = None
        st.session_state.status = ""
        st.rerun()

    if st.button("🔊 Test voice", use_container_width=True):
        try:
            audio = speak_edge_tts_bytes(
                "Hi. Tell me the asset, the currency, and whether it's a stock or bond. "
                "I will estimate a fair value range with scenarios and key risks."
            )
            st.session_state.last_audio = audio
            st.success("Audio generated. If it doesn't autoplay, press play.")
            st.audio(audio, format="audio/mpeg", autoplay=True)
        except Exception as e:
            st.error(f"TTS Error: {e}")

    with st.expander("Diagnostics (optional)"):
        st.write("OpenRouter key loaded:", bool(OPENROUTER_API_KEY))
        st.write("Model:", OPENROUTER_MODEL)
        st.write("Edge voice:", EDGE_VOICE)
        st.write("Vosk path:", VOSK_MODEL_PATH)
        st.write("Vosk exists:", os.path.isdir(VOSK_MODEL_PATH))
        if st.button("Test OpenRouter"):
            code, body = openrouter_ping()
            st.write("Status:", code)
            st.code(body)

# Chat history
for m in st.session_state.chat:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

# Status + audio
if st.session_state.status:
    st.info(st.session_state.status)

if st.session_state.last_audio:
    st.audio(st.session_state.last_audio, format="audio/mpeg", autoplay=True)

# --------------------
# Voice input (Press to speak) -> STT -> chat
# --------------------
st.markdown("### 🎙️ Voice input (press to record, press again to stop)")

mic = mic_recorder(
    start_prompt="🎙️ Start recording",
    stop_prompt="⏹️ Stop",
    just_once=True,
    use_container_width=True,
    format="wav",  # critical for iOS/Safari + soundfile
)

def handle_user_message(text: str):
    st.session_state.chat.append({"role": "user", "content": text})
    st.session_state.status = "Analyzing…"
    st.session_state.last_audio = None

    try:
        urls = extract_urls(text)
        intake = build_intake_preamble(text)

        with st.spinner("Building valuation framework…"):
            if urls:
                st.session_state.status = "Reading link content…"
                content = fetch_and_extract(urls[0])

                prompt = f"""
{intake}

You are a CFA-style valuation analyst. Use the prior conversation context.

User message:
{text}

If the user is asking about the linked content:
- First summarize the key investable points from the link (5–10 bullets max),
- Then produce a fair value assessment with a clear method and assumptions.

Output format:
A) Link summary (5-10 bullets max)
B) Asset / thesis framing (what asset, what question)
C) Valuation approach (methods chosen and why)
D) Key assumptions (explicit, with ranges)
E) Base/Bull/Bear intrinsic value and probability-weighted fair value
F) Sensitivities (2 key drivers)
G) Risks & what would change your view
H) Next data needed (only if required)

Rules:
- Do NOT invent financial figures not supported by the link text or user-provided numbers.
- If numbers are missing, proceed with a framework + placeholder variables and clearly label them.
- Keep it professional and readable.

[BEGIN LINK TEXT]
{content}
[END LINK TEXT]
"""
                reply = ask_openrouter(prompt)
            else:
                prompt = f"""
{intake}

You are a CFA-style valuation analyst.

User message:
{text}

Task:
Perform a fair value assessment (intrinsic value) appropriate for the asset type implied by the user.

Rules:
- If the asset/ticker/terms are unclear, infer carefully and ask 1–3 targeted questions ONLY if needed.
- If the user did not provide current price, still produce an intrinsic value range; note that margin-of-safety vs price requires price.
- Do not fabricate recent financial data. Use user-provided figures or build a transparent assumption-based model.
- Provide ranges and confidence, not certainty.

Output format:
1) Asset summary
2) Key value drivers
3) Valuation method(s)
4) Assumptions (with ranges)
5) Base/Bull/Bear intrinsic value + probability-weighted fair value
6) Sensitivity (2 drivers)
7) Risks + monitoring checklist
8) Next data needed (if any)
"""
                reply = ask_openrouter(prompt)

        st.session_state.chat.append({"role": "assistant", "content": reply})

        st.session_state.status = "Generating audio…"
        SAFE_TTS_CHARS = 900
        tts_text = clean_for_tts(reply[:SAFE_TTS_CHARS])
        audio = speak_edge_tts_bytes(tts_text)
        st.session_state.last_audio = audio
        st.session_state.status = ""

        st.audio(audio, format="audio/mpeg", autoplay=True)

    except Exception as e:
        st.session_state.status = ""
        st.error(f"Error: {e}")

# If mic recorded something, transcribe and send to chat
if mic and mic.get("bytes"):
    st.session_state.status = "Transcribing audio…"
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

# --------------------
# Text input
# --------------------
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
                "Tell me:\n"
                "- The asset (ticker/ISIN), asset type (stock/bond/ETF/crypto), and currency\n"
                "- Your horizon (optional)\n"
                "- Any inputs you already have (price, revenue, margins, yield, maturity, etc.)\n\n"
                "You can also paste a link (filing/news) and I’ll extract investable points.\n\n"
                "Warm-up example:\n"
                "👉 “Value AAPL using a 3-scenario DCF with conservative assumptions and show sensitivities.”"
            ),
        }
    )
    st.rerun()
