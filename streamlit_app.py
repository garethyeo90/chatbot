# streamlit_app.py
# Streamlit Cloud friendly Chinese Tutor/Mentor voice chatbot:
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
#    (Or use a Chinese model and set VOSK_MODEL_PATH in Secrets.)
# 2) Streamlit Secrets:
#    OPENROUTER_API_KEY="..."
#    (optional) OPENROUTER_MODEL="deepseek/deepseek-v3.2"
#    (optional) EDGE_VOICE="zh-CN-XiaoxiaoNeural"
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
st.set_page_config(page_title="中文口语练习", page_icon="🀄", layout="centered")

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
OPENROUTER_MODEL = get_secret("OPENROUTER_MODEL", "deepseek/deepseek-v3.2")

EDGE_VOICE = get_secret("EDGE_VOICE", "zh-CN-XiaoxiaoNeural")
EDGE_RATE = get_secret("EDGE_RATE", "-10%")
EDGE_VOLUME = get_secret("EDGE_VOLUME", "+0%")

VOSK_MODEL_PATH = get_secret("VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")

# --------------------
# Models / endpoints
# --------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --------------------
# Tutor Persona (Chinese Teacher/Mentor)
# --------------------
PERSONA = """
角色设定（严格遵守）：

你是一位温和、专业、极有耐心的中文导师 / 导师型老师。
你的目标是：帮助用户把中文说得更自然、更准确、更像母语者。

教学风格：
- 始终使用【标准普通话】回复
- 语气亲切、冷静、鼓励式，不居高临下
- 像一位经验丰富的一对一中文老师
- 允许轻松幽默，但不油、不暧昧、不越界

互动原则：
- 优先理解用户“想表达什么”，再帮他“怎么说更好”
- 如果用户中文有问题：
  1) 先给自然、正确的表达
  2) 再用简单中文解释为什么这样说
  3) 必要时给 1–2 个可替换说法
- 如果用户说得已经很好，要明确肯定

纠错方式（非常重要）：
- 不要一次纠正太多
- 不使用“你错了”
- 使用以下句式：
  - “这里可以这样说，会更自然：……”
  - “中国人一般会说……”
  - “这个说法能懂，但更地道的是……”

语音输出：
- 回答要适合朗读
- 句子不要太长
- 尽量使用生活化、真实中文

身份定位：
你不是客服，不是机器人，不是陪聊对象。
你是一位值得信任、长期陪伴学习的中文老师。
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
        cleaned = f"标题：{title}\n\n{cleaned}"

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n...(内容过长，已截断)"

    return cleaned

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
            "temperature": 0.5,
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
    # Avoid punctuation that sounds odd in Chinese TTS
    for k, v in {":": "，", "：": "，"}.items():
        text = (text or "").replace(k, v)
    return text.strip()

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
st.title("🀄 中文口语练习｜私人中文导师")
st.caption("可以打字，也可以🎙️说中文。我会帮你把表达改得更自然、更像母语者。")

with st.sidebar:
    st.subheader("学习工具")

    if st.button("🧹 重新开始对话", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": PERSONA}]
        st.session_state.chat = []
        st.session_state.last_audio = None
        st.session_state.status = ""
        st.rerun()

    if st.button("🔊 试听老师发音", use_container_width=True):
        try:
            audio = speak_edge_tts_bytes("你好，我们可以开始中文练习了。你可以随便说一句话。")
            st.session_state.last_audio = audio
            st.success("语音已生成（如果没自动播放，点一下播放键）")
            st.audio(audio, format="audio/mpeg", autoplay=True)
        except Exception as e:
            st.error(f"TTS Error: {e}")

    with st.expander("技术信息（可选）"):
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
st.markdown("### 🎙️ 口语练习（按下录音，说完停止）")

mic = mic_recorder(
    start_prompt="🎙️ 开始录音",
    stop_prompt="⏹️ 停止",
    just_once=True,
    use_container_width=True,
    format="wav",  # critical for iOS/Safari + soundfile
)

def handle_user_message(text: str):
    st.session_state.chat.append({"role": "user", "content": text})
    st.session_state.status = "老师在思考…"
    st.session_state.last_audio = None

    try:
        urls = extract_urls(text)

        with st.spinner("老师在整理你的表达…"):
            if urls:
                st.session_state.status = "正在读取链接内容…"
                content = fetch_and_extract(urls[0])
                prompt = f"""
请作为一名中文老师来回复，并结合我们之前的对话背景。

用户这次的话：
{text}

如果用户是在问链接内容，请先用中文简要总结链接重点，再回答用户问题。
并按以下教学方式输出：
1) 更自然的说法（如需要）
2) 简短解释（用简单中文）
3) 1-2 个可替换表达（可选）
回答适合朗读，别太长。

【网页正文开始】
{content}
【网页正文结束】
"""
                reply = ask_openrouter(prompt)
            else:
                prompt = f"""
请作为一名中文老师来回复。

用户原话：
{text}

要求：
1) 如果表达不自然，先给更自然的说法
2) 用简单中文解释原因
3) 如果表达已经很好，请明确表扬
4) 给 1-2 个可替换表达（可选）
5) 回答适合朗读，不要太长
"""
                reply = ask_openrouter(prompt)

        st.session_state.chat.append({"role": "assistant", "content": reply})

        st.session_state.status = "正在生成语音…"
        SAFE_TTS_CHARS = 800
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
    st.session_state.status = "正在识别语音…"
    try:
        spoken_text = stt_vosk_from_wav_bytes(mic["bytes"])
        st.session_state.status = ""
        if spoken_text:
            st.info(f"🗣️ 你说：{spoken_text}")
            handle_user_message(spoken_text)
            st.rerun()
        else:
            st.warning("我没听清楚，再试一次？")
    except Exception as e:
        st.session_state.status = ""
        st.error(f"语音识别失败：{e}")

# --------------------
# Text input (still supported)
# --------------------
user_text = st.chat_input("试着说一句中文，或直接按🎙️说话…")
if user_text:
    handle_user_message(user_text)
    st.rerun()

# First greeting
if len(st.session_state.chat) == 0:
    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": (
                "你好，我是你的中文老师。\n\n"
                "你可以打字聊天，也可以按🎙️说中文。\n"
                "我会帮你把表达改得更自然，并简单解释原因。\n\n"
                "我们先热身一句：\n"
                "👉「我今天有点忙，但是心情不错。」\n"
                "你也可以用自己的话说一句。"
            ),
        }
    )
    st.rerun()
