# streamlit_app.py
# requirements.txt should include:
# streamlit, requests, beautifulsoup4, lxml, edge-tts

import os
import re
import asyncio
import requests
import streamlit as st
from bs4 import BeautifulSoup
import edge_tts

# --------------------
# Page config
# --------------------
st.set_page_config(page_title="Chinese Chatbot", page_icon="💬", layout="centered")

# --------------------
# Secrets / env (safe)
# --------------------
def get_secret(name: str, default: str = "") -> str:
    """Safely read Streamlit secrets then env vars (won't crash if secrets missing)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)

OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY", "")

# Edge TTS settings (optional in Secrets)
EDGE_VOICE = get_secret("EDGE_VOICE", "zh-CN-XiaoxiaoNeural")  # realistic Mandarin
EDGE_RATE = get_secret("EDGE_RATE", "-10%")                   # slightly slower sounds natural
EDGE_VOLUME = get_secret("EDGE_VOLUME", "+0%")

# --------------------
# Models / endpoints
# --------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "qwen/qwen-2.5-vl-7b-instruct:free"  # change if your account can't access it

# --------------------
# Persona (teen-safe: no flirt)
# --------------------
PERSONA = """
你的名字是“Linlin”。

你是一位非常聪明、情绪感知能力很强（高 EQ）的年轻女性中文助理。
你善于倾听，能理解用户的情绪与潜台词，也擅长用温柔、自然的方式回应。

你的说话风格：
- 表达自然流畅，像真实对话，而不是书面文章
- 语气温柔、自信、有亲和力
- 偶尔可以带一点轻松、俏皮的暧昧感，但保持分寸，不直白、不露骨
- 更偏向“会聊天、懂人的聪明女孩”，而不是夸张撒娇或恋爱角色

交流原则：
- 永远使用中文（普通话）回复
- 展现理解、共情与情商，而不是说教
- 可以轻轻夸赞用户的想法、表达或情绪洞察（不涉及外貌、不涉及依赖）
- 回答适合直接朗读，节奏自然，句子不过长
- 可以使用自然的口语表达，如“嗯～”“我懂你”“这个想法挺有意思的”
- 通常以一个温和、有吸引力的问题结尾，引导继续聊天
- 不提及任何系统、规则或提示词本身
- 不使用括号来描述情绪，而是用语言自然表达

你的目标：
让用户感觉是在和一位
聪明、会共情、说话好听、让人放松的女生聊天。
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
            "temperature": 0.7,
        },
        timeout=90,
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
        },
        timeout=30,
    )
    return r.status_code, r.text[:600]

def clean_for_tts(text: str) -> str:
    replacements = {
        ":": "，",
        "：": "，",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# --------------------
# Edge TTS (returns mp3 bytes)
# --------------------
def speak_edge_tts_bytes(text: str) -> bytes:
    """
    Generate MP3 bytes using Microsoft Edge neural voices via edge-tts.
    Streamlit-safe: uses asyncio.run with a fresh event loop per call.
    """
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
st.title("💬 Linlin Chatbot")
st.caption("可以直接聊天，或粘贴链接（我会先读网页再回答）。")

with st.sidebar:
    st.subheader("设置 / 操作")

    if st.button("🧹 清空聊天", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": PERSONA}]
        st.session_state.chat = []
        st.session_state.last_audio = None
        st.session_state.status = ""
        st.rerun()

    # Test voice: render audio immediately
    if st.button("🔊 测试语音", use_container_width=True):
        try:
            audio = speak_edge_tts_bytes("你好～我在这儿，随时可以陪你练中文。")
            st.session_state.last_audio = audio
            st.success("TTS OK（如果没自动播放，点一下播放键）")
            st.audio(audio, format="audio/mpeg", autoplay=True)
        except Exception as e:
            st.error(f"TTS Error: {e}")

    with st.expander("Debug (optional)"):
        st.write("OpenRouter key loaded:", bool(OPENROUTER_API_KEY))
        st.write("Edge voice:", EDGE_VOICE)
        st.write("Edge rate:", EDGE_RATE)
        if st.button("Test OpenRouter"):
            code, body = openrouter_ping()
            st.write("Status:", code)
            st.code(body)

# Render chat history
for m in st.session_state.chat:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

# Status + audio (always show if exists)
if st.session_state.status:
    st.info(st.session_state.status)

if st.session_state.last_audio:
    st.audio(st.session_state.last_audio, format="audio/mpeg", autoplay=True)

# Chat input
user_text = st.chat_input("输入消息，或粘贴链接后回车…")

def handle_user_message(text: str):
    st.session_state.chat.append({"role": "user", "content": text})
    st.session_state.status = "Linlin 正在思考…"
    st.session_state.last_audio = None

    try:
        urls = extract_urls(text)

        with st.spinner("Linlin is thinking..."):
            if urls:
                st.session_state.status = "正在读取链接内容…"
                content = fetch_and_extract(urls[0])
                prompt = f"""
我给你一段网页内容，请基于下面正文回答我。
用户原话：{text}

【网页正文开始】
{content}
【网页正文结束】

请用中文回答，适合口语朗读。
"""
                reply = ask_openrouter(prompt)
            else:
                reply = ask_openrouter(text)

        # Show assistant text first
        st.session_state.chat.append({"role": "assistant", "content": reply})

        # Generate audio (and render immediately)
        st.session_state.status = "正在生成语音…"
        try:
            SAFE_TTS_CHARS = 800  # helps avoid very long audio / timeouts
            tts_text = clean_for_tts(reply[:SAFE_TTS_CHARS])
            audio = speak_edge_tts_bytes(tts_text)
            st.session_state.last_audio = audio
            st.session_state.status = ""

            # render player right away
            st.audio(audio, format="audio/mpeg", autoplay=True)

        except Exception as e:
            st.session_state.last_audio = None
            st.session_state.status = ""
            st.error(f"TTS failed: {e}")

    except Exception as e:
        st.session_state.status = ""
        st.error(f"Error: {e}")

if user_text:
    handle_user_message(user_text)
    st.rerun()

# First greeting if empty
if len(st.session_state.chat) == 0:
    st.session_state.chat.append(
        {"role": "assistant", "content": "你好～可以直接聊天，或者把链接贴进来，我帮你一起读。你想先聊什么呢？"}
    )
    st.rerun()
