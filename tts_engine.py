# tts_engine.py — Kokoro TTS with emotion detection (no RVC)
#
# pip install kokoro soundfile transformers

import io
import re
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_PROFILES = {
    "joy":      {"voice": "af_bella", "pitch": +2, "speed": 1.07},
    "surprise": {"voice": "af_sky",   "pitch": +3, "speed": 1.08},
    "sadness":  {"voice": "af_sarah", "pitch": -2, "speed": 0.92},
    "anger":    {"voice": "af_bella", "pitch": +1, "speed": 1.05},
    "fear":     {"voice": "af_sky",   "pitch": +2, "speed": 1.09},
    "disgust":  {"voice": "af_sarah", "pitch": -1, "speed": 0.96},
    "neutral":  {"voice": "af_sky",   "pitch":  0, "speed": 1.00},
}

KOKORO_SAMPLE_RATE = 24000

# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

_kokoro     = None
_classifier = None


def _get_kokoro():
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    try:
        from kokoro import KPipeline
    except ImportError:
        raise RuntimeError("Run: pip install kokoro soundfile")
    print("[TTS] Loading Kokoro...")
    _kokoro = KPipeline(lang_code="a")
    print("[TTS] Kokoro ready")
    return _kokoro


def _get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    try:
        from transformers import pipeline as hf_pipeline
        print("[TTS] Loading emotion classifier...")
        _classifier = hf_pipeline(
            "text-classification",
            model  = "j-hartmann/emotion-english-distilroberta-base",
            device = 0,
            top_k  = 1
        )
        print("[TTS] Emotion classifier ready")
    except Exception as e:
        print(f"[TTS] Emotion classifier unavailable ({e}) - using keyword fallback")
        _classifier = "fallback"
    return _classifier


# ─────────────────────────────────────────────────────────────────────────────
# Emotion detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_emotion(text: str) -> str:
    clf = _get_classifier()
    if clf != "fallback":
        try:
            result = clf(text[:512])[0]
            label  = result[0]["label"].lower() if isinstance(result, list) else result["label"].lower()
            if label in EMOTION_PROFILES:
                return label
        except Exception:
            pass
    return _keyword_emotion(text)


def _keyword_emotion(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["wow", "omg", "no way", "seriously", "whoa"]) or text.count("!") >= 2:
        return "surprise"
    if any(w in t for w in ["happy", "love", "great", "awesome", "yay", "excited", "fun", "wonderful"]):
        return "joy"
    if any(w in t for w in ["sorry", "sad", "unfortunately", "miss", "hard", "difficult", "wish"]):
        return "sadness"
    if any(w in t for w in ["angry", "furious", "annoying", "frustrated", "stop", "worst"]):
        return "anger"
    if any(w in t for w in ["scared", "afraid", "nervous", "worried", "anxious"]):
        return "fear"
    return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"\[Source:[^\]]*\]", "", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"[*_#]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > 100:
        text = " ".join(words[:100]) + "."
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def synthesize(text: str) -> bytes:
    text = _clean(text)
    if not text:
        return b""

    emotion = detect_emotion(text)
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])
    print(f"[TTS] '{emotion}' -> voice={profile['voice']} speed={profile['speed']}")

    try:
        import numpy as np
        import soundfile as sf

        kokoro = _get_kokoro()
        chunks = []

        for result in kokoro(text, voice=profile["voice"], speed=profile["speed"]):
            # Handle both old tuple format and new Choice object format
            if isinstance(result, tuple):
                audio = result[2]
            elif hasattr(result, 'audio'):
                audio = result.audio
            else:
                audio = result
            if audio is not None:
                chunks.append(audio)

        if not chunks:
            print("[TTS] No audio generated.")
            return b""

        full_audio = np.concatenate(chunks)

        # Write to in-memory buffer instead of temp file
        buf = io.BytesIO()
        sf.write(buf, full_audio, KOKORO_SAMPLE_RATE, format="WAV")
        buf.seek(0)
        return buf.read()

    except Exception as e:
        print(f"[TTS] Error: {e}")
        return b""
