# tts_engine.py — Expressive TTS pipeline
#
# Flow:  text → emotion detection → Kokoro TTS → RVC (your voice) → WAV bytes
#
# pip install kokoro soundfile transformers rvc-python

import os
import re
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE THESE — rename to match your actual filenames
# ─────────────────────────────────────────────────────────────────────────────

VOICE_PTH   = "voices/your_voice.pth"      # rename to your .pth filename
VOICE_INDEX = "voices/your_voice.index"    # rename to your .index filename

# Overall pitch shift in semitones. 0 = no change.
# If the final voice sounds too high, try -2. Too low, try +2.
BASE_PITCH = 0

# How much to follow the RVC voice model (0.0 to 1.0)
# 0.75 is the sweet spot. Lower = more stable. Higher = more like the model.
INDEX_RATE = 0.75

# ── Emotion profiles ──────────────────────────────────────────────────────────
# voice:  which Kokoro voice variant to use as base
#   af_sky    — light, airy, youthful       <- used for energetic emotions
#   af_bella  — warm, expressive, natural   <- used for neutral/happy
#   af_sarah  — calm, measured, softer      <- used for sad/scared
# pitch:  semitones added on top of BASE_PITCH for this emotion
# speed:  1.0 = normal. 1.08 = 8% faster. 0.92 = 8% slower.
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
# Singletons — loaded once on first use so startup stays fast
# ─────────────────────────────────────────────────────────────────────────────

_kokoro     = None
_classifier = None
_rvc        = None


def _get_kokoro():
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    try:
        from kokoro import KPipeline
    except ImportError:
        raise RuntimeError("Run: pip install kokoro soundfile")
    print("[TTS] Loading Kokoro...")
    _kokoro = KPipeline(lang_code="a")  # 'a' = American English
    print("[TTS] Kokoro ready")
    return _kokoro


def _get_classifier():
    """
    Loads j-hartmann/emotion-english-distilroberta-base (~330MB, downloads once).
    Falls back to simple keyword rules if it can't load.
    """
    global _classifier
    if _classifier is not None:
        return _classifier
    try:
        from transformers import pipeline as hf_pipeline
        print("[TTS] Loading emotion classifier...")
        _classifier = hf_pipeline(
            "text-classification",
            model  = "j-hartmann/emotion-english-distilroberta-base",
            device = 0,     # GPU. Change to -1 if you get CUDA errors.
            top_k  = 1
        )
        print("[TTS] Emotion classifier ready")
    except Exception as e:
        print(f"[TTS] Emotion classifier failed ({e}) - using keyword fallback")
        _classifier = "fallback"
    return _classifier


def _get_rvc():
    global _rvc
    if _rvc is not None:
        return _rvc
    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        raise RuntimeError("Run: pip install rvc-python")

    pth   = Path(VOICE_PTH)
    index = Path(VOICE_INDEX)

    if not pth.exists():
        raise FileNotFoundError(
            f"\n[TTS] Voice model not found: {pth}\n"
            f"      Make sure your .pth file is in the voices/ folder\n"
            f"      and that VOICE_PTH in tts_engine.py matches the filename."
        )

    print(f"[TTS] Loading RVC voice: {pth.name} ...")
    _rvc = RVCInference(device="cuda:0")  # uses your 3060
    _rvc.load_model(str(pth), str(index) if index.exists() else "")
    print("[TTS] RVC voice loaded")
    return _rvc


# ─────────────────────────────────────────────────────────────────────────────
# Emotion detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_emotion(text: str) -> str:
    """Returns one of: joy, surprise, sadness, anger, fear, disgust, neutral"""
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
    """Simple fallback - reads keywords and punctuation."""
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
# Text cleaning - strips things that sound weird when spoken
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
# Main function - called from main.py
# ─────────────────────────────────────────────────────────────────────────────

def synthesize(text: str) -> bytes:
    """
    Full pipeline: text -> emotion -> Kokoro TTS -> RVC -> WAV bytes.
    Returns empty bytes on failure so the chat keeps working.
    """
    text = _clean(text)
    if not text:
        return b""

    emotion = detect_emotion(text)
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])

    print(f"[TTS] '{emotion}' -> voice={profile['voice']} "
          f"pitch={BASE_PITCH + profile['pitch']:+d} speed={profile['speed']}")

    tmp_kokoro = None
    tmp_rvc    = None

    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        raise RuntimeError("Run: pip install soundfile numpy")

    try:
        # Step 1: Kokoro -> temp WAV
        kokoro = _get_kokoro()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_kokoro = f.name

        chunks = []
        for _, _, audio in kokoro(text, voice=profile["voice"], speed=profile["speed"]):
            if audio is not None:
                chunks.append(audio)

        if not chunks:
            print("[TTS] Kokoro produced no audio.")
            return b""

        sf.write(tmp_kokoro, np.concatenate(chunks), KOKORO_SAMPLE_RATE)

        # Step 2: RVC -> final WAV in your downloaded voice
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_rvc = f.name

        rvc = _get_rvc()
        rvc.infer_file(
            input_path  = tmp_kokoro,
            output_path = tmp_rvc,
            f0_up_key   = BASE_PITCH + profile["pitch"],
            index_rate  = INDEX_RATE,
            protect     = 0.33,
        )

        with open(tmp_rvc, "rb") as f:
            return f.read()

    except Exception as e:
        print(f"[TTS] Error: {e}")
        return b""

    finally:
        for p in (tmp_kokoro, tmp_rvc):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
