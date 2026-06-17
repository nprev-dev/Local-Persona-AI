# tts_engine.py — Kokoro TTS with emotion detection (no RVC)
#
# pip install kokoro soundfile transformers

import io
import re
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE
# ─────────────────────────────────────────────────────────────────────────────

# Path to your reference audio clip (5-30 seconds of clean voice for good result)
REFERENCE_CLIP = "aemeath_ref.wav" # Modify to sample of your voice
 
# Emotion exaggeration level (0.0 to 1.0)
# 0.0 = flat/monotone
# 0.5 = natural conversational
# 0.7 = noticeably expressive  ← recommended
# 1.0 = very dramatic
BASE_EXAGGERATION = 0.6
 
# CFG weight — how closely to follow the reference voice (0.0 to 1.0)
# Higher = more like the reference, lower = more natural variation
CFG_WEIGHT = 0.7

EMOTION_PROFILES = {
    "joy":      {"exaggeration": 0.75, "cfg": 0.5},
    "surprise": {"exaggeration": 0.80, "cfg": 0.4},
    "sadness":  {"exaggeration": 0.45, "cfg": 0.6},
    "anger":    {"exaggeration": 0.70, "cfg": 0.5},
    "fear":     {"exaggeration": 0.65, "cfg": 0.4},
    "disgust":  {"exaggeration": 0.50, "cfg": 0.6},
    "neutral":  {"exaggeration": 0.60, "cfg": 0.5},
}


# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

_model     = None
_classifier = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        raise RuntimeError("Run: pip install chatterbox-tts") # if missing dependency
    
    ref = Path(REFERENCE_CLIP)
    if not ref.exists():
        raise FileNotFoundError(
            f"\n[TTS] Reference clip not found: {ref}\n"
            f"      Put your reference audio file there and update REFERENCE_CLIP in tts_engine.py"
        )
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TTS] Loading Chatterbox on {device}...")
    _model = ChatterboxTTS.from_pretrained(device=device)
    print("[TTS] Chatterbox ready ✓")
    return _model
 

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
            device = 0 if torch.cuda.is_available() else -1,
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
    """
    Full pipeline: text -> emotion -> Chatterbox -> WAV bytes
    Returns empty bytes on failure so the chat keeps working.
    """
    text = _clean(text)
    if not text:
        return b""
    
    # Split into sentences for faster chunk generation
    import re as _re
    sentences = _re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [text]
    text = " ".join(sentences[:3]) #Only speak first 3 sentence for speed
    if len(text) < 20:
        text = " ".join(sentences)
 
    emotion = detect_emotion(text)
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])
 
    print(f"[TTS] '{emotion}' -> exaggeration={profile['exaggeration']} cfg={profile['cfg']}")
 
    try:
        import torchaudio
        import torch

        model = _get_model()

        audio_chunks = []
        for sentence in sentences:
            if not sentence.strip():
                continue
 
            wav = model.generate(
                sentence,
                audio_prompt_path = REFERENCE_CLIP,
                exaggeration      = profile["exaggeration"],
                cfg_weight        = profile["cfg"],
            )
            audio_chunks.append(wav)

        if not audio_chunks:
            return b""
        
        full_wav = torch.cat(audio_chunks, dim=-1)

        buf = io.BytesIO()
        torchaudio.save(buf, full_wav, model.sr, format="wav")
        buf.seek(0)
        return buf.read()
 
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return b""
 
# ─────────────────────────────────────────────────────────────────────────────
# Voice switching — called from main.py /set-voice endpoint
# ─────────────────────────────────────────────────────────────────────────────
 
def set_reference_clip(path: str):
    """Switch to a different voice reference clip without restarting.

    If the clip is missing or empty, keep the current voice instead of
    crashing — a persona without its own voice just uses whatever was loaded.
    """
    global REFERENCE_CLIP
    if not path:
        return
    if not Path(path).exists():
        print(f"[TTS] Voice clip not found, keeping current voice: {path}")
        return
    REFERENCE_CLIP = path
    print(f"[TTS] Voice switched to: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Generate audio sentence by sentence
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_sentence(text: str) -> bytes:
    """Generate audio for a single sentence."""
    text = _clean(text)
    if not text:
        return b""

    emotion = detect_emotion(text)
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])

    try:
        import torchaudio
        import torch

        model = _get_model()
        wav = model.generate(
            text,
            audio_prompt_path = REFERENCE_CLIP,
            exaggeration      = profile["exaggeration"],
            cfg_weight        = profile["cfg"],
        )
        buf = io.BytesIO()
        torchaudio.save(buf, wav, model.sr, format="wav")
        buf.seek(0)
        return buf.read()

    except Exception as e:
        print(f"[TTS] Error: {e}")
        return b""