"""
Tamil Speech → ISL (Indian Sign Language) Gloss Pipeline
=========================================================
Requirements:
    pip install sounddevice scipy elevenlabs stanza openai

Usage:
    python tamil_to_isl.py
"""

import sounddevice as sd
from scipy.io.wavfile import write
from elevenlabs.client import ElevenLabs
import stanza
import openai
import json
import os
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
DURATION    = 5     
AUDIO_FILE  = "recorded.wav"

KEEP_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}


def record_audio(duration: int = DURATION, sample_rate: int = SAMPLE_RATE) -> str:
    print(f"\n Recording for {duration} seconds... Speak now in Tamil!")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    write(AUDIO_FILE, sample_rate, audio)
    print(f"Recording saved → {AUDIO_FILE}")
    return AUDIO_FILE

def speech_to_text(audio_path: str) -> str:
    print("\nConverting speech to Tamil text...")
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    with open(audio_path, "rb") as f:
        transcription = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            language_code="ta" 
        )
    tamil_text = transcription.text.strip()
    print(f"Transcribed Tamil: {tamil_text}")
    return tamil_text


def load_stanza_pipeline() -> stanza.Pipeline:
    print("\noading Stanza Tamil NLP pipeline...")
    try:
        nlp = stanza.Pipeline("ta", processors="tokenize,pos", verbose=False)
    except Exception:
        print("   Tamil model not found — downloading (one-time)...")
        stanza.download("ta")
        nlp = stanza.Pipeline("ta", processors="tokenize,pos", verbose=False)
    print("Stanza pipeline ready.")
    return nlp


def tokenize_and_tag(nlp: stanza.Pipeline, text: str) -> list[dict]:
    doc = nlp(text)
    tokens = []
    for sentence in doc.sentences:
        for word in sentence.words:
            tokens.append({
                "word": word.text,
                "lemma": word.lemma or word.text,
                "upos": word.upos,
                "keep": word.upos in KEEP_UPOS,
            })
    print("\nPOS-tagged tokens:")
    for t in tokens:
        marker = "✓" if t["keep"] else "✗"
        print(f"   {marker}  {t['word']:<20} [{t['upos']}]")
    return tokens


def build_llm_prompt(tokens: list[dict], original_text: str) -> str:
    # Only pass kept tokens to the LLM to reduce noise
    content_words = [
        f"{t['word']} ({t['upos']})"
        for t in tokens if t["keep"]
    ]
    return f"""You are an expert in Indian Sign Language (ISL) linguistics.

Original Tamil sentence:
"{original_text}"

Content-word tokens extracted (already filtered by POS):
{', '.join(content_words)}

Task: Convert to ISL gloss following these rules:
1. ISL uses Subject-Object-Verb (SOV) word order.
2. Keep only meaningful content words (nouns, verbs, adjectives, numbers, proper nouns).
3. Drop all function words: postpositions, auxiliaries, particles, conjunctions.
4. Use the ROOT / dictionary form of each word (do not inflect).
5. Write the gloss in CAPITAL LETTERS, words separated by hyphens.
6. Also provide a short English gloss for reference.

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "isl_gloss": "WORD1-WORD2-WORD3",
  "english_gloss": "english equivalent gloss",
  "explanation": "one sentence explaining the grammar change"
}}"""


def grammar_shift_to_isl(tokens: list[dict], original_text: str) -> dict:
    print("\nSending to LLM for ISL grammar conversion...")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_llm_prompt(tokens, original_text)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)
    return result

def display_result(tamil_text: str, tokens: list[dict], isl_result: dict) -> None:
    print("\n" + "═" * 55)
    print("  TAMIL → ISL CONVERSION RESULT")
    print("═" * 55)
    print(f"  Tamil input   : {tamil_text}")
    print(f"  ISL gloss     : {isl_result.get('isl_gloss', 'N/A')}")
    print(f"  English gloss : {isl_result.get('english_gloss', 'N/A')}")
    print(f"  Note          : {isl_result.get('explanation', '')}")
    print("═" * 55)


def main():
    # Load Stanza once (reusable across multiple sentences)
    nlp = load_stanza_pipeline()

    while True:
        print("\n" + "─" * 40)
        choice = input("Press ENTER to record, or type a Tamil sentence directly, or 'q' to quit: ").strip()

        if choice.lower() == "q":
            print("Goodbye!")
            break

        # ── Get Tamil text ──────────────────────
        if choice == "":
            audio_path  = record_audio()
            tamil_text  = speech_to_text(audio_path)
        else:
            tamil_text = choice   # use typed input for testing

        if not tamil_text:
            print("No text detected. Try again.")
            continue

        # ── NLP pipeline ───────────────────────
        tokens     = tokenize_and_tag(nlp, tamil_text)
        isl_result = grammar_shift_to_isl(tokens, tamil_text)

        # ── Show result ─────────────────────────
        display_result(tamil_text, tokens, isl_result)


# import requests
#
# def grammar_shift_to_isl_ollama(tokens, original_text):
#     prompt = build_llm_prompt(tokens, original_text)
#     resp = requests.post("http://localhost:11434/api/generate", json={
#         "model": "mistral",
#         "prompt": prompt,
#         "stream": False,
#     })
#     raw = resp.json()["response"].strip()
#     return json.loads(raw)
# ──────────────────────────────────────────────


if __name__ == "__main__":
    main()