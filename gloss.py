"""
Requirements:
    pip install sounddevice scipy elevenlabs stanza langchain langchain-openai langchain-core
"""

import json
import os
import sounddevice as sd
from scipy.io.wavfile import write
from elevenlabs.client import ElevenLabs
import stanza

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000
DURATION    = 5
AUDIO_FILE  = "recorded.wav"

KEEP_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}

class ISLOutput(BaseModel):
    isl_gloss: str = Field(description="ISL gloss in CAPITAL-HYPHENATED form e.g. BOY SCHOOL GO")
    english_gloss: str = Field(description="English equivalent of the ISL gloss")
    dropped_words: list[str] = Field(description="Words removed as function words")
    explanation: str = Field(description="One sentence explaining the grammar transformation")

def record_audio(duration: int = DURATION, sample_rate: int = SAMPLE_RATE) -> str:
    print(f"\nRecording for {duration} seconds... Speak now in Tamil!")
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
        )
    tamil_text = transcription.text.strip()
    print(f"Transcribed: {tamil_text}")
    return tamil_text

def load_stanza_pipeline() -> stanza.Pipeline:
    print("\nLoading Stanza Tamil pipeline...")
    try:
        nlp = stanza.Pipeline("ta", processors="tokenize,pos", verbose=False)
    except Exception:
        print("   Downloading Tamil model (one-time)...")
        stanza.download("ta")
        nlp = stanza.Pipeline("ta", processors="tokenize,pos", verbose=False)
    print("Stanza ready.")
    return nlp


def tokenize_and_tag(nlp: stanza.Pipeline, text: str) -> dict:
    doc = nlp(text)
    all_tokens = []
    content_tokens = []

    for sentence in doc.sentences:
        for word in sentence.words:
            token = {
                "word": word.text,
                "lemma": word.lemma or word.text,
                "upos": word.upos,
            }
            all_tokens.append(token)
            if word.upos in KEEP_UPOS:
                content_tokens.append(token)

    print("\nPOS-tagged tokens:")
    for t in all_tokens:
        marker = "✓" if t in content_tokens else "✗"
        print(f"   {marker}  {t['word']:<20} [{t['upos']}]")

    return {
        "all_tokens": all_tokens,
        "content_tokens": content_tokens,
    }


def build_isl_chain(llm: ChatOpenAI):
    """
    LangChain LCEL chain:
        input dict → prompt → LLM → JsonOutputParser → ISLOutput
    """
    parser = JsonOutputParser(pydantic_object=ISLOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert linguist in Indian Sign Language (ISL) and Tamil.
Your job is to convert Tamil text into ISL gloss following strict rules.

ISL Grammar Rules:
- Word order: Subject-Object-Verb (SOV)
- Drop all function words: postpositions, auxiliaries, particles, conjunctions, articles
- Use dictionary/root form of each content word
- Write gloss in CAPITAL LETTERS, hyphen-separated
- ISL is visual — keep only the words that carry meaning

{format_instructions}"""),
        ("human", """Tamil sentence: "{tamil_text}"

Content-word tokens (POS-filtered):
{content_words}

Convert to ISL gloss.""")
    ])

    chain = (
        RunnablePassthrough.assign(
            format_instructions=lambda _: parser.get_format_instructions()
        )
        | prompt
        | llm
        | parser
    )

    return chain



def build_full_pipeline(nlp: stanza.Pipeline, isl_chain):
    """
        tamil_text → tag → format → isl_chain → result
    """

    def tag_and_format(tamil_text: str) -> dict:
        tagged = tokenize_and_tag(nlp, tamil_text)
        content_words_str = ", ".join(
            f"{t['word']} ({t['upos']})" for t in tagged["content_tokens"]
        )
        return {
            "tamil_text": tamil_text,
            "content_words": content_words_str,
            "all_tokens": tagged["all_tokens"],
            "content_tokens": tagged["content_tokens"],
        }

    def run_isl_chain(data: dict) -> dict:
        print("\nRunning LangChain ISL conversion chain...")
        result = isl_chain.invoke({
            "tamil_text": data["tamil_text"],
            "content_words": data["content_words"],
        })
        return {**data, "isl_result": result}

    pipeline = (
        RunnableLambda(tag_and_format)
        | RunnableLambda(run_isl_chain)
    )

    return pipeline


def display_result(data: dict) -> None:
    r = data["isl_result"]
    print("\n" + "═" * 58)
    print("  TAMIL → ISL CONVERSION RESULT")
    print("═" * 58)
    print(f"  Tamil input    : {data['tamil_text']}")
    print(f"  ISL gloss      : {r.get('isl_gloss', 'N/A')}")
    print(f"  English gloss  : {r.get('english_gloss', 'N/A')}")
    print(f"  Dropped words  : {', '.join(r.get('dropped_words', []))}")
    print(f"  Note           : {r.get('explanation', '')}")
    print("═" * 58)


def main():
    nlp = load_stanza_pipeline()

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )

    isl_chain = build_isl_chain(llm)
    pipeline  = build_full_pipeline(nlp, isl_chain)

    print("\nFull LangChain pipeline ready.")

    while True:
        print("\n" + "─" * 45)
        choice = input(
            "Press ENTER to record | type Tamil text | 'q' to quit: "
        ).strip()

        if choice.lower() == "q":
            print("Goodbye!")
            break

        if choice == "":
            audio_path = record_audio()
            tamil_text = speech_to_text(audio_path)
        else:
            tamil_text = choice  

        if not tamil_text:
            print(" No text detected. Try again.")
            continue

        try:
            result = pipeline.invoke(tamil_text)
            display_result(result)
        except Exception as e:
            print(f"Pipeline error: {e}")


# ──────────────────────────────────────────────
# ALTERNATIVE: Swap OpenAI for Ollama (free/local)
# ──────────────────────────────────────────────
# from langchain_community.chat_models import ChatOllama
#
# llm = ChatOllama(model="mistral", temperature=0.2)
#
# Works identically — LangChain abstracts the provider.
# Run: ollama pull mistral  →  ollama serve
# ──────────────────────────────────────────────

if __name__ == "__main__":
    main()