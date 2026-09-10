"""Sentiment/emotion lexicon loaded from the NRC Word-Emotion Association
Lexicon (NRCLexicon/*.txt): one file per emotion, each listing the full
~14k-word vocabulary with a 0/1 association flag (word<TAB>flag per line).
"""

from pathlib import Path

LEXICON_DIR = Path(__file__).resolve().parent / "NRCLexicon"

EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust",
]


def _load_words(emotion: str) -> set:
    path = LEXICON_DIR / f"{emotion}-NRC-Emotion-Lexicon.txt"
    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            word, flag = parts
            if flag == "1":
                words.add(word)
    return words


EMOTION_WORDS = {emotion: _load_words(emotion) for emotion in EMOTIONS}

POSITIVE_WORDS = sorted(_load_words("positive"))
NEGATIVE_WORDS = sorted(_load_words("negative"))
