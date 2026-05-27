"""
Downloads Google's ~10k most common English words (no swears), ranked by frequency.
Source: https://github.com/first20hours/google-10000-english
Filters down to clean, meaningful nouns/words (4–12 chars, letters only).
"""
import urllib.request
import sys
from pathlib import Path

URL      = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
OUT_FILE = Path("words.txt")

if OUT_FILE.exists():
    count = sum(1 for _ in OUT_FILE.open())
    print(f"words.txt already exists ({count:,} words). Delete it to re-download.")
    sys.exit(0)

print(f"Downloading common word list...")

tmp = Path("words_raw.txt")
urllib.request.urlretrieve(URL, tmp)

raw   = tmp.read_text(encoding="utf-8").splitlines()
words = [
    w.strip().lower() for w in raw
    if 4 <= len(w.strip()) <= 12      # skip very short (a, the, of) and very long words
    and w.strip().isalpha()           # letters only, no hyphens/apostrophes
]
tmp.unlink()

OUT_FILE.write_text("\n".join(words), encoding="utf-8")
print(f"Done. Saved {len(words):,} words → {OUT_FILE.resolve()}")
print("Preview:", ", ".join(words[:20]))
