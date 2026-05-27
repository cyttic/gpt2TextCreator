import re
import random
import torch
import time
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH     = "/mnt/ssd2/cyttic/models/dictalm2"
OUTPUT_FILE    = "generated_hebrew.txt"
N_SENTENCES    = 200
MAX_NEW_TOKENS = 30       # 15 words ≈ 20-25 tokens, 30 gives a small buffer
BATCH_SIZE     = 32
TARGET         = 1_000_000

SYSTEM_PROMPT = (
    "אתה עוזר שכותב אך ורק בעברית. "
    "חל איסור מוחלט להשתמש באנגלית או בכל שפה אחרת. "
    "כתוב משפטים עבריים תקינים וקצרים בלבד."
)

# ── Load word dictionary ──────────────────────────────────────────────────────
DICT_FILE = Path("words.txt")
if not DICT_FILE.exists():
    print("ERROR: words.txt not found. Run:  python3 download_dict.py")
    raise SystemExit(1)

WORDS = DICT_FILE.read_text(encoding="utf-8").splitlines()
WORDS = [w for w in WORDS if w.strip()]  # drop empty lines
print(f"Loaded dictionary: {len(WORDS):,} words")

def random_seed() -> str:
    """Pick 2 random English words as a topic seed."""
    return " ".join(random.sample(WORDS, 2))

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} (4-bit quantization)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()
DEVICE = next(model.parameters()).device

# ── Detect chat template ──────────────────────────────────────────────────────
HAS_CHAT_TEMPLATE = (
    hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
)
print(f"Chat template: {'yes ✓' if HAS_CHAT_TEMPLATE else 'no — using plain Hebrew prefix'}")
print(f"Ready. Device={DEVICE}  Batch size={BATCH_SIZE}\n")

def build_prompt(seed: str) -> str:
    """Wrap 2 random English topic words in a Hebrew system prompt."""
    if HAS_CHAT_TEMPLATE:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"כתוב משפט אחד בעברית על הנושא: {seed}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Base model: Hebrew instruction + topic as plain-text prefix
        return f"{SYSTEM_PROMPT}\n\nכתוב משפט אחד בעברית על הנושא: {seed}\n"

# ── Hebrew-only filter ────────────────────────────────────────────────────────
LATIN_RE = re.compile(r"[A-Za-z]")

def is_hebrew_only(text: str) -> bool:
    """Return True if the sentence contains no Latin/English characters."""
    return not LATIN_RE.search(text)

MAX_WORDS = 15

def trim_to_sentence(text: str) -> str:
    # Cut at first sentence boundary
    for sep in [".", "!", "?", "\n"]:
        if sep in text:
            text = text.split(sep)[0].strip() + sep
            break
    else:
        text = text.strip()

    # Hard cap at MAX_WORDS words
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
        # add a period if the truncation removed the punctuation
        if text[-1] not in ".!?":
            text += "."

    return text

# ── Batch generation ──────────────────────────────────────────────────────────
def generate_batch(seeds: list[str]) -> list[str]:
    prompts = [build_prompt(s) for s in seeds]
    inputs  = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.99,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for out in outputs:
        text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        results.append(trim_to_sentence(text))
    return results

# ── Generate until we have N_SENTENCES valid Hebrew-only lines ────────────────
print(f"Generating {N_SENTENCES} Hebrew-only sentences (batch={BATCH_SIZE})…\n")

sentences   = []
batch_times = []
total_generated = 0
skipped         = 0
start_total = time.perf_counter()

while len(sentences) < N_SENTENCES:
    # pick a fresh pair of random words for every sentence in the batch
    batch_seeds = [random_seed() for _ in range(BATCH_SIZE)]

    t0      = time.perf_counter()
    results = generate_batch(batch_seeds)
    dt      = time.perf_counter() - t0
    batch_times.append(dt)
    total_generated += len(results)

    for r in results:
        if is_hebrew_only(r) and r.strip():
            sentences.append(r)
            if len(sentences) >= N_SENTENCES:
                break
        else:
            skipped += 1

    done     = min(len(sentences), N_SENTENCES)
    avg_sent = sum(batch_times) / len(batch_times) / BATCH_SIZE
    eta      = avg_sent * (N_SENTENCES - done)
    bar      = "█" * (done * 40 // N_SENTENCES) + "░" * (40 - done * 40 // N_SENTENCES)
    print(f"\r[{bar}] {done}/{N_SENTENCES}  skipped={skipped}  ETA {eta:.0f}s", end="", flush=True)

total_time = time.perf_counter() - start_total
print()

# ── Save — plain lines, no numbering ─────────────────────────────────────────
out_path = Path(OUTPUT_FILE)
with out_path.open("w", encoding="utf-8") as f:
    for s in sentences:
        f.write(s + "\n")

print(f"\nSaved {N_SENTENCES} sentences → {out_path.resolve()}")
print(f"Skipped (had English): {skipped} / {total_generated} total generated")

# ── Stats ─────────────────────────────────────────────────────────────────────
throughput  = len(sentences) / total_time
ext_seconds = TARGET / throughput
ext_td      = datetime.timedelta(seconds=int(ext_seconds))

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Results  (batch={BATCH_SIZE}, system_prompt={'chat' if HAS_CHAT_TEMPLATE else 'prefix'})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Valid sentences     : {len(sentences)}
  Total generated     : {total_generated}
  Skipped (English)   : {skipped}
  Total wall time     : {total_time:.1f}s
  Throughput          : {throughput:.2f} sent/sec

  ── Extrapolation to {TARGET:,} sentences ──
  Estimated time      : {ext_seconds:,.0f}s
    • {ext_seconds/3600:.1f} hours
    • {ext_seconds/86400:.1f} days
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
