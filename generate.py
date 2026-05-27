import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "/mnt/ssd2/cyttic/models/dictalm2"

print(f"Loading model from {MODEL_PATH} (4-bit quantization)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",          # places layers on GPU/CPU automatically
)
model.eval()
DEVICE = next(model.parameters()).device
print(f"Model loaded on {DEVICE}.\n")

def generate(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Hebrew text generator | type a word or phrase, Ctrl+C to exit\n")

while True:
    try:
        prompt = input(">>> ")
        if not prompt.strip():
            continue
        print(generate(prompt))
        print()
    except KeyboardInterrupt:
        print("\nBye!")
        break
