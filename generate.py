import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "dicta-il/dictalm2.0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
model.to(DEVICE)
model.eval()
print("Model loaded.\n")

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
