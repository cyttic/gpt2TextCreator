import torch
import transformers as tr

MODEL_PATH = "/mnt/ssd2/cyttic/projects/gpt2TextCreator/model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt = tr.GPT2LMHeadModel.from_pretrained(MODEL_PATH)
ttokenizer = tr.PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
gpt.to(device)
gpt.eval()

print("Hebrew text generator | type a word or phrase, Ctrl+C to exit\n")

while True:
    try:
        prompt = input(">>> ")
        if not prompt.strip():
            continue

        inputs = ttokenizer(prompt, return_tensors="pt").to(device)

        output = gpt.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=30,
            top_p=0.85,
            repetition_penalty=2.0
        )

        print(ttokenizer.decode(output[0], skip_special_tokens=True))
        print()

    except KeyboardInterrupt:
        print("\nBye!")
        break
