import torch

from vasu.model.model import VASUModel
from vasu.config import ModelConfig
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.inference.generate import generate
import time



device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load tokenizer
# -------------------------
tokenizer = VASUTokenizer()
tokenizer.load("assets/tokenizer.json")

# -------------------------
# Load model
# -------------------------
config = ModelConfig()
model = VASUModel(config).to(device)

checkpoint = torch.load("checkpoints/ultrachat/best.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

print("VASU ready. Type something (exit to stop)")

# -------------------------
# Chat loop
# -------------------------
while True:
    prompt = input("\nYou: ")

    if prompt.lower() == "exit":
        break
    
    start = time.time()

    response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    device=device,
    max_new_tokens=50,
    temperature=0.7,
    top_k=30,
    top_p=0.9,
    )
    print(
        f"\nGeneration time: "
        f"{time.time()-start:.2f}s"
    )

    if response.startswith(prompt):
        response = response[len(prompt):]

    print("\nVASU:", response.strip())
