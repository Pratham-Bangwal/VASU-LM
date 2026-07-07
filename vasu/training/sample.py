import torch
from vasu.inference.generate import generate


@torch.no_grad()
def generate_sample(
    model,
    tokenizer,
    device,
):

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="Once upon a time",
        device=device,
        max_new_tokens=50,
        temperature=0.7,
        top_k=30,
    )

    print("\n" + "=" * 60)
    print("VASU SAMPLE")
    print("=" * 60)
    print(text)
    print("=" * 60)