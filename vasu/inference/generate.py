import torch

from .sampling import sample_next_token


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
):

    model.eval()

    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        device=device,
    )

    prompt_length = input_ids.shape[1]

    eos_token = tokenizer.tokenizer.token_to_id("[EOS]")

    for _ in range(max_new_tokens):

        logits = model(input_ids)


        next_token = sample_next_token(
            logits,
            input_ids,
            temperature,
            top_k,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

        if next_token.item() == eos_token:
            break

    return tokenizer.decode(
        input_ids[0].tolist()
    )