import torch

from .sampling import sample_next_token
from .prompt import build_prompt
from vasu.cache import KVCache

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
):

    model.eval()
    
    prompt = build_prompt(prompt)

    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        device=device,
    )

    prompt_length = input_ids.shape[1]

    eos_token = tokenizer.tokenizer.token_to_id("[EOS]")
    kv_cache = KVCache(
        model.config.n_layers
    )

    is_first_step = True
    
    for _ in range(max_new_tokens):

        if is_first_step:
            model_input = input_ids
        else:
            model_input = input_ids[:, -1:]

        logits = model(
            model_input,
            kv_cache=kv_cache,
        )


        next_token = sample_next_token(
            logits,
            input_ids,
            temperature,
            top_k,
            top_p,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

        is_first_step = False

        if next_token.item() == eos_token:
            break

    generated_ids = input_ids[
        0,
        prompt_length:
    ].tolist()

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )
