import torch

from .sampling import sample_next_token
from vasu.cache import KVCache, PreallocatedKVCache
from vasu.data.prompt_templates import format_prompt


def _select_next_token(
    logits,
    token_history,
    temperature,
    top_k,
    top_p,
    do_sample,
    repetition_penalty,
):
    if do_sample:
        return sample_next_token(
            logits,
            token_history,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )
    return torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )


@torch.no_grad()
def generate_token_ids(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    prompt_format="alpaca",
    use_kv_cache: bool = False,
    kv_cache_implementation: str = "dynamic",
    repetition_penalty: float = 1.1,
):
    """Generate token IDs with the reference or explicit cached path."""

    model.eval()
    formatted_prompt = format_prompt(prompt, prompt_format=prompt_format)
    encoded_prompt = tokenizer.encode(formatted_prompt)
    if use_kv_cache and not encoded_prompt:
        raise ValueError("cached generation requires a non-empty prompt")
    input_ids = torch.tensor([encoded_prompt], device=device)
    prompt_length = input_ids.shape[1]
    eos_token = tokenizer.tokenizer.token_to_id("[EOS]")
    max_seq_len = getattr(getattr(model, "config", None), "max_seq_len", None)

    if use_kv_cache:
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("cached generation requires model.config")
        if prompt_length > config.max_seq_len:
            raise ValueError("prompt exceeds model maximum sequence length")
        if kv_cache_implementation == "dynamic":
            cache = KVCache(config.n_layers, config.max_seq_len)
        elif kv_cache_implementation == "preallocated":
            cache = PreallocatedKVCache(
                config.n_layers,
                config.max_seq_len,
                input_ids.size(0),
                config.n_heads,
                config.dim // config.n_heads,
                device=input_ids.device,
                dtype=next(model.parameters()).dtype,
            )
        else:
            raise ValueError(
                "kv_cache_implementation must be 'dynamic' or 'preallocated'"
            )
        logits = model(input_ids, kv_cache=cache, cache_mode="prefill")
        token_history = torch.empty(
            (input_ids.size(0), config.max_seq_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        token_history[:, :prompt_length].copy_(input_ids)
        history_length = prompt_length
        for _ in range(max_new_tokens):
            if history_length >= config.max_seq_len:
                break
            history_view = token_history[:, :history_length]
            next_token = _select_next_token(
                logits,
                history_view,
                temperature,
                top_k,
                top_p,
                do_sample,
                repetition_penalty,
            )
            token_history[:, history_length].copy_(next_token[:, 0])
            history_length += 1
            if next_token.item() == eos_token:
                break
            if history_length >= config.max_seq_len:
                break
            logits = model(
                next_token,
                kv_cache=cache,
                cache_mode="decode",
            )
        return token_history[0, prompt_length:history_length].tolist()

    # Reference implementation: retain full-history model calls. The only
    # added guard prevents a call beyond a declared model context window.
    token_history = input_ids
    for _ in range(max_new_tokens):
        if max_seq_len is not None and token_history.size(1) >= max_seq_len:
            break
        logits = model(token_history, kv_cache=None)
        next_token = _select_next_token(
            logits,
            token_history,
            temperature,
            top_k,
            top_p,
            do_sample,
            repetition_penalty,
        )
        token_history = torch.cat((token_history, next_token), dim=1)
        if next_token.item() == eos_token:
            break
    return token_history[0, prompt_length:].tolist()

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
    do_sample=True,
    prompt_format="alpaca",
    use_kv_cache: bool = False,
    kv_cache_implementation: str = "dynamic",
    repetition_penalty: float = 1.1,
):
    generated_ids = generate_token_ids(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        prompt_format=prompt_format,
        use_kv_cache=use_kv_cache,
        kv_cache_implementation=kv_cache_implementation,
        repetition_penalty=repetition_penalty,
    )
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )
