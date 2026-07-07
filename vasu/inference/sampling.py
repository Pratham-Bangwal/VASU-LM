import torch
import torch.nn.functional as F


def sample_next_token(
    logits,
    input_ids,
    temperature=0.8,
    top_k=40,
    repetition_penalty=1.2,
):

    temperature = max(temperature, 1e-5)
    
    logits = logits[:, -1, :] / temperature

    # repetition penalty
    for token_id in set(input_ids[0].tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= repetition_penalty
        else:
            logits[0, token_id] /= repetition_penalty

    # top-k
    if top_k is not None:
        values, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(
            logits < values[:, [-1]],
            float("-inf"),
        )

    probs = F.softmax(logits, dim=-1)

    next_token = torch.multinomial(
        probs,
        num_samples=1,
    )

    return next_token