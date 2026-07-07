import torch
import torch.nn.functional as F


def sample_next_token(
    logits,
    input_ids,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
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

    # top-p
    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )

    sorted_probs = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probs = sorted_probs.cumsum(dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p

    sorted_indices_to_remove[..., 1:] = (
        sorted_indices_to_remove[..., :-1].clone()
    )

    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )

    logits = logits.masked_fill(
        indices_to_remove,
        float("-inf"),
    )

    probs = F.softmax(logits, dim=-1)

    next_token = torch.multinomial(
        probs,
        num_samples=1,
    )

    return next_token