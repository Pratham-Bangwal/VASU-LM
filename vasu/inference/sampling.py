import math

import torch
import torch.nn.functional as F


def validate_sampling_parameters(
    temperature: float,
    top_k: int | None,
    top_p: float,
    repetition_penalty: float | None,
) -> None:
    """Reject invalid sampling controls before tensor operations begin."""

    if isinstance(temperature, bool) or not isinstance(
        temperature, (int, float)
    ):
        raise TypeError("temperature must be a finite positive number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
    ):
        raise ValueError("top_k must be None or a positive integer")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise TypeError("top_p must be a finite number in (0, 1]")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be a finite number in (0, 1]")
    if repetition_penalty is not None:
        if isinstance(repetition_penalty, bool) or not isinstance(
            repetition_penalty, (int, float)
        ):
            raise TypeError(
                "repetition_penalty must be None or a finite positive number"
            )
        if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
            raise ValueError(
                "repetition_penalty must be None or a finite positive number"
            )


def sample_next_token(
    logits,
    input_ids,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
):
    validate_sampling_parameters(
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    )

    logits = logits[:, -1, :] / temperature

    if input_ids is not None and repetition_penalty is not None:
        for token_id in set(input_ids[0].tolist()):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= repetition_penalty
            else:
                logits[0, token_id] /= repetition_penalty

    # top-k
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
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
