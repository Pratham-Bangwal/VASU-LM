# Known Issues

## Model quality

### Factual hallucinations and limited knowledge

Both model families can invent people, places, statistics, and explanations. VASU-60M base pretraining has improved grammar but has not established reliable factuality.

### Repetition loops

Long or greedy generations can repeat words, phrases, sentence forms, or topic fragments. Sampling reduces some loops but does not solve the underlying behavior.

### Semantic drift and weak topic retention

Outputs may begin near the prompt and then move to unrelated concepts. Raw base continuations are especially susceptible.

### Weak long-range coherence and reasoning

The models struggle to preserve a consistent argument, narrative, or chain of reasoning. The 256-token context further limits longer tasks.

### Exact-formatting failures

Instruction-tuned VASU-31M can ignore requested counts, list lengths, line counts, or response formats.

### Incomplete generations

Responses may terminate mid-word or mid-thought when the generation budget is reached.

## Training-system issues and mitigations

### Laptop thermal constraints — mitigated operationally

Sustained VASU-60M training can heat the laptop quickly. The block trainer stops at 88°C, while the long runner uses short blocks and a 10-minute cooldown after thermal-stop checkpoints. Physical ventilation and user monitoring remain necessary.

### Checkpoint disk usage — mitigated

Optimizer-bearing VASU-60M checkpoints are large, and unlimited periodic saves previously filled disk space. Bounded retention and a minimum-free-space check are now used. Preserved milestone checkpoints live outside the retention directory.

### Interrupted checkpoint corruption — mitigated

Incomplete serialization can leave unusable files. Saves now use a temporary path followed by atomic replacement, and resume selection ignores corrupt, incomplete, and temporary candidates.

### General DataLoader sampler state is not fully restored — open

The general trainer restores model, optimizer, scheduler, epoch, and global step but does not restore the exact DataLoader shuffle position after interruption. Data may repeat or be skipped. The specialized VASU-60M block trainer mitigates this by deriving the FineWeb slice from `global_step`, but this does not fix general sampler-state restoration.

### VASU-31M and VASU-60M weights are incompatible by shape — by design

The two model sizes share tokenizer and checkpoint container conventions but have different tensor shapes. Loading one model's state dict into the other is unsupported.

### Block scripts are specialized operational tools

The VASU-60M block trainer and 11-hour runner encode assumptions for the current laptop, dataset, checkpoint directory, and training recipe. They are not general distributed-training tools.

### Validation is narrow

Block training uses a small fixed held-out sample at selected milestones. This is useful for trend monitoring but is not a complete benchmark and can miss regressions in factuality, reasoning, or generation behavior.

## Not yet implemented or mature

- broader formal evaluation;
- longer context;
- complete data-sampler resume state;
- mature KV-cache optimization;
- streaming generation;
- quantization;
- VASU-60M instruction and safety tuning;
- reliable tools, memory, voice, or agent behavior.
