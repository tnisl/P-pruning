import logging
import random
import sys

import torch
from src.utils.submodule import Architecture
from tqdm import tqdm

logger = logging.getLogger()


def collect_hidden_states(arch: Architecture, dataloader):
    model_config = arch.model_config
    sampled = 200
    for index, arg in enumerate(sys.argv[1:]):
        if arg == "--sampling_tokens":
            sampled = int(sys.argv[index + 2])
        elif arg.startswith("--sampling_tokens="):
            sampled = int(arg.split("=", 1)[1])

    dataset = dataloader.dataset
    total_tokens = sum(int(sum(dataset[index]["attention_mask"])) for index in range(len(dataset)))
    assert 0 < sampled <= total_tokens

    # mask.py normally draws these indices after collecting every activation.
    # The forward passes below are in eval mode and do not consume Python random state.
    selected_indices = random.sample(range(total_tokens), sampled)
    state = random.getstate()
    downstream_order = random.sample(range(sampled), sampled)
    random.setstate(state)
    slot_for_token = dict(zip(selected_indices, downstream_order))

    selected_by_example = {}
    token = 0
    for example_index in range(len(dataset)):
        for position, valid in enumerate(dataset[example_index]["attention_mask"]):
            if valid:
                slot = slot_for_token.get(token)
                if slot is not None:
                    selected_by_example.setdefault(example_index, []).append((position, slot))
                token += 1
    assert token == total_tokens

    attention = [None] * model_config["layer_num"]
    neurons = [None] * model_config["layer_num"]
    current = []
    handles = []

    def capture(cache, layer):
        def hook(_, inputs):
            value = inputs[0].detach()
            if cache[layer] is None:
                cache[layer] = torch.empty(sampled, value.shape[-1], dtype=value.dtype)
            for row, position, slot in current:
                cache[layer][slot] = value[row, position].cpu()
        return hook

    for layer in range(model_config["layer_num"]):
        handles.append(arch.get_attn_o(layer).register_forward_pre_hook(capture(attention, layer)))
        handles.append(arch.get_ffn2(layer).register_forward_pre_hook(capture(neurons, layer)))

    arch.model.eval()
    logger.info("Start collecting %s sampled hidden states", sampled)
    offset = 0
    try:
        for batch in tqdm(dataloader, total=len(dataloader)):
            current = [
                (row, position, slot)
                for row in range(len(batch["input_ids"]))
                for position, slot in selected_by_example.get(offset + row, ())
            ]
            offset += len(batch["input_ids"])
            if not current:
                continue
            batch = {key: value.to(arch.model.device) for key, value in batch.items()}
            with torch.no_grad():
                arch.model(**batch)
    finally:
        for handle in handles:
            handle.remove()

    assert offset == len(dataset)
    assert all(cache is not None for cache in attention + neurons)
    return attention, neurons
