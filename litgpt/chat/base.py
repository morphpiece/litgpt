# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Iterator, List, Literal, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer
from litgpt.generate.base import next_token
from litgpt.prompts import has_prompt_style, load_prompt_style
from litgpt.scripts.merge_lora import merge_lora
from litgpt.utils import (
    auto_download_checkpoint,
    check_file_size_on_cpu_and_warn,
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    stop_tokens: Tuple[List[int], ...] = (),
) -> Iterator[torch.Tensor]:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as possible.

    Arguments:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        stop_tokens: If specified, stop generating any more token once one of this list is generated.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device = prompt.device
    buffer_length = max((len(tokens) for tokens in stop_tokens), default=1)
    yield_i = 0
    input_pos = torch.arange(0, T, device=device)
    tokens = []
    token = prompt
    for t in range(1, max_returned_tokens - T + 1):
        token = next_token(model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k, top_p=top_p)
        tokens.append(token)
        # check the stop condition
        if any((l := len(st)) <= len(tokens) and all(a == b for a, b in zip(tokens[-l:], st)) for st in stop_tokens):
            return
        # if the buffer is full
        if t - yield_i >= buffer_length:
            # we know this idx is not part of stop tokens, safe to yield
            yield from tokens[yield_i:t]
            yield_i = t
        input_pos = input_pos[-1:].add_(1)


def decode(fabric: L.Fabric, tokenizer: Tokenizer, token_stream: Iterator[torch.Tensor]) -> int:
    tokens_generated = 0
    if tokenizer.backend == "huggingface":
        try:
            for token in token_stream:
                fabric.print(tokenizer.decode(token), end="", flush=True)
                tokens_generated += 1
        except KeyboardInterrupt:
            # support stopping generation
            return tokens_generated
    elif tokenizer.backend == "sentencepiece":
        # sentencepiece does not support decoding token-by-token because it adds spaces based on the surrounding tokens
        # meaning that we need to decode everything each time
        so_far = torch.tensor([], dtype=torch.long, device=fabric.device)
        decoded_so_far = ""
        try:
            for token in token_stream:
                so_far = so_far.to(device=token.device)
                so_far = torch.cat((so_far, token.view(-1)))
                decoded_new = tokenizer.decode(so_far)
                fabric.print(decoded_new[len(decoded_so_far) :], end="", flush=True)
                decoded_so_far = decoded_new
                tokens_generated += 1
        except KeyboardInterrupt:
            # support stopping generation
            return tokens_generated
    else:
        raise NotImplementedError(tokenizer.backend)
    return tokens_generated


def process_prompt(prompt, model, tokenizer, prompt_style, fabric, temperature, max_new_tokens, top_k, top_p, stop_tokens):
    prompt = prompt_style.apply(prompt=prompt)
    encoded_prompt = tokenizer.encode(prompt, device=fabric.device)

    if max_new_tokens is None:
        max_returned_tokens = model.max_seq_length
    else:
        first_turn = model.mask_cache is None
        max_returned_tokens = encoded_prompt.size(0) + max_new_tokens
        if first_turn or max_returned_tokens > model.max_seq_length:
            model.max_seq_length = max_returned_tokens
            model.set_kv_cache(batch_size=1, device=fabric.device)

    y = generate(
        model, encoded_prompt, max_returned_tokens, temperature=temperature, top_k=top_k, top_p=top_p, stop_tokens=stop_tokens
    )
    fabric.print(">> Reply: ", end="")
    t0 = time.perf_counter()
    tokens_generated = decode(fabric, tokenizer, y)
    t = time.perf_counter() - t0
    for block in model.transformer.h:
        block.attn.kv_cache.reset_parameters()
    fabric.print(
        f"\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec,"
        f" {tokens_generated} tokens",
        file=sys.stderr,
    )
    fabric.print()


def interact(multiline, model, tokenizer, prompt_style, fabric, temperature, max_new_tokens, top_k, top_p, stop_tokens):
    while True:
        try:
            if not multiline:
                prompt = input(">> Prompt: ")
            else:
                print(">> Prompt: (Type '!submit' on a new line to end input).")
                prompt_lines = []
                while True:
                    line = input()
                    if line.strip().lower() in ("!submit", "!quit", "!exit"):
                        break
                    prompt_lines.append(line)
                prompt = "\n".join(prompt_lines)

        except KeyboardInterrupt:
            break

        prompt = prompt.lower().strip()
        if not prompt or prompt in ("!quit", "!exit"):
            break

        process_prompt(prompt, model, tokenizer, prompt_style, fabric, temperature, max_new_tokens, top_k, top_p, stop_tokens)


@torch.inference_mode()
def main(
    checkpoint_dir: Path,
    *,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 50,
    top_p: float = 1.0,
    temperature: float = 0.8,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
    multiline: bool = False,
    access_token: Optional[str] = None,
) -> None:
    """Chat with a model.

    Args:
        checkpoint_dir: A local path to a directory containing the model weights or a valid model name.
            You can get a list of valid model names via the `litgpt download list` command line argument.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to use compilation to speed up token generation. Will increase startup time.
        multiline: Whether to support multiline input prompts.
        access_token: Optional API token to access models with restrictions.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    # Merge if this is a raw LoRA checkpoint
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    if (checkpoint_dir / "lit_model.pth.lora").is_file() and not checkpoint_path.is_file():
        print("Merging LoRA weights with the base model. This won't take long and is a one-time-only thing.")
        merge_lora(checkpoint_dir)

    checkpoint_dir = auto_download_checkpoint(model_name=checkpoint_dir, access_token=access_token)
    check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)

    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    with fabric.init_module(empty_init=True):
        model = GPT(config)
        if compile:
            print(
                "IMPORTANT: with enabled compilation the KV-cache size is determined by model's maximum context size, which leads to "
                "a higher memory consumption. In case of an OOM error, try to set `--compile=False`."
            )
            model.set_kv_cache(batch_size=1)
    load_checkpoint(fabric, model, checkpoint_path)
    model.eval()

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead", dynamic=True)

    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir) else PromptStyle.from_config(config)
    )
    stop_tokens = prompt_style.stop_tokens(tokenizer)

    if multiline:
        exit_instruction = "To exit, enter '!quit' or '!exit' on an empty prompt and press 'Enter'."
    else:
        exit_instruction = "To exit, press 'Enter' on an empty prompt."

    print(f"Now chatting with {config.name}.\n{exit_instruction}\n")
    L.seed_everything(1234)

    interact(
        multiline=multiline,
        model=model,
        tokenizer=tokenizer,
        prompt_style=prompt_style,
        fabric=fabric,
        temperature=temperature,
        max_new_tokens=(None if compile else max_new_tokens),
        top_k=top_k,
        top_p=top_p,
        stop_tokens=stop_tokens
    )

    if fabric.device.type == "cuda":
        fabric.print(f"\nMemory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)
