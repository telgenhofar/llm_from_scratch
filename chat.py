"""
Interactive CLI for prompting a trained model.

Usage:
    python chat.py --checkpoint checkpoints/ckpt.pt --tokenizer tokenizer.json

Once running:
    >>> Once upon a time
    [model streams output token-by-token]

    Commands:
        /temp 0.8       set sampling temperature
        /topk 50        set top-k sampling (0 = disabled)
        /max 200        set max new tokens
        /reset          (no-op, prompts are independent — placeholder for chat mode later)
        /quit           exit
"""

import argparse
import itertools
import sys
import threading
import time
from pathlib import Path

import torch

from LLM import GPT, ModelConfig
from tokenizer import Tokenizer


class ThinkingDots:
    def __init__(self, message: str = "thinking"):
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def _run(self):
        frames = itertools.cycle(["   ", ".  ", ".. ", "..."])
        while not self._stop.is_set():
            frame = next(frames)
            sys.stdout.write(f"\r\033[K{self.message}{frame}")
            sys.stdout.flush()
            time.sleep(0.25)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None


def load_model(checkpoint_path: Path, device: str) -> tuple[GPT, ModelConfig]:
    print(f"loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = GPT(model_cfg).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded model: {model.num_parameters():,} params, "
          f"trained for {ckpt['step']:,} steps")
    return model, model_cfg


@torch.no_grad()
def generate_streaming(model: GPT, tokenizer: Tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_k: int | None, device: str, dtype: torch.dtype) -> None:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        eot = tokenizer.special_tokens.get("<|endoftext|>")
        prompt_ids = [eot] if eot is not None else tokenizer.encode(" ")

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    pending_bytes = b""
    ctx = torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=dtype) \
        if "cuda" in device else torch.amp.autocast(device_type="cpu", enabled=False)

    eot = tokenizer.special_tokens.get("<|endoftext|>")

    dots = ThinkingDots("thinking")
    dots.__enter__()
    first_token = True

    try:
        with ctx:
            for _ in range(max_new_tokens):
                idx_cond = (idx if idx.size(1) <= model.cfg.context_length
                            else idx[:, -model.cfg.context_length:])
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_id = next_token.item()

                if eot is not None and next_id == eot:
                    break

                idx = torch.cat([idx, next_token], dim=1)

                pending_bytes += tokenizer.vocab[next_id]
                try:
                    text = pending_bytes.decode("utf-8")
                    if first_token:
                        dots.stop()
                        sys.stdout.write(prompt)
                        first_token = False
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    pending_bytes = b""
                except UnicodeDecodeError:
                    continue
    finally:
        dots.stop()

    if pending_bytes:
        sys.stdout.write(pending_bytes.decode("utf-8", errors="replace"))
    sys.stdout.write("\n")
    sys.stdout.flush()


def repl(model, tokenizer, device, dtype):
    settings = {"temperature": 0.8, "top_k": 50, "max_new_tokens": 200}
    print(f"\nReady. Settings: {settings}")
    print("Commands: /temp <f>  /topk <n>  /max <n>  /quit\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt.strip():
            continue

        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0]
            if cmd == "/quit":
                break
            elif cmd == "/temp" and len(parts) == 2:
                settings["temperature"] = float(parts[1])
                print(f"  temperature = {settings['temperature']}")
            elif cmd == "/topk" and len(parts) == 2:
                settings["top_k"] = int(parts[1])
                print(f"  top_k = {settings['top_k']}")
            elif cmd == "/max" and len(parts) == 2:
                settings["max_new_tokens"] = int(parts[1])
                print(f"  max_new_tokens = {settings['max_new_tokens']}")
            else:
                print("  unknown command")
            continue

        try:
            generate_streaming(
                model, tokenizer, prompt,
                max_new_tokens=settings["max_new_tokens"],
                temperature=settings["temperature"],
                top_k=settings["top_k"] if settings["top_k"] > 0 else None,
                device=device,
                dtype=dtype,
            )
        except KeyboardInterrupt:
            print("\n  [interrupted]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_best.pt",
                   help="Path to model checkpoint")
    p.add_argument("--tokenizer", type=str, default="tokenizer.json",
                   help="Path to tokenizer")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    args = p.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    tokenizer = Tokenizer.load(args.tokenizer)
    model, _ = load_model(Path(args.checkpoint), args.device)

    repl(model, tokenizer, args.device, dtype)


if __name__ == "__main__":
    main()