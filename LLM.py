import argparse
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 512
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1
    tie_weights: bool = True

@dataclass
class TrainConfig:
    data_path: str = "train.bin"
    val_data_path: str = "val.bin"
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_steps: int = 100_000
    warmup_steps: int = 500
    grad_accum_steps: int = 1
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 50
    checkpoint_interval: int = 1000
    max_runtime_hours: float = 23.5
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = True
    seed: int = 0

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V in one go, then split
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Fused attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        # Merge heads back: (B, n_heads, T, d_head) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
    
class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))
    
class Block(nn.Module):
    """Transformer block"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_final = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
                self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.position_embedding.weight.numel()
        return n
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.context_length, f"sequence length {T} exceeds context {self.cfg.context_length}"
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
 
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None
        
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.context_length else idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
 
    def configure_optimizer(self, weight_decay: float, lr: float, betas: tuple[float, float], device_type: str) -> torch.optim.Optimizer:
        params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        no_decay_params = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
    
class DataLoader:
    def __init__(self, path: str, batch_size: int, context_length: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        print(f"loaded {path}: {len(self.data):,} tokens")

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(0, len(self.data) - self.context_length - 1, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i + self.context_length].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1:i + 1 + self.context_length].astype(np.int64))
            for i in ix
        ])
        x = x.pin_memory().to(self.device, non_blocking=True)
        y = y.pin_memory().to(self.device, non_blocking=True)
        return x, y
    
def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

@torch.no_grad()
def estimate_loss(model: nn.Module, loaders: dict[str, DataLoader], eval_iters: int, ctx) -> dict[str, float]:
    out = {}
    model.eval()
    for split, loader in loaders.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = loader.get_batch()
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, best_val_loss: float, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    print(f"  saved checkpoint to {path}")
 
 
def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> tuple[int, float]:
    print(f"resuming from {path}")
    ckpt = torch.load(path, map_location=device)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng_state"])
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
    return ckpt["step"], ckpt["best_val_loss"]

def train(model_cfg: ModelConfig, train_cfg: TrainConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    best_path = out_dir / "ckpt_best.pt"
 
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
 
    device = train_cfg.device
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    ptdtype = dtype_map[train_cfg.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" \
        else torch.amp.autocast(device_type="cpu", enabled=False)
 
    loaders = {
        "train": DataLoader(train_cfg.data_path, train_cfg.batch_size,
                            model_cfg.context_length, device)
    }
    if train_cfg.val_data_path and os.path.exists(train_cfg.val_data_path):
        loaders["val"] = DataLoader(train_cfg.val_data_path, train_cfg.batch_size,
                                    model_cfg.context_length, device)
 
    model = GPT(model_cfg).to(device)
    print(f"model parameters: {model.num_parameters():,}")
    optimizer = model.configure_optimizer(
        train_cfg.weight_decay, train_cfg.learning_rate,
        (train_cfg.beta1, train_cfg.beta2), device_type,
    )
 
    start_step = 0
    best_val_loss = float("inf")
    if ckpt_path.exists():
        start_step, best_val_loss = load_checkpoint(ckpt_path, model, optimizer, device)
        start_step += 1
    else:
        print("no checkpoint found — starting fresh")
 
    if train_cfg.compile_model:
        print("compiling model (first step will be slow)...")
        model = torch.compile(model)
 
    model.train()
    start_time = time.time()
    runtime_limit = train_cfg.max_runtime_hours * 3600
    x, y = loaders["train"].get_batch()
 
    for step in range(start_step, train_cfg.max_steps):
        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
 
        t0 = time.time()
        loss_accum = 0.0
        for micro_step in range(train_cfg.grad_accum_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum_steps
            x, y = loaders["train"].get_batch()
            loss.backward()
            loss_accum += loss.item()
 
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dt = time.time() - t0
 
        if step % train_cfg.log_interval == 0:
            tokens_per_step = (train_cfg.batch_size * model_cfg.context_length
                               * train_cfg.grad_accum_steps)
            tok_per_sec = tokens_per_step / dt
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} "
                  f"| {dt*1000:.1f}ms | {tok_per_sec:,.0f} tok/s")
 
        if step > 0 and step % train_cfg.eval_interval == 0:
            losses = estimate_loss(model, loaders, train_cfg.eval_iters, ctx)
            msg = " | ".join(f"{k} {v:.4f}" for k, v in losses.items())
            print(f"  eval @ step {step}: {msg}")
            if "val" in losses and losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(best_path, model, optimizer, step, best_val_loss,
                                model_cfg, train_cfg)
 
        if step > 0 and step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(ckpt_path, model, optimizer, step, best_val_loss,
                            model_cfg, train_cfg)
 
        if time.time() - start_time > runtime_limit:
            print(f"runtime limit ({train_cfg.max_runtime_hours}h) reached — saving and exiting")
            save_checkpoint(ckpt_path, model, optimizer, step, best_val_loss,
                            model_cfg, train_cfg)
            return
 
    save_checkpoint(ckpt_path, model, optimizer, train_cfg.max_steps - 1,
                    best_val_loss, model_cfg, train_cfg)
    print("training complete")

def main() -> None:
    from config_loader import load_config
 
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Path to a YAML run config (e.g., configs/runs/tinystories.yaml)")
    p.add_argument("--out_dir", type=str, default="checkpoints",
                   help="Where to save checkpoints (overrides any config value)")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--no_compile", action="store_true")
    args = p.parse_args()
 
    model_cfg, train_cfg = load_config(args.config, ModelConfig, TrainConfig)
 
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.no_compile:
        train_cfg.compile_model = False
 
    print(f"loaded config from {args.config}")
    print(f"model: {model_cfg}")
    print(f"train: {train_cfg}")
    train(model_cfg, train_cfg, Path(args.out_dir))
 
 
if __name__ == "__main__":
    main()