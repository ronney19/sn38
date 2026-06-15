"""
SN38 ChronoGPT — Training Script

Trains a ChronoGPT model for a single year using a pre-built .bin dataset.
Produces config.json + model.safetensors ready to upload to HuggingFace.

IMPORTANT: The validator's chronogpt_model.py uses @torch.inference_mode()
on every forward method, which disables gradients. This script defines a
training-compatible version of the same architecture (identical weights/names,
no inference_mode decorators).

Usage:
    # Train year 2018 with default settings
    python train_chronogpt.py --year 2018

    # Custom model size
    python train_chronogpt.py --year 2018 --num-layers 24 --num-heads 8 --model-dim 512

    # Resume from checkpoint
    python train_chronogpt.py --year 2018 --resume checkpoints/2018/ckpt_step5000.pt

    # Full reference model size (52 layers, ~1.3B params — needs large GPU)
    python train_chronogpt.py --year 2018 --num-layers 52 --num-heads 12 --model-dim 1536

Requirements:
    pip install torch safetensors tiktoken numpy tqdm
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

try:
    from safetensors.torch import save_file as safetensors_save
except ImportError:
    raise SystemExit("Run: pip install safetensors")


# ─────────────────────────────────────────────────────────────────────────────
# Load pretrained weights from a HuggingFace repo (safetensors)
# ─────────────────────────────────────────────────────────────────────────────

def _load_pretrained(model: "ChronoGPT", repo_id: str, revision: str | None, device: str):
    """
    Download and load weights from a HuggingFace model repo into the training model.
    Works with manelalab/chrono-gpt-v1-* models (safetensors branch).

    The validator's load_model() and this function use identical state_dict keys
    so weights transfer without any remapping.
    """
    from safetensors.torch import load_file as safetensors_load
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Run: pip install huggingface_hub")

    print(f"  Downloading pretrained weights from {repo_id} (revision={revision or 'main'})...")
    local_dir = snapshot_download(repo_id=repo_id, revision=revision)
    weights_path = Path(local_dir) / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {local_dir}")

    state_dict = safetensors_load(str(weights_path), device=device)

    # The pretrained config may use different dim — verify compatibility
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  WARNING: missing keys: {missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")

    params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded pretrained weights ({params / 1e9:.2f}B params) ✓")


# ─────────────────────────────────────────────────────────────────────────────
# ChronoGPT — Training-compatible architecture
#
# Identical to sn38/template/chronogpt_model.py EXCEPT:
#   - All @torch.inference_mode() decorators removed (required for gradients)
#   - .bfloat16() casts kept (they work fine with autocast + gradient flow)
#
# Parameter names are identical so saved weights load into the validator's
# inference copy without any key remapping.
# ─────────────────────────────────────────────────────────────────────────────

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x):
        cos = self.cos[None, : x.size(-3), None, :]
        sin = self.sin[None, : x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim)
        self.c_proj = CastedLinear(dim, dim)

    def forward(self, x, ve):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, ve, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), ve)
        x = x + self.mlp(norm(x))
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    def forward(self, inputs):
        base = [emb(inputs).bfloat16() for emb in self.embed]
        L = self.num_layers
        half = L // 2
        encoder = [base[i] if i < 3 else None for i in range(half)]
        decoder = [base[i - (half - 3)] if i >= (half - 3) else None for i in range(half)]
        return encoder + decoder


class ChronoGPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads) for _ in range(num_layers)])
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers=num_layers)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_()
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

    def forward(self, input_ids):
        B = input_ids.size(0)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        x0 = norm(self.embed(input_ids).bfloat16())
        x = x0

        ve = [self.value_embeds(input_ids[i].view(-1)) for i in range(B)]
        ve = [
            torch.stack([ve[b][i] for b in range(B)]) if ve[0][i] is not None else None
            for i in range(len(ve[0]))
        ]
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)
        return logits.float()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

class BinDataset:
    """Memory-mapped loader for .bin token files produced by build_dataset.py."""

    def __init__(self, path: str, block_size: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.n_tokens = len(self.data)
        print(f"  Dataset: {path}")
        print(f"  Tokens : {self.n_tokens:,}")
        print(f"  Blocks : {(self.n_tokens - 1) // block_size:,} (block_size={block_size})")

    def __len__(self):
        return (self.n_tokens - 1) // self.block_size

    def get_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (input, target) pairs."""
        ix = torch.randint(self.n_tokens - self.block_size, (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1 : i + self.block_size + 1].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule — cosine decay with linear warmup
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Export to HuggingFace format
# ─────────────────────────────────────────────────────────────────────────────

def export_model(model: ChronoGPT, out_dir: Path, year: int, cfg: dict):
    """
    Save model in the format expected by the validator's load_model():
      config.json         — architecture hyperparameters
      model.safetensors   — weights (no pickle, safetensors format)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # config.json — exact keys required by chronogpt_model.py:load_model
    config = {
        "vocab_size": cfg["vocab_size"],
        "num_layers": cfg["num_layers"],
        "num_heads":  cfg["num_heads"],
        "model_dim":  cfg["model_dim"],
        "year":       year,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # model.safetensors — CPU float32 state dict
    state_dict = {k: v.cpu().float() for k, v in model.state_dict().items()}
    safetensors_save(state_dict, str(out_dir / "model.safetensors"))

    params = model.num_params()
    print(f"  Exported to {out_dir}")
    print(f"  Params: {params / 1e6:.1f}M  ({params / 1e9:.3f}B)")
    print(f"  config.json + model.safetensors ready for HuggingFace upload")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")
    use_amp = device == "cuda"

    # ── model config ─────────────────────────────────────────────────────────
    cfg = {
        "vocab_size": 50304,
        "num_layers": args.num_layers,
        "num_heads":  args.num_heads,
        "model_dim":  args.model_dim,
    }

    # When fine-tuning a pretrained model, override arch args with the
    # pretrained model's config to ensure compatibility
    if args.pretrained and not args.resume:
        try:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(
                args.pretrained, "config.json", revision=args.pretrained_revision
            )
            remote_cfg = json.load(open(cfg_path))
            cfg["num_layers"] = remote_cfg["num_layers"]
            cfg["num_heads"]  = remote_cfg["num_heads"]
            cfg["model_dim"]  = remote_cfg["model_dim"]
            print(f"  Using pretrained model config: {cfg}")
        except Exception as e:
            print(f"  Could not fetch pretrained config ({e}), using CLI args")

    print(f"\nModel config: {cfg}")

    # ── dataset ──────────────────────────────────────────────────────────────
    data_path = Path(args.data_dir) / str(args.year) / "train.bin"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Run: python build_dataset.py --year {args.year}"
        )
    dataset = BinDataset(str(data_path), args.block_size)

    # ── model ────────────────────────────────────────────────────────────────
    model = ChronoGPT(**cfg).to(device)
    print(f"Parameters: {model.num_params() / 1e6:.1f}M")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}")
    elif args.pretrained:
        _load_pretrained(model, args.pretrained, args.pretrained_revision, device)
        start_step = 0
    else:
        start_step = 0

    # ── optimizer ────────────────────────────────────────────────────────────
    # Separate weight decay from bias/norm/embedding params
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2 and "embed" not in n]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and (p.dim() < 2 or "embed" in n)]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device == "cuda",
    )
    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer"])

    scaler = GradScaler(device, enabled=use_amp)

    # ── checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir) / str(args.year)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining year={args.year} for {args.max_steps} steps")
    print(f"  block_size={args.block_size}  batch_size={args.batch_size}"
          f"  grad_accum={args.grad_accum_steps}")
    print(f"  effective batch = {args.batch_size * args.grad_accum_steps * args.block_size:,} tokens/step")
    print(f"  checkpoints → {ckpt_dir}")
    print(f"  export      → {Path(args.export_dir) / str(args.year)}\n")

    model.train()
    t0 = time.time()
    tokens_processed = 0
    loss_accum = 0.0

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(args.grad_accum_steps):
            x, y = dataset.get_batch(args.batch_size, device)
            with autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                # logits: (B, T, vocab_size) — compute cross-entropy over T dimension
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()
            tokens_processed += x.numel()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ── logging ──────────────────────────────────────────────────────────
        if step % args.log_interval == 0:
            dt = time.time() - t0
            tok_per_sec = tokens_processed / dt if dt > 0 else 0
            print(
                f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} "
                f"| grad_norm {grad_norm:.3f} | {tok_per_sec/1e3:.1f}k tok/s"
            )

        # ── checkpoint ───────────────────────────────────────────────────────
        if step > 0 and step % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"ckpt_step{step}.pt"
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config":    cfg,
                "year":      args.year,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # ── intermediate export ───────────────────────────────────────────────
        if step > 0 and step % args.export_interval == 0:
            model.eval()
            export_dir = Path(args.export_dir) / str(args.year) / f"step{step}"
            export_model(model, export_dir, args.year, cfg)
            model.train()

    # ── final export ──────────────────────────────────────────────────────────
    model.eval()
    final_dir = Path(args.export_dir) / str(args.year)
    export_model(model, final_dir, args.year, cfg)

    total_time = time.time() - t0
    print(f"\nDone. Total time: {total_time/3600:.2f}h | {tokens_processed/1e9:.2f}B tokens")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ChronoGPT for SN38")

    # Required
    parser.add_argument("--year", type=int, required=True, choices=range(2013, 2025),
                        help="Which year to train (must have data/{year}/train.bin)")

    # Model architecture
    arch = parser.add_argument_group("Model architecture")
    arch.add_argument("--num-layers", type=int, default=12,
                      help="Number of transformer layers (default: 12 — ~125M params at dim=768)")
    arch.add_argument("--num-heads",  type=int, default=12,
                      help="Number of attention heads (default: 12)")
    arch.add_argument("--model-dim",  type=int, default=768,
                      help="Model dimension (default: 768)")

    # Training hyperparameters
    hp = parser.add_argument_group("Training hyperparameters")
    hp.add_argument("--block-size",       type=int,   default=512,    help="Context length (default: 512)")
    hp.add_argument("--batch-size",       type=int,   default=8,      help="Batch size per grad step (default: 8)")
    hp.add_argument("--grad-accum-steps", type=int,   default=4,      help="Gradient accumulation steps (default: 4)")
    hp.add_argument("--max-steps",        type=int,   default=10_000, help="Total training steps (default: 10000)")
    hp.add_argument("--warmup-steps",     type=int,   default=500,    help="LR warmup steps (default: 500)")
    hp.add_argument("--max-lr",           type=float, default=3e-4,   help="Peak learning rate (default: 3e-4)")
    hp.add_argument("--min-lr",           type=float, default=3e-5,   help="Final learning rate (default: 3e-5)")
    hp.add_argument("--weight-decay",     type=float, default=0.1,    help="AdamW weight decay (default: 0.1)")
    hp.add_argument("--grad-clip",        type=float, default=1.0,    help="Gradient clip norm (default: 1.0)")

    # I/O
    io = parser.add_argument_group("I/O")
    io.add_argument("--data-dir",           type=str, default="data",        help="Dataset root (default: data/)")
    io.add_argument("--checkpoint-dir",     type=str, default="checkpoints", help="Checkpoint root (default: checkpoints/)")
    io.add_argument("--export-dir",         type=str, default="models",      help="Export root (default: models/)")
    io.add_argument("--resume",             type=str, default=None,          help="Resume from checkpoint .pt file")
    io.add_argument("--pretrained",         type=str, default=None,
                    help="HuggingFace repo ID to fine-tune from (e.g. manelalab/chrono-gpt-v1-20181231)")
    io.add_argument("--pretrained-revision",type=str, default="safetensors",
                    help="HF repo revision/branch for --pretrained (default: safetensors)")
    io.add_argument("--log-interval",       type=int, default=50,            help="Log every N steps (default: 50)")
    io.add_argument("--save-interval",      type=int, default=1000,          help="Save checkpoint every N steps (default: 1000)")
    io.add_argument("--export-interval",    type=int, default=5000,          help="Export HF model every N steps (default: 5000)")

    args = parser.parse_args()

    # Validate num_heads divides model_dim
    if args.model_dim % args.num_heads != 0:
        raise ValueError(f"model_dim ({args.model_dim}) must be divisible by num_heads ({args.num_heads})")
    if args.num_layers % 2 != 0:
        raise ValueError(f"num_layers ({args.num_layers}) must be even (encoder-decoder split)")

    train(args)


if __name__ == "__main__":
    main()
