#!/usr/bin/env python3
"""
Auto-curriculum trainer for the Mastermind single-turn environment.

Strategy:
- Evaluate a batch on the single-step env with reward=ig_relative (closeness to optimal IG).
- Adjust difficulty (L, K, history_len) to steer the average reward toward a target band.
- Repeat for a number of epochs, logging metrics and difficulty per epoch.

Requires an OpenAI-compatible endpoint (OPENAI_API_KEY, optional OPENAI_BASE_URL).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from openai import OpenAI
import verifiers as vf


def _load_mm_module():
    """Import the mastermind env module, whether installed or local."""
    # preferred: package import
    try:
        import mastermind as mm  # type: ignore
        if hasattr(mm, "load_environment"):
            return mm
    except Exception:
        pass

    # fallback: load from local path
    import importlib.util
    here = os.path.dirname(__file__)
    mm_path = os.path.join(here, "mastermind.py")
    spec = importlib.util.spec_from_file_location("_mm_mod", mm_path)
    mm = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mm)  # type: ignore
    return mm


@dataclass
class Difficulty:
    L: int = 4
    K: int = 6
    history_len: int = 0

    def to_env_args(self) -> Dict[str, Any]:
        return {
            "mode": "single",
            "L": self.L,
            "K": self.K,
            "allow_repeats": True,
            "reward_mode": "ig_relative",
            "history_len": self.history_len,
        }


def adjust_difficulty(d: Difficulty, avg_reward: float, low: float, high: float,
                      L_min: int, L_max: int, K_min: int, K_max: int, H_min: int, H_max: int) -> Difficulty:
    """Simple controller to keep avg_reward in [low, high].

    - If reward > high: increase difficulty: L -> K -> history.
    - If reward < low: decrease difficulty: history -> K -> L.
    - Else: keep.
    """
    new = Difficulty(d.L, d.K, d.history_len)
    if avg_reward > high:
        if new.L < L_max:
            new.L += 1
        elif new.K < K_max:
            new.K = min(K_max, new.K + 2)
        elif new.history_len < H_max:
            new.history_len += 1
        return new
    if avg_reward < low:
        if new.history_len > H_min:
            new.history_len -= 1
        elif new.K > K_min:
            new.K = max(K_min, new.K - 2)
        elif new.L > L_min:
            new.L -= 1
        return new
    return new


def evaluate_epoch(mm, model: str, client: OpenAI, d: Difficulty, n: int, r: int, toks: int, temp: float, seed: int | None) -> Tuple[float, Dict[str, float]]:
    env_args = d.to_env_args()
    if seed is not None:
        env_args["seed"] = seed
    env_args["max_examples"] = max(n, 1)
    env = mm.load_environment(env_args=env_args)

    results = env.evaluate(
        client=client,
        model=model,
        num_examples=n,
        rollouts_per_example=r,
        sampling_args={"max_tokens": toks, "temperature": temp},
    )
    rewards = results.reward or []
    avg = float(statistics.fmean(rewards)) if rewards else 0.0

    # Collect a couple of key metrics if present
    metrics = {}
    for key in ("mastermind_reward", "<lambda>", "sh_size_metric"):
        vals = (results.metrics.get(key) or [])
        if vals:
            metrics[f"avg_{key}"] = float(statistics.fmean(vals))
    return avg, metrics


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mastermind auto-curriculum trainer (single-step ig_relative)")
    ap.add_argument("-m", "--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("-e", "--epochs", type=int, default=10)
    ap.add_argument("-n", "--num-examples", type=int, default=20)
    ap.add_argument("-r", "--repeats", type=int, default=1)
    ap.add_argument("-t", "--max-tokens", type=int, default=512)
    ap.add_argument("-T", "--temperature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--low", type=float, default=0.6, help="target lower bound for avg reward")
    ap.add_argument("--high", type=float, default=0.8, help="target upper bound for avg reward")
    ap.add_argument("--L-min", type=int, default=3)
    ap.add_argument("--L-max", type=int, default=8)
    ap.add_argument("--K-min", type=int, default=4)
    ap.add_argument("--K-max", type=int, default=10)
    ap.add_argument("--H-min", type=int, default=1)
    ap.add_argument("--H-max", type=int, default=3)
    ap.add_argument("--log", type=str, default="environments/mastermind/outputs/training/curriculum.jsonl")
    args = ap.parse_args(argv)

    # Model client
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        print("[error] OPENAI_API_KEY not set. Export it or point to a local OpenAI-compatible server.")
        return 2
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load mastermind module
    mm = _load_mm_module()

    # Prepare logging dir
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    log_f = open(args.log, "a", encoding="utf-8")

    d = Difficulty(L=4, K=6, history_len=0)
    for epoch in range(1, args.epochs + 1):
        avg, metrics = evaluate_epoch(mm, args.model, client, d, args.num_examples, args.repeats, args.max_tokens, args.temperature, args.seed)
        rec = {
            "epoch": epoch,
            "L": d.L,
            "K": d.K,
            "history_len": d.history_len,
            "avg_reward": avg,
            **metrics,
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(f"Epoch {epoch:03d} | L={d.L} K={d.K} H={d.history_len} | avg_reward={avg:.4f}")
        # adjust difficulty for next epoch
        d = adjust_difficulty(
            d, avg, args.low, args.high,
            args.L_min, args.L_max, args.K_min, args.K_max, args.H_min, args.H_max,
        )
    log_f.close()
    print(f"Done. Log written to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

