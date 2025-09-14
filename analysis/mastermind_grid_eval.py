#!/usr/bin/env python3
"""
Evaluate single-turn Mastermind (ig_relative) over a grid of (L, K, history_len)
and save results (CSV/JSON). Optionally plot heatmaps per history level.

Defaults:
- Model: gpt-5-nano
- Grid: L=3..7, K=3..9, history_len=0..5
- N=10, r=1, reward_mode=ig_relative, curriculum=False


Example script
python analysis/mastermind_grid_eval.py -m gpt-4.1-nano -n 2 -r 1 --L-min 3 --L-max 7 --K-min 8 --K-max 8 --H-min 1 --H-max 1 -t 1000 --rel-sh-cap 3000 --rel-cand-cap 500 --save-completions

Requires an OpenAI-compatible endpoint via OPENAI_API_KEY (and optionally OPENAI_BASE_URL).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from typing import Any, Dict, List, Tuple

from openai import OpenAI, AsyncOpenAI
import asyncio


def _load_mm_module():
    """Import the mastermind env module, whether installed or local."""
    # preferred: package import if installed
    try:
        import mastermind as mm  # type: ignore
        if hasattr(mm, "load_environment"):
            return mm
    except Exception:
        pass
    # fallback to local path
    import importlib.util
    here = os.path.dirname(os.path.dirname(__file__))  # environments/mastermind
    mm_path = os.path.join(here, "mastermind.py")
    spec = importlib.util.spec_from_file_location("_mm_mod", mm_path)
    mm = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mm)  # type: ignore
    return mm


async def aevaluate_config(mm, aclient: AsyncOpenAI, model: str, L: int, K: int, H: int, n: int, r: int,
                           toks: int, temp: float, allow_repeats: bool = True, seed: int | None = None) -> Dict[str, Any]:
    env_args: Dict[str, Any] = {
        "mode": "single",
        "L": L,
        "K": K,
        "allow_repeats": allow_repeats,
        "reward_mode": "ig_relative",
        "history_len": H,
        "curriculum": False,
        "max_examples": n,
    }
    if seed is not None:
        env_args["seed"] = seed
    env = mm.load_environment(env_args=env_args)
    results = await env.a_generate(
        inputs=env.get_dataset(n=n),
        client=aclient,
        model=model,
        sampling_args={"max_tokens": toks, "temperature": temp},
        score_rollouts=True,
    )
    rewards = results.reward or []
    ig_rel = results.metrics.get("mastermind_reward", []) or []
    fmt = results.metrics.get("<lambda>", []) or []
    val_mask = [f >= 0.99 for f in fmt]
    ig_valid = [v for v, ok in zip(ig_rel, val_mask) if ok]
    rec = {
        "L": L,
        "K": K,
        "history_len": H,
        "avg_reward": float(statistics.fmean(rewards)) if rewards else 0.0,
        "avg_ig_relative": float(statistics.fmean(ig_rel)) if ig_rel else 0.0,
        "avg_ig_relative_valid": float(statistics.fmean(ig_valid)) if ig_valid else 0.0,
        "valid_rate": float(sum(val_mask)) / float(len(val_mask)) if val_mask else 0.0,
        "avg_format": float(statistics.fmean(fmt)) if fmt else 0.0,
        "n": n,
        "r": r,
    }
    return rec


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grid evaluate Mastermind ig_relative over (L,K,history_len)")
    ap.add_argument("-m", "--model", type=str, default="gpt-4.1-nano")
    ap.add_argument("-n", "--num-examples", type=int, default=10)
    ap.add_argument("-r", "--repeats", type=int, default=1)
    ap.add_argument("-t", "--max-tokens", type=int, default=512)
    ap.add_argument("-T", "--temperature", type=float, default=0.5)
    ap.add_argument("--L-min", type=int, default=3)
    ap.add_argument("--L-max", type=int, default=7)
    ap.add_argument("--K-min", type=int, default=3)
    ap.add_argument("--K-max", type=int, default=9)
    ap.add_argument("--H-min", type=int, default=0)
    ap.add_argument("--H-max", type=int, default=5)
    ap.add_argument("--no-repeats", action="store_true", help="Disallow repeated symbols")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", type=str, default=os.path.join("environments", "mastermind", "analysis", "outputs"))
    ap.add_argument("--save-completions", action="store_true", help="Save raw prompts/completions per config as JSONL")
    ap.add_argument("--completions-dir", type=str, default="", help="Directory to save completions (defaults to out-dir/completions/<stamp>)")
    ap.add_argument("--no-plot", action="store_true", help="Skip plotting even if matplotlib is available")
    ap.add_argument("--rel-sh-cap", type=int, default=5000, help="Cap S_H size for ig_relative scoring")
    ap.add_argument("--rel-cand-cap", type=int, default=1000, help="Cap candidate guesses for ig_relative best search")
    args = ap.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        print("[error] OPENAI_API_KEY not set")
        return 2
    aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

    mm = _load_mm_module()

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out_dir, f"grid_{stamp}.json")
    csv_path = os.path.join(args.out_dir, f"grid_{stamp}.csv")
    if args.save_completions:
        comp_root = args.completions_dir or os.path.join(args.out_dir, "completions", stamp)
        os.makedirs(comp_root, exist_ok=True)

    async def run_grid():
        rows: List[Dict[str, Any]] = []
        for L in range(args.L_min, args.L_max + 1):
            for K in range(args.K_min, args.K_max + 1):
                for H in range(args.H_min, args.H_max + 1):
                    # Evaluate
                    env_args: Dict[str, Any] = {
                        "mode": "single",
                        "L": L,
                        "K": K,
                        "allow_repeats": not args.no_repeats,
                        "reward_mode": "ig_relative",
                        "history_len": H,
                        "curriculum": False,
                        "max_examples": args.num_examples,
                        "relative_sh_cap": args.rel_sh_cap,
                        "relative_candidate_cap": args.rel_cand_cap,
                    }
                    if args.seed is not None:
                        env_args["seed"] = args.seed
                    env = mm.load_environment(env_args=env_args)
                    results = await env.a_generate(
                        inputs=env.get_dataset(n=args.num_examples),
                        client=aclient,
                        model=args.model,
                        sampling_args={"max_tokens": args.max_tokens, "temperature": args.temperature},
                        score_rollouts=True,
                    )
                    # Aggregate
                    rewards = results.reward or []
                    ig_rel = results.metrics.get("mastermind_reward", []) or []
                    fmt = results.metrics.get("<lambda>", []) or []
                    import statistics as _stats
                    rec = {
                        "L": L,
                        "K": K,
                        "history_len": H,
                        "avg_reward": float(_stats.fmean(rewards)) if rewards else 0.0,
                        "avg_ig_relative": float(_stats.fmean(ig_rel)) if ig_rel else 0.0,
                        "avg_format": float(_stats.fmean(fmt)) if fmt else 0.0,
                        "n": args.num_examples,
                        "r": args.repeats,
                    }
                    print(
                        f"L={L} K={K} H={H} | avg_ig_rel={rec['avg_ig_relative']:.4f} avg_reward={rec['avg_reward']:.4f}"
                    )
                    rows.append(rec)

                    # Save raw prompts/completions per-config if requested
                    if args.save_completions:
                        comp_root = args.completions_dir or os.path.join(args.out_dir, "completions", stamp)
                        os.makedirs(comp_root, exist_ok=True)
                        fn = os.path.join(comp_root, f"L{L}_K{K}_H{H}.jsonl")
                        # Metric lists by index
                        mnames = list(results.metrics.keys())
                        with open(fn, "w", encoding="utf-8") as f:
                            for i in range(len(results.completion)):
                                # extract last assistant content if available
                                comp = results.completion[i]
                                last_assist = ""
                                if isinstance(comp, list):
                                    for m in reversed(comp):
                                        if isinstance(m, dict) and m.get("role") == "assistant":
                                            last_assist = str(m.get("content", ""))
                                            break
                                row = {
                                    "L": L,
                                    "K": K,
                                    "history_len": H,
                                    "idx": i,
                                    "reward": float(results.reward[i]) if i < len(results.reward) else None,
                                    "prompt": results.prompt[i],
                                    "completion": comp,
                                    "last_assistant": last_assist,
                                }
                                # attach per-metric values
                                for name in mnames:
                                    vals = results.metrics.get(name) or []
                                    if i < len(vals):
                                        row[name] = float(vals[i])
                                f.write(json.dumps(row) + "\n")
                        print(f"  saved completions: {fn}")
        return rows

    all_rows: List[Dict[str, Any]] = asyncio.run(run_grid())

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "grid": {
                "L": [args.L_min, args.L_max],
                "K": [args.K_min, args.K_max],
                "history_len": [args.H_min, args.H_max],
            },
            "params": {
                "n": args.num_examples,
                "r": args.repeats,
                "max_tokens": args.max_tokens,
                "allow_repeats": not args.no_repeats,
                "seed": args.seed,
            },
            "results": all_rows,
        }, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["L", "K", "history_len", "avg_ig_relative", "avg_ig_relative_valid", "valid_rate", "avg_reward", "avg_format", "n", "r"])
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"Saved CSV: {csv_path}")

    # Optional plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            print("[warn] matplotlib not available; skipping plots")
            return 0
        # For each H, make a heatmap of avg_ig_relative over (L,K)
        for H in range(args.H_min, args.H_max + 1):
            sub = [r for r in all_rows if r["history_len"] == H]
            if not sub:
                continue
            Ls = sorted({r["L"] for r in sub})
            Ks = sorted({r["K"] for r in sub})
            grid = np.zeros((len(Ls), len(Ks)))
            for i, L in enumerate(Ls):
                for j, K in enumerate(Ks):
                    val = next((r["avg_ig_relative"] for r in sub if r["L"] == L and r["K"] == K), 0.0)
                    grid[i, j] = val
            plt.figure(figsize=(8, 5))
            im = plt.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                            extent=[min(Ks)-0.5, max(Ks)+0.5, min(Ls)-0.5, max(Ls)+0.5])
            plt.colorbar(im, label="avg_ig_relative")
            plt.xlabel("K")
            plt.ylabel("L")
            plt.title(f"Mastermind avg_ig_relative — history_len={H}")
            out_png = os.path.join(args.out_dir, f"heatmap_H{H}_{stamp}.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"Saved heatmap: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
