#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple


def _load_mastermind_symbols():
    """
    Robustly import helpers from sibling mastermind.py whether installed as a
    package or run directly from the repository.
    """
    try:
        # package-style
        from .mastermind import (  # type: ignore
            MastermindConfig,
            _render_prompt_text,
            _parse_guess_from_text,
            _compute_reward_for_guess,
            suggest_best_guesses,
            _generate_random_history,
        )
        return {
            "MastermindConfig": MastermindConfig,
            "_render_prompt_text": _render_prompt_text,
            "_parse_guess_from_text": _parse_guess_from_text,
            "_compute_reward_for_guess": _compute_reward_for_guess,
            "suggest_best_guesses": suggest_best_guesses,
            "_generate_random_history": _generate_random_history,
        }
    except Exception:
        # path-style
        import importlib.util

        here = os.path.dirname(__file__)
        mm_path = os.path.join(here, "mastermind.py")
        spec = importlib.util.spec_from_file_location("_mm_mod", mm_path)
        mm = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mm)  # type: ignore
        return {
            "MastermindConfig": mm.MastermindConfig,
            "_render_prompt_text": mm._render_prompt_text,
            "_parse_guess_from_text": mm._parse_guess_from_text,
            "_compute_reward_for_guess": mm._compute_reward_for_guess,
            "suggest_best_guesses": mm.suggest_best_guesses,
            "_generate_random_history": mm._generate_random_history,
        }


def main(argv: List[str] | None = None) -> int:
    syms = _load_mastermind_symbols()
    MastermindConfig = syms["MastermindConfig"]
    _render_prompt_text = syms["_render_prompt_text"]
    _parse_guess_from_text = syms["_parse_guess_from_text"]
    _compute_reward_for_guess = syms["_compute_reward_for_guess"]
    suggest_best_guesses = syms["suggest_best_guesses"]
    _generate_random_history = syms["_generate_random_history"]

    p = argparse.ArgumentParser(
        description="Play single-turn Mastermind locally and see your reward."
    )
    p.add_argument("--L", type=int, default=4, help="Code length")
    p.add_argument("--K", type=int, default=6, help="Alphabet size (0..K-1)")
    p.add_argument(
        "--allow-repeats",
        dest="allow_repeats",
        action="store_true",
        default=True,
        help="Allow repeated symbols (default: true)",
    )
    p.add_argument(
        "--no-repeats",
        dest="allow_repeats",
        action="store_false",
        help="Disallow repeated symbols",
    )
    p.add_argument(
        "--reward-mode",
        choices=["elim", "ig"],
        default="ig",
        help="Reward mode for suggestions/scoring: information gain (bits) or elimination (Gini). Note: env default is ig_relative for single-turn, which ranks equivalent to ig.",
    )
    p.add_argument(
        "--max-space-enum",
        type=int,
        default=200_000,
        help="Enumerate full space up to this size; otherwise sample",
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=10_000,
        help="Sample size when approximating large spaces",
    )
    p.add_argument("--history-len", type=int, default=0, help="Random history length")
    p.add_argument("--suggest-top", type=int, default=0, help="If >0, print top-K suggested guesses before prompting")
    p.add_argument(
        "--candidate-pool",
        choices=["consistent", "all"],
        default="consistent",
        help="Candidate pool for suggestions",
    )
    p.add_argument(
        "--history",
        type=str,
        default="",
        help='JSON list like [{"guess":[0,1,2,3],"feedback":[1,1]}] (overrides --history-len)',
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")

    args = p.parse_args(argv)
    rng = random.Random(args.seed)

    cfg = MastermindConfig(
        L=args.L,
        K=args.K,
        allow_repeats=args.allow_repeats,
        reward_mode=args.reward_mode,
        max_space_enum=args.max_space_enum,
        sample_n=args.sample_n,
        history_len=args.history_len,
    )

    # Build history
    if args.history.strip():
        try:
            obj = json.loads(args.history)
            history = [
                (list(item["guess"]), (int(item["feedback"][0]), int(item["feedback"][1])))
                for item in obj
            ]
        except Exception as e:
            print(f"Failed to parse --history JSON: {e}", file=sys.stderr)
            return 2
    else:
        history = _generate_random_history(cfg, rng)

    # Render prompt and instructions
    prompt = _render_prompt_text(cfg.L, cfg.K, cfg.allow_repeats, history)
    print("\n" + prompt + "\n")
    if args.suggest_top and args.suggest_top > 0:
        try:
            top = suggest_best_guesses(
                cfg,
                history,
                top_k=int(args.suggest_top),
                candidate_pool=str(args.candidate_pool),
                rng=rng,
            )
            if top:
                print("Top suggestions:")
                for i, (g, sc) in enumerate(top, 1):
                    print(f"  {i}. {' '.join(map(str, g))}  score={sc:.4f}")
                print("")
        except Exception as e:
            print(f"[warn] suggestion failed: {e}")
    print("Enter your guess. You can type either:")
    print("  - GUESS: a b c d")
    print("  - a b c d (we will wrap it for you)\n")

    # Read and normalize input
    user = input("> ").strip()
    if not user:
        print("No input provided.")
        return 1
    if not user.upper().startswith("GUESS:"):
        # try to normalize "a b c d"
        toks = [t for t in user.replace(",", " ").split() if t]
        user = "GUESS: " + " ".join(toks)

    ok, guess = _parse_guess_from_text(user, L=cfg.L, K=cfg.K, repeats=cfg.allow_repeats)
    if not ok:
        print("Invalid guess format or values. Expected 'GUESS: a b c d' with a..d in range.")
        return 1

    reward, metrics = _compute_reward_for_guess(guess, cfg, history, rng)
    print("\nResult")
    print("------")
    print(f"Guess: {' '.join(map(str, guess))}")
    print(f"Reward ({cfg.reward_mode}): {reward}")
    print(f"|S_H| (consistent set size): {metrics.get('S_H_size')}")
    print(f"Feedback bucket variants: {metrics.get('bucket_variants')}")
    if metrics.get("used_sampling"):
        print(f"Used sampling with n={cfg.sample_n} (space too large)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
