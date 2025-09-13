import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import verifiers as vf

try:
    # Optional convenience for dataset creation
    from datasets import Dataset as HFDataset
except Exception:  # pragma: no cover
    HFDataset = None  # type: ignore


Feedback = Tuple[int, int]  # (bulls, cows)


# -------------------------------
# Core Mastermind functionality
# -------------------------------

def _feedback(code: Sequence[int], guess: Sequence[int]) -> Feedback:
    L = len(code)
    assert len(guess) == L
    bulls = sum(1 for i in range(L) if code[i] == guess[i])
    c_count = Counter(code)
    g_count = Counter(guess)
    total_overlap = sum(min(c_count[s], g_count[s]) for s in set(c_count) | set(g_count))
    cows = total_overlap - bulls
    return bulls, cows


def _all_codes_enumerate(L: int, K: int, allow_repeats: bool) -> List[Tuple[int, ...]]:
    # Enumerate all K^L codes (as tuples). Filter by repeats if needed.
    # For performance and memory, keep tuples instead of lists.
    if allow_repeats:
        # Base-K counting
        out: List[Tuple[int, ...]] = []
        current = [0] * L
        while True:
            out.append(tuple(current))
            # increment base-K
            i = L - 1
            while i >= 0:
                current[i] += 1
                if current[i] < K:
                    break
                current[i] = 0
                i -= 1
            if i < 0:
                break
        return out
    else:
        # No repeats: generate permutations with replacement constraint
        # Use iterative generation of combinations without repetition (order matters)
        # For small L, K, this is fine.
        out: List[Tuple[int, ...]] = []
        def backtrack(build: List[int], used: List[bool]):
            if len(build) == L:
                out.append(tuple(build))
                return
            for s in range(K):
                if not used[s]:
                    used[s] = True
                    build.append(s)
                    backtrack(build, used)
                    build.pop()
                    used[s] = False
        backtrack([], [False] * K)
        return out


def _sample_codes(L: int, K: int, allow_repeats: bool, n: int, rng: random.Random) -> List[Tuple[int, ...]]:
    # Draw n codes uniformly (by simple independent draws with or without repeats)
    # Ensure uniqueness to reduce duplicates in small spaces.
    seen = set()
    out: List[Tuple[int, ...]] = []
    attempts = 0
    max_attempts = max(10_000, 10 * n)
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        if allow_repeats:
            code = tuple(rng.randrange(K) for _ in range(L))
        else:
            if K < L:
                break  # impossible
            symbols = list(range(K))
            rng.shuffle(symbols)
            code = tuple(symbols[:L])
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def _filter_by_history(codes: Iterable[Tuple[int, ...]], history: List[Tuple[List[int], Feedback]]):
    # Keep codes consistent with all given (guess, feedback) pairs
    out = []
    for c in codes:
        ok = True
        for g, y in history:
            if _feedback(c, g) != tuple(y):
                ok = False
                break
        if ok:
            out.append(c)
    return out


def _partition_counts(consistent_codes: Iterable[Tuple[int, ...]], guess: Sequence[int]) -> Dict[Feedback, int]:
    counts: Dict[Feedback, int] = defaultdict(int)
    for code in consistent_codes:
        y = _feedback(code, guess)
        counts[y] += 1
    return counts


def _reward_from_counts(counts: Dict[Feedback, int], total: int, mode: str = "elim", base2: bool = True) -> float:
    if total <= 0:
        return 0.0
    ps = [c / total for c in counts.values()]
    if mode == "elim":
        return 1.0 - sum(p * p for p in ps)
    elif mode == "ig":
        logf = math.log2 if base2 else math.log
        H = -sum(p * (logf(p) if p > 0 else 0.0) for p in ps)
        return H
    else:
        raise ValueError(f"Unknown reward_mode: {mode}")


# -------------------------------
# Prompt and parsing helpers
# -------------------------------

def _render_prompt_text(L: int, K: int, repeats: bool, history: List[Tuple[List[int], Feedback]]) -> str:
    lines = [
        "TASK: Mastermind (single-turn)",
        f"Alphabet: 0..{K-1} (K={K}), Code length: {L}",
        f"History count: {len(history)}",
    ]
    if history:
        lines.append("History:")
        for g, (b, w) in history:
            g_str = " ".join(str(x) for x in g)
            lines.append(f"- Guess {g_str} -> feedback b={b}, w={w}")
    lines.append(f"CONSTRAINTS: repeats {'allowed' if repeats else 'not allowed'}")
    lines.append("ACTION FORMAT: Put the guess inside <answer>...</answer> exactly as 'GUESS: a b c d'.")
    return "\n".join(lines)


_JSON_GUESS_RE = re.compile(r"\{\s*\"guess\"\s*:\s*\[(.*?)\]\s*\}\s*$", re.IGNORECASE)


def _parse_guess_from_text(text: str, L: int, K: int, repeats: bool) -> Tuple[bool, List[int]]:
    """Return (is_valid, guess).

    Accepts either:
      - GUESS: a b c d
      - {"guess":[a,b,c,d]}
    """
    t = text.strip()
    guess: List[int] = []
    ok = False
    if t.upper().startswith("GUESS:"):
        parts = t.split(":", 1)[1].strip().replace(",", " ")
        toks = [p for p in parts.split() if p]
        try:
            guess = [int(x) for x in toks]
            ok = True
        except Exception:
            ok = False
    else:
        m = _JSON_GUESS_RE.match(t)
        if m:
            inner = m.group(1)
            vals = [v.strip() for v in inner.split(";")]
            if len(vals) == 1:
                vals = [v.strip() for v in inner.split(",")]
            try:
                guess = [int(x) for x in vals if x]
                ok = True
            except Exception:
                ok = False

    if not ok:
        return False, []
    if len(guess) != L:
        return False, []
    if any(x < 0 or x >= K for x in guess):
        return False, []
    if not repeats and len(set(guess)) != len(guess):
        return False, []
    return True, guess


# -------------------------------
# Dataset generation (random histories)
# -------------------------------

@dataclass
class MastermindConfig:
    L: int = 4
    K: int = 6
    allow_repeats: bool = True
    reward_mode: str = "ig"  # "elim" or "ig"
    max_space_enum: int = 200_000
    sample_n: int = 10_000  # used when space is too large
    history_len: int = 0
    obs_format: str = "text"  # reserved
    seed: int | None = None


def _generate_random_history(cfg: MastermindConfig, rng: random.Random) -> List[Tuple[List[int], Feedback]]:
    # Sample hidden code, then draw history_len random guesses and record true feedbacks
    # Guesses may repeat symbols as per cfg.allow_repeats (even if code disallows repeats)
    # Ensure guesses are valid per constraints
    if cfg.allow_repeats:
        hidden = [rng.randrange(cfg.K) for _ in range(cfg.L)]
    else:
        symbols = list(range(cfg.K))
        rng.shuffle(symbols)
        hidden = symbols[: cfg.L]
    history: List[Tuple[List[int], Feedback]] = []
    seen = set()
    for _ in range(max(0, int(cfg.history_len))):
        # sample a valid guess (not necessarily distinct from hidden)
        attempt = 0
        g: List[int]
        while True:
            attempt += 1
            if cfg.allow_repeats:
                g = [rng.randrange(cfg.K) for _ in range(cfg.L)]
            else:
                if cfg.K < cfg.L:
                    break
                syms = list(range(cfg.K))
                rng.shuffle(syms)
                g = syms[: cfg.L]
            t = tuple(g)
            if t in seen and attempt < 20:
                continue
            seen.add(t)
            break
        fb = _feedback(hidden, g)
        history.append((g, fb))
    return history


# -------------------------------
# Reward functions
# -------------------------------

def _compute_reward_for_guess(
    guess: List[int],
    cfg: MastermindConfig,
    history: List[Tuple[List[int], Feedback]],
    rng: random.Random,
) -> Tuple[float, Dict[str, Any]]:
    # Build code universe (enumerate or sample), filter by history, partition by feedback
    space_size = cfg.K ** cfg.L
    used_sampling = False
    if space_size <= cfg.max_space_enum:
        universe = _all_codes_enumerate(cfg.L, cfg.K, cfg.allow_repeats)
    else:
        used_sampling = True
        n = max(1000, int(cfg.sample_n))
        universe = _sample_codes(cfg.L, cfg.K, cfg.allow_repeats, n=n, rng=rng)
    consistent = _filter_by_history(universe, history)
    total = len(consistent)
    counts = _partition_counts(consistent, guess) if total > 0 else {}
    reward = _reward_from_counts(counts, total, mode=cfg.reward_mode)
    metrics = {
        "S_H_size": total,
        "used_sampling": used_sampling,
        "bucket_variants": len(counts),
    }
    return reward, metrics


# -------------------------------
# Environment factory
# -------------------------------


def load_environment(**kwargs) -> vf.Environment:
    """
    Mastermind-IG (single-turn) environment for Verifiers.

    Env args (via -a/--env-args JSON):
      - L: int = 4
      - K: int = 6
      - allow_repeats: bool = True
      - reward_mode: str = "ig" | "elim"  # default ig
      - max_space_enum: int = 200000
      - sample_n: int = 10000
      - history_len: int = 0
      - max_examples: int = 200
      - seed: int | None
      - use_think: bool = True (formatting only)
      - mode: str = "single" | "solve"  # single-turn scoring vs. multi-turn solve-to-completion
      - solve_max_turns: int = 12        # cap for multi-turn mode
      - fixed_history: list[[guess:list[int], feedback:[b,w]]] | None
    """
    # Support being called with either env_args dict or flattened kwargs
    env_args: Dict[str, Any] = (kwargs.get("env_args") if "env_args" in kwargs else kwargs) or {}

    cfg = MastermindConfig(
        L=int(env_args.get("L", 4)),
        K=int(env_args.get("K", 6)),
        allow_repeats=bool(env_args.get("allow_repeats", True)),
        reward_mode=str(env_args.get("reward_mode", "ig")),
        max_space_enum=int(env_args.get("max_space_enum", 200_000)),
        sample_n=int(env_args.get("sample_n", 10_000)),
        history_len=int(env_args.get("history_len", 0)),
        obs_format=str(env_args.get("obs_format", "text")),
        seed=env_args.get("seed", None),
    )

    max_examples: int = int(env_args.get("max_examples", 200))
    use_think: bool = bool(env_args.get("use_think", True))
    mode: str = str(env_args.get("mode", "solve")).lower()
    solve_max_turns: int = int(env_args.get("solve_max_turns", 12))
    rng = random.Random(cfg.seed)

    # Build dataset examples
    examples: List[Dict[str, Any]] = []
    fixed_history = env_args.get("history") or env_args.get("fixed_history")
    if mode == "single":
        if fixed_history is not None:
            # accept a single provided history and use it for all examples
            # normalize: list of {"guess":[...], "feedback":[b,w]}
            H: List[Tuple[List[int], Feedback]] = []
            for item in fixed_history:
                g = list(item["guess"]) if isinstance(item, dict) else list(item[0])
                fb = tuple(item["feedback"]) if isinstance(item, dict) else tuple(item[1])
                H.append((g, (int(fb[0]), int(fb[1]))))
            prompt_text = _render_prompt_text(cfg.L, cfg.K, cfg.allow_repeats, H)
            example_info = {
                "L": cfg.L,
                "K": cfg.K,
                "allow_repeats": cfg.allow_repeats,
                "reward_mode": cfg.reward_mode,
                "max_space_enum": cfg.max_space_enum,
                "sample_n": cfg.sample_n,
                "history": H,
            }
            for _ in range(max_examples if max_examples > 0 else 1):
                examples.append({
                    "question": prompt_text,
                    "answer": "",
                    "info": example_info,
                })
        else:
            total = max_examples if max_examples and max_examples > 0 else 200
            for _ in range(total):
                H = _generate_random_history(cfg, rng)
                prompt_text = _render_prompt_text(cfg.L, cfg.K, cfg.allow_repeats, H)
                examples.append({
                    "question": prompt_text,
                    "answer": "",
                    "info": {
                        "L": cfg.L,
                        "K": cfg.K,
                        "allow_repeats": cfg.allow_repeats,
                        "reward_mode": cfg.reward_mode,
                        "max_space_enum": cfg.max_space_enum,
                        "sample_n": cfg.sample_n,
                        "history": H,
                    },
                })
    else:  # mode == "solve": multi-turn solve-to-completion dataset
        total = max_examples if max_examples and max_examples > 0 else 200
        for i in range(total):
            # sample hidden code; store in info, not in the prompt
            if cfg.allow_repeats:
                hidden = [rng.randrange(cfg.K) for _ in range(cfg.L)]
            else:
                if cfg.K < cfg.L:
                    raise ValueError("K must be >= L when allow_repeats is False")
                syms = list(range(cfg.K))
                rng.shuffle(syms)
                hidden = syms[: cfg.L]
            H: List[Tuple[List[int], Feedback]] = []
            prompt_text = _render_prompt_text(cfg.L, cfg.K, cfg.allow_repeats, H)
            examples.append({
                "question": "TASK: Mastermind Solve (multi-turn)\n" + prompt_text,
                "answer": "",
                "info": {
                    "L": cfg.L,
                    "K": cfg.K,
                    "allow_repeats": cfg.allow_repeats,
                    "hidden_code": hidden,
                    "max_turns": solve_max_turns,
                },
            })

    if HFDataset is not None:
        dataset = HFDataset.from_list(examples)
    else:
        # If datasets is unavailable, verifiers supports HF-style datasets;
        # to keep compatibility, construct a minimal shim using verifiers' GenerateInputs later.
        # However, most installations include datasets. We'll still pass the list here.
        dataset = examples  # type: ignore

    # Parser: require <answer>GUESS: a b c d</answer>
    parser = vf.XMLParser(fields=["answer"], answer_field="answer")

    # System prompt
    if mode == "single":
        base_sys = (
            "You are playing Mastermind. Follow the exact answer format.\n\n"
            "Return only the guess inside <answer>...</answer> tags."
        )
        system_prompt = base_sys if not use_think else (
            "You are playing Mastermind. Think step-by-step, then answer.\n\n"
            "Put your reasoning in <think>...</think> and only the guess inside <answer>...</answer>."
        )
    else:
        base_sys = (
            "You are playing Mastermind. Solve the hidden code within the turn limit.\n"
            "After each of your guesses, you'll receive feedback b (bulls) and w (cows).\n\n"
            "Reply each turn ONLY with the guess inside <answer>...</answer> as 'GUESS: a b c d'."
        )
        system_prompt = base_sys if not use_think else (
            "You are playing Mastermind. Think step-by-step, then answer.\n"
            "After each guess you'll receive feedback, keep guessing until solved.\n\n"
            "Put your reasoning in <think>...</think> and only the next guess inside <answer>...</answer>."
        )

    if mode == "single":
        # Main reward: elimination fraction or IG depending on info[reward_mode]
        async def mastermind_reward(parser, completion, info: Dict[str, Any], **_):  # type: ignore
            # Extract raw answer text
            try:
                ans_text = (parser.parse_answer(completion) or "").strip()
            except Exception:
                ans_text = ""
            L = int(info.get("L", 4))
            K = int(info.get("K", 6))
            repeats = bool(info.get("allow_repeats", True))
            reward_mode = str(info.get("reward_mode", "elim"))
            max_space_enum = int(info.get("max_space_enum", 200_000))
            sample_n = int(info.get("sample_n", 10_000))
            history_raw = info.get("history", [])
            history = [
                (list(item[0]) if not isinstance(item, dict) else list(item["guess"]),
                 (int(item[1][0]) if not isinstance(item, dict) else int(item["feedback"][0]),
                  int(item[1][1]) if not isinstance(item, dict) else int(item["feedback"][1])))
                for item in history_raw
            ]

            ok, guess = _parse_guess_from_text(ans_text, L=L, K=K, repeats=repeats)
            if not ok:
                return -1.0  # punish malformed or invalid guesses

            cfg_local = MastermindConfig(
                L=L,
                K=K,
                allow_repeats=repeats,
                reward_mode=reward_mode,
                max_space_enum=max_space_enum,
                sample_n=sample_n,
                history_len=0,
            )
            reward, _ = _compute_reward_for_guess(guess, cfg_local, history, rng=random.Random(42))
            return float(reward)

        rubric = vf.Rubric(parser=parser)
        rubric.add_reward_func(mastermind_reward, weight=1.0)

        # Strict format reward (light weight)
        format_reward_fn = parser.get_format_reward_func()
        rubric.add_reward_func(lambda completion, **__: float(format_reward_fn(completion)), weight=0.05)  # type: ignore

        # Expose size of S_H as a diagnostic metric
        async def sh_size_metric(parser, completion, info: Dict[str, Any], **_):  # type: ignore
            try:
                ans_text = (parser.parse_answer(completion) or "").strip()
            except Exception:
                ans_text = ""
            L = int(info.get("L", 4))
            K = int(info.get("K", 6))
            repeats = bool(info.get("allow_repeats", True))
            max_space_enum = int(info.get("max_space_enum", 200_000))
            sample_n = int(info.get("sample_n", 10_000))
            history_raw = info.get("history", [])
            history = [
                (list(item[0]) if not isinstance(item, dict) else list(item["guess"]),
                 (int(item[1][0]) if not isinstance(item, dict) else int(item["feedback"][0]),
                  int(item[1][1]) if not isinstance(item, dict) else int(item["feedback"][1])))
                for item in history_raw
            ]
            ok, guess = _parse_guess_from_text(ans_text, L=L, K=K, repeats=repeats)
            if not ok:
                return 0.0
            cfg_local = MastermindConfig(
                L=L,
                K=K,
                allow_repeats=repeats,
                reward_mode="elim",
                max_space_enum=max_space_enum,
                sample_n=sample_n,
            )
            # compute metrics only
            space_size = K ** L
            if space_size <= cfg_local.max_space_enum:
                universe = _all_codes_enumerate(L, K, repeats)
            else:
                universe = _sample_codes(L, K, repeats, n=max(1000, int(cfg_local.sample_n)), rng=random.Random(123))
            consistent = _filter_by_history(universe, history)
            return float(len(consistent))

        rubric.add_reward_func(sh_size_metric, weight=0.0)

        env = vf.SingleTurnEnv(
            dataset=dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            message_type="chat",
        )
        return env

    # ----------------------
    # Multi-turn solve mode
    # ----------------------

    class MastermindSolveEnv(vf.MultiTurnEnv):
        async def is_completed(self, messages, state, **kwargs) -> bool:  # type: ignore
            return bool(state.get("done", False))

        async def setup_state(self, state, **kwargs):  # type: ignore
            info = state.get("info", {}) or {}
            L = int(info.get("L", 4))
            K = int(info.get("K", 6))
            repeats = bool(info.get("allow_repeats", True))
            hidden = list(info.get("hidden_code", []))
            if not hidden:
                # Fallback if not provided
                rr = random.Random(info.get("seed", None))
                if repeats:
                    hidden = [rr.randrange(K) for _ in range(L)]
                else:
                    syms = list(range(K))
                    rr.shuffle(syms)
                    hidden = syms[:L]
            state["L"] = L
            state["K"] = K
            state["repeats"] = repeats
            state["hidden"] = hidden
            state["history"] = []  # list of (guess, (b,w))
            state["valid_guesses"] = 0
            state["done"] = False
            state["solved"] = False
            state["turns_to_solve"] = 0
            state["max_turns_cap"] = int(info.get("max_turns", solve_max_turns))
            return state

        async def env_response(self, messages, state, **kwargs):  # type: ignore
            parser: vf.Parser = self.parser
            L: int = state["L"]
            K: int = state["K"]
            repeats: bool = state["repeats"]
            hidden: List[int] = state["hidden"]
            # get last assistant content
            last_assistant = [m for m in messages if m.get("role") == "assistant"][-1]
            try:
                ans_text = (parser.parse_answer(messages) or "").strip()
            except Exception:
                ans_text = ""
            ok, guess = _parse_guess_from_text(ans_text, L=L, K=K, repeats=repeats)
            if not ok:
                msg = (
                    "Invalid guess format. Reply exactly as <answer>GUESS: a b c d</answer> "
                    f"with a..d in 0..{K-1} and length {L}."
                )
                return [{"role": "user", "content": msg}], state
            # compute feedback
            b, w = _feedback(hidden, guess)
            state["valid_guesses"] = int(state.get("valid_guesses", 0)) + 1
            state["history"].append((list(guess), (b, w)))
            if b == L:
                state["solved"] = True
                state["done"] = True
                state["turns_to_solve"] = int(state.get("valid_guesses", 0))
                msg = (
                    f"Feedback: b={b}, w={w}.\n"
                    f"Solved in {state['turns_to_solve']} valid guesses."
                )
                return [{"role": "user", "content": msg}], state
            # not solved: check cap
            if state.get("valid_guesses", 0) >= int(state.get("max_turns_cap", solve_max_turns)):
                state["done"] = True
                msg = (
                    f"Feedback: b={b}, w={w}.\n"
                    f"Turn limit reached ({state['max_turns_cap']})."
                )
                return [{"role": "user", "content": msg}], state
            # continue: send updated history prompt
            hist: List[Tuple[List[int], Feedback]] = state["history"]
            prompt_text = _render_prompt_text(L, K, repeats, hist)
            msg = (
                f"Feedback: b={b}, w={w}. Keep going.\n\n" + prompt_text
            )
            return [{"role": "user", "content": msg}], state

    # Rubric for solve mode
    rubric = vf.Rubric(parser=parser)

    def speed_reward(state, **_):  # type: ignore
        solved = bool(state.get("solved", False))
        if not solved:
            return 0.0
        turns = int(state.get("turns_to_solve", 0))
        cap = int(state.get("max_turns_cap", solve_max_turns))
        # normalized speed in (0,1], faster → higher
        return max(0.0, 1.0 - float(turns - 1) / max(1, cap))

    def solved_metric(state, **_):  # type: ignore
        return 1.0 if bool(state.get("solved", False)) else 0.0

    def turns_metric(state, **_):  # type: ignore
        return float(state.get("turns_to_solve", 0))

    rubric.add_reward_func(speed_reward, weight=1.0)
    rubric.add_reward_func(solved_metric, weight=0.0)
    rubric.add_reward_func(turns_metric, weight=0.0)

    # Format reward (ensure model sends proper tags each turn)
    format_reward_fn = parser.get_format_reward_func()
    rubric.add_reward_func(lambda completion, **__: float(format_reward_fn(completion)), weight=0.05)  # type: ignore

    env = MastermindSolveEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_turns=solve_max_turns,
        message_type="chat",
    )
    return env
