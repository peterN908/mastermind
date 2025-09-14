# Mastermind

A mastermind game simulator. Enables evaluating models in a multi-step (full playthrough), and single-step (making a single move) environment. 

### The concept

The critical idea that I want to explore is to have an environment with arbitrary complexity and scale the complexity in response to the reward the model accumulates so as to optimise training.

Adjusting the Code length (L), Alphabet size (K) and history allows us to tune the complexity of the single-step environment. 

Different setups are available but the main idea is that we reward the model based on how close to the "perfect" decision it makes (based on information gain) in a single-step environment.

Finally we evaluate on full multi-step playthroughs. 


 
### Overview
- **Environment ID**: `mastermind`
- **Short description**: Single‑turn and multi‑turn Mastermind. Single‑turn rewards closeness to the best information‑gain move; multi‑turn evaluates speed to solve.
- **Tags**: mastermind, games, single-turn, multi-turn, eval, rl

### Datasets
- **Primary dataset**: Programmatically generated prompts with optional prior history (guess/feedback pairs). Histories are derived from a hidden code to ensure consistency.
- **Size**: Controlled by `max_examples` (defaults to 200 if unspecified). Randomness is seeded via `seed`.

### Task
- **Types**: single‑turn (chat) and multi‑turn solve (chat)
- **Parser**: `XMLParser` expecting `<answer>GUESS: a b c d</answer>`
- **Rubric overview**:
  - Single‑turn: ig_relative (default) or ig/elim; plus a small format bonus
  - Solve: speed to solve (normalized), plus solved/turns metrics

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval mastermind
```

Configure model and sampling (single‑turn with built‑in curriculum):

```bash
uv run vf-eval mastermind \
  -m gpt-4.1-nano \
  -n 4 -r 1 -t 512 -T 0.5 -s \
  -a '{"mode":"single", "L":4, "K":6, "history_len":2, "allow_repeats":true, "seed":99, "curriculum": false}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The model must reply with the guess inside `<answer>...</answer>` as `GUESS: a b c d`.

### Multi‑turn Solve Mode (default)
Run a full Mastermind game until solved (or turn cap):

```bash
uv run vf-eval mastermind -a '{"mode":"solve", "L":4, "K":6, "allow_repeats":true, "solve_max_turns":12}'
```

Metrics in solve mode:
- `reward` (speed): normalized in (0,1] when solved (faster → higher), else 0.
- `solved_metric`: 1 if solved, 0 otherwise.
- `turns_metric`: number of valid guesses taken to solve.

Notes:
- Default mode is `solve`. For single-turn scoring, use `-a '{"mode":"single", ...}'`.
- Default single-turn reward is ig_relative (closeness to best IG). Use `-a '{"reward_mode":"ig"}'` for raw IG.

### Built‑in Auto‑Curriculum (flag)
Enable persistent difficulty adaptation in single‑turn mode:

```bash
uv run vf-eval mastermind \
  -m gpt-4.1-mini \
  -n 20 -r 1 -t 512 -T 0.5 -s \
  -a '{"mode":"single", "curriculum": {"low": 0.6, "high": 0.8}}'
```

Details:
- Keeps the average reward in `[low, high]` by adjusting `(L, K, history_len)` across runs.
- Logs to `environments/mastermind/outputs/training/curriculum.jsonl` and reads the last run to pick the next difficulty.
- You can set bounds and log path: `{L_min,L_max,K_min,K_max,H_min,H_max,log_path}`.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `L` | int | `4` | Code length |
| `K` | int | `6` | Alphabet size (symbols `0..K-1`) |
| `allow_repeats` | bool | `true` | Allow repeated symbols in codes/guesses |
| `reward_mode` | str | `"ig_relative"` | `"ig_relative"` (ratio to best IG), `"ig"` (information gain, bits), or `"elim"` (normalized elimination) |
| `max_space_enum` | int | `200000` | Enumerate all codes if `K^L <= max_space_enum`, else sample |
| `sample_n` | int | `10000` | Sample size when approximating large code spaces |
| `history_len` | int | `0` | Number of random (guess, feedback) items to include in the prompt |
| `history` | list | `null` | If provided, a fixed history overrides `history_len` (list of `{guess:[...], feedback:[b,w]}`) |
| `max_examples` | int | `200` | Number of examples to generate |
| `seed` | int | `null` | Seed for reproducibility |
| `use_think` | bool | `true` | If true, allow `<think>...</think>` before `<answer>` in prompt guidance |
| `mode` | str | `"solve"` | `"single"` (one guess, scored) or `"solve"` (multi‑turn to solution) |
| `solve_max_turns` | int | `12` | Max valid guesses in solve mode |
| `relative_pool` | str | `"consistent"` | Candidate pool for `ig_relative`: `consistent` or `all` (sampled if large) |
| `curriculum` | bool/dict | `false` | Enable built‑in auto‑curriculum in single‑turn mode; dict accepts `{low,high,L_min,L_max,K_min,K_max,H_min,H_max,log_path}` |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar (ig_relative/ig/elim or speed) |
| `format_reward_func` | Format bonus for correct `<answer>` tag usage |
| `sh_size_metric` | Size of the consistent set `|S_H|` (logged, weight 0) |

Reward computation (`reward_mode`):
- `elim`: `1 - sum_y p_y^2`, where `p_y` is the fraction of consistent codes that would yield feedback `y` for the guess.
- `ig`: `-sum_y p_y log2 p_y` (bits). For very large spaces, both use an empirical estimate over `sample_n` draws.
- `ig_relative`: `IG(guess) / max_g IG(g)` over a candidate pool. Defaults to the consistent set; change with `relative_pool`.

Additional args for `ig_relative`:
- `relative_pool`: `"consistent"` (default) to compare against the best consistent code; or `"all"` to compare against the best code in the entire space (sampled when large).

### Prompt Format
Each prompt succinctly states the alphabet, code length, whether repeats are allowed, and any prior (guess,feedback) history. The model must respond with:

```
<answer>GUESS: a b c d</answer>
```

Where `a..d` are integers in `0..K-1` and count equals `L`.

### Example Prompt (L=4, K=6)

```
TASK: Mastermind (single-turn)
Alphabet: 0..5 (K=6), Code length: 4
History count: 1
History:
- Guess 0 1 2 3 -> feedback b=1, w=1
CONSTRAINTS: repeats allowed
ACTION FORMAT: Put the guess inside <answer>...</answer> exactly as 'GUESS: a b c d'.
```

### Evaluation Reports
Saved reports generated by `vf-eval` will auto-render here when published to a static site.

### Concept & Curriculum
The critical idea is to train on a single-step environment of adjustable complexity and reward decisions by how close they are to the perfect information-gain choice, then evaluate on full multi-step playthroughs.

- Adjust complexity by Code length (L), Alphabet size (K), and history length; large spaces are automatically approximated by sampling.
- Default single-step reward is a relative IG score: IG(guess) divided by the best IG for the current state, giving a [0,1] “closeness to optimal” target.
- For end-to-end capability, the environment also supports a full multi-turn solve mode (default) that measures how quickly a model can solve a hidden code with bulls/cows feedback.

Auto-curriculum idea: increase (L,K,history_len) when the model’s average reward exceeds a threshold, and decrease otherwise. This can be orchestrated by an outer training loop that updates `-a` parameters across epochs (e.g., start with L=4,K=6; step up to L=5,K=8 when avg relative IG ≥ 0.7). If you’d like, I can add a small sample training script that implements this schedule.
