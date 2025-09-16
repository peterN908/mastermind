# Mastermind

**Source Code Repository:** [https://github.com/peterN908/mastermind](https://github.com/peterN908/mastermind)

A mastermind game simulator. Enables evaluating models in a multi-step (full playthrough), and single-step (making a single move) environment. Here the model goes back and forth guessing the code and receiving feedback on how many digits are present or correct.

### The concept

I was interested in this initially because its easy to tune the complexity of the task by adjusting the Code length (L), Alphabet size (K) and history.

Different setups are available but the main idea is that we reward the model based on how close to the "perfect" decision it makes (based on information gain) in a single-step environment.

Finally we evaluate on full multi-step playthroughs. 


### Results

This turned out to be a very interesting way to discriminate model capabilities. Qwen Next 80B A3B was shockingly good with a 54% success rate - twice as high as the next best (also Qwen).

On Qwen 2.5-7B the best way to improve performance seems to be to train on the same difficulty you are assessing on. After ~300 steps GRPO the success rate increases from 0.005% to 2% in our standard L=4, K=6 multi-step eval.  

If you look at how performance varies with Code size (L) and Alphabet Size (K) and History (H) - in general it appears that Code size is the only knob that predictably increases the difficulty (as measured by relative information game). This is because when the Alphabet Size is large, or history is small - its very easy to make a perfect information gaining move by just predicting a code that hasnt been seen yet. 

 
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

Configure model and sampling:

```bash
uv run vf-eval mastermind \
  -m gpt-4.1-nano \
  -n 50 -r 1 -t 2048 -s \
  -a '{"mode":"solve","L":4,"K":6,"allow_repeats":true,"seed":99,"curriculum":false}'
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

### Random Curriculum
Enable per‑example randomization of difficulty in single‑turn mode:

```bash
uv run vf-eval mastermind \
  -m gpt-4.1-mini \
  -n 20 -r 1 -t 512 -T 0.5 -s \
  -a '{"mode":"single", "curriculum": {"mode": "random"}}'
```

Details:
- Randomizes `(L, K, history_len)` per example within bounds.
- Optional bounds: `{L_min,L_max,K_min,K_max,H_min,H_max}`. If `k_equals_l_plus_2=true`, tie `K` to `L+2` within bounds.

Curriculum modes (single‑turn):
- `random`: Per‑example randomize L/K/H within bounds; optional `k_equals_l_plus_2=true` ties K to L.

Performance caps for `ig_relative` (to avoid O(|S_H|^2) blowups):
- `relative_sh_cap` (default 5000): cap the consistent set `S_H` used for scoring.
- `relative_candidate_cap` (default 1000): cap the number of candidate guesses used to find the best IG in the denominator.
- Tip: lowering these caps speeds up runs but makes ig_relative noisier; raise for fidelity if compute allows.

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
| `relative_sh_cap` | int | `5000` | Cap on consistent set size for `ig_relative` scoring |
| `relative_candidate_cap` | int | `1000` | Cap on number of candidate guesses for best-IG search |
| `curriculum` | bool/dict | `false` | Enable per‑example randomization in single‑turn mode; dict accepts `{mode:"random",L_min,L_max,K_min,K_max,H_min,H_max,k_equals_l_plus_2}` |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar (ig_relative/ig/elim or speed) |
| `format_reward` | Format bonus for correct `<answer>` tag usage |
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
<think>Optional short reasoning</think>
<answer>GUESS: a b c ...</answer>
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
(The example adapts to L — for L=3 it will show 'GUESS: a b c'.)
```

### Evaluation Reports
Saved reports generated by `vf-eval` will auto-render here when published to a static site.


### Scripts

- CLI play (single-turn scoring):
  - `mastermind-play` — interactive prompt with optional top‑K suggestions by reward.
  - Example: `uv run mastermind-play --L 4 --K 6 --history-len 1 --suggest-top 5`

- Grid analysis (ig_relative heatmaps and raw completions):
  - `python environments/mastermind/analysis/mastermind_grid_eval.py -m gpt-4.1-nano -n 10 -r 1 --L-min 3 --L-max 7 --K-min 3 --K-max 9 --H-min 0 --H-max 5 --save-completions`
  - Flags: `--rel-sh-cap`, `--rel-cand-cap` to bound compute; `--no-plot` to skip heatmaps.
