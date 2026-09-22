# QCache Lab

**QCache Lab** is an experimental framework for **Query Cache attention mechanism intervention** using HuggingFace Transformers.

- **Purpose**: Analyze language model generation behavior by intervening in attention outputs using past query readouts
- **Features**: Teacher-forcing-free full generation, state forking, checkpoints, repetition detection
- **Supported Models**: Llama / Mistral / Qwen2 (Transformers == 4.57.6)

---

## What is QCache?

**Query Cache (QCache)** is an intervention technique for attention mechanism outputs.

In standard attention, each token generates a new Query (Q) which computes attention scores against Keys (K) and Values (V). QCache **stores all past Q vectors and caches their readouts** against K and V. It then modifies the attention output using:

```
Replay Mode (lambda > 0):
  output = (1-lambda) * attention_output + lambda * mean(past_Q_readouts)

Attenuation Mode (lambda > 0):
  output = (1-lambda) * attention_output
```

**Why "QCache"?**
- **Query** (Q) **Cache** - we cache past Query vectors
- We retain past Q and use their readouts to correct attention mechanism outputs
- Makes "memory" and "context" explicitly manipulable

---

## CLI Usage

### Architecture Overview

```
QCache Lab
├── V0.1.2 (legacy)  - Fixed token comparison + basic intervention
└── V1.2+ (live)     - Free trajectory + state forking + multiple branch modes
```

---

### Command Overview

| Command | Description | Output | Time |
|---|---|---|---|
| `device-check` | Verify device behavior and metric boundaries | JSON | ~1s |
| `self-test` | Test PyTorch core and independent toy decoder | JSON | ~10s |
| `hf-smoke` | Test actual HF tiny model adapter integration | JSON | ~30s |
| `run` | **Free trajectory experiments + forking** | Directory | Minutes |
| `legacy-run` | Fixed token comparison (legacy V0.1.2) | Directory | Minutes |

---

## Environment Verification Commands

### 1. `device-check` — Device Behavior Verification

```bash
python3 sr_qcache_lab_v14.py device-check --out qcache_device_check.json
```

**Purpose**: Verify that CPU scoring works correctly under current device hooks without loading any model.

**Checks**:
- Default device detection
- Implicit factory device placement  
- Metric computation device boundary

**Features**:
- No model loading required
- Tests CPU metric boundary under current process hooks
- Verifies factory functions respect explicit CPU device

---

### 2. `self-test` — Internal Logic Validation

```bash
python3 sr_qcache_lab_v14.py self-test --out qcache_selftest.json
```

**Purpose**: Comprehensive test suite for PyTorch mathematical core and independent random-weight toy decoder.

**Test Coverage**:
- **Mathematical Core Tests (27 tests)**:
  - Grouped attention matches explicit head repeat
  - Online recurrence vs dense float64 multiple prefixes
  - Float32 160-position recurrence
  - Half and bfloat queries with float32 accumulators
  - Raw queries survive growth without mutation
  - Prefill bank initialization against entire observed prompt
  - History mean excludes current query
  - Large positive/negative scores stay finite
  - Store-only exact output and null controls
  - Clean prefill unchanged at various lambda values
  - Protocol violation rejections
  - Toy multilayer store and lambda-zero bitwise matching
  - Online vs dense feedback comparison
  - Nonzero intervention changes later logits
  - State reset prevents cross-trial contamination
  - Prefix causal behavior
  - Only selected layers allocate query banks
  - Null stream vs native batch prefill numerical agreement
  - Toy model weights unchanged after all trials
  - Callback contract preserves mask and rejects unmasked prefill
  - Fixed token comparison metrics have exact zero control
  - Metric boundary checks under various device hooks

- **Live Tests (26 tests)**:
  - Live bank admit-all preserves legacy recurrence bitwise
  - Freeze query membership preserves online readout and resumes
  - Bank snapshot preserves stride and never aliases
  - Parallel norm direction/dtype/zero guard
  - Prospective period detector (rotations, exit, return, min period)
  - Event detection is prefix causal
  - Safe snapshot disk roundtrip and tamper rejection
  - Legacy free generation regression across precisions/layers/prefills/strengths
  - Restored free continuations match untouched sources bitwise
  - Sampler RNG snapshot reproduces stochastic free suffix
  - Fork branches are independent with no stale logits
  - Pulse cut exact forward timing and shadow readout continuity
  - Live sources and forks never force future tokens
  - Fork context capacity rejection (not truncated)
  - Whole runner safe disk roundtrip preserves free continuation
  - Boundary direct metrics without canonical continuation
  - Automatic event forks are prefix-selected and source uninterrupted
  - Free report JSON/CSV/HTML and upload packet generation
  - Invalid boundary metadata rejection
  - Manual pre-onset motif probe and unique trace labels
  - Lambda-zero all free fork laws are exact controls
  - Live run end-to-end CLI pipeline with toy adapter stub
  - Live toy parameters remain unchanged

**Total**: 53 test groups, 78+ individual tests

- **Model Required**: No (uses random toy decoder)
- **Status**: 78/78 PASS (verified 2026-09-23)

---

### 3. `hf-smoke` — HF Adapter Integration Test

```bash
python3 sr_qcache_lab_v14.py hf-smoke --out qcache_hf_smoke.json --device cpu
```

**Purpose**: Test actual HF model architectures with random weights - no pretrained model downloads required.

**Test Coverage**:
- Random tiny model creation for each architecture
- Native baseline vs control comparison (baseline/store/replay_0)
- Online vs dense engine comparison
- Active vs native later token comparison
- Attenuation vs independent W_O hook verification
- Schedule matching between replay and attenuation
- Direct pair comparison (JS/KL divergence)
- Model parameter immutability check
- Causal mask and backend restoration verification
- CPU fixed token metrics exercise

**Supported Architectures**:
- Llama
- Mistral
- Qwen2 / Qwen3
- Phi3
- Olmo2
- Mixtral
- Gemma2 (may fail due to device placement issues in some environments)

**Note**: Gemma2 may fail with `RuntimeError('Expected all tensors to be on the same device...')` in Spiralton+MPS environments. This is an environment-specific device placement issue, not a QCache Lab bug.

**Model Required**: No (random initialization)

---

## Experiment Commands

### 4. `run` — Free Trajectory + Forking Experiments

```bash
# Basic usage
python3 sr_qcache_lab_v14.py run \
  --model /path/to/llama-7b \
  --prompt "Tell me a short story." \
  --lambdas "0,0.2,0.55" \
  --branches "continue,cut,native,parallel,freeze" \
  --seeds "42,123" \
  --max-new-tokens 192 \
  --branch-new-tokens 96 \
  --fork-at "50,100" \
  --out my_experiment
```

**Function**: Free source generation with state forking at specified points. Each fork continues from the exact same pre-forward checkpoint with different intervention modes.

**Branch Modes**:

| Mode | Description | Intervention |
|---|---|---|
| `continue` | Normal replay intervention continues | Q input + readout update |
| `cut` | Attenuation mode | Output scaled by (1-lambda), no Q bank |
| `native` | Baseline | No intervention |
| `parallel` | Parallel norm normalization | Per-head pre-W_O norm matching |
| `freeze` | Query input freeze | Stop new Q admission, readouts continue |
| `pulse_cut_N` | Temporary cut | Cut for N steps, then replay |

**Main Options**:

| Option | Default | Description |
|---|---|---|
| `--model` | (required) | Model path or Hub ID (safetensors required) |
| `--prompt` / `--prompt-file` / `--prompt-token-ids` | - | Prompt input (mutually exclusive) |
| `--chat` | - | Apply tokenizer chat template |
| `--lambdas` | `"0.55"` | Comma-separated intervention strengths |
| `--branches` | `"continue,cut,native,parallel,freeze"` | Branch modes to test |
| `--seeds` | `"42"` | Random seeds for reproducibility |
| `--max-new-tokens` | `192` | Source generation token count |
| `--branch-new-tokens` | `96` | Branch generation token count |
| `--fork-at` | - | Comma-separated generated token counts to fork |
| `--max-forks` | `3` | Maximum forks per source |
| `--device` | `auto` | `cpu` / `cuda` / `mps` / `auto` |
| `--dtype` | `auto` | `float32` / `float16` / `bfloat16` / `auto` |
| `--layers` | `all` | Intervention layers (`0:8,16` syntax) |
| `--engine` | `online` | `online` (fast) / `dense` (accurate reference) |
| `--temperature` | `0` | 0=greedy, >0=sampling |
| `--top-p` | `1.0` | Top-p sampling |
| `--prefill` | `clean` | `clean` / `stream` prefill mode |
| `--save-logits` | - | Save full CPU logits (large) |

**Output Files**:
- `reading_room.html` — Browser-viewable HTML report
- `branches.md` — Markdown format report
- `results.json` — All experiment data
- `share_packet.json` — Lightweight sharing packet
- `token_trace.jsonl` — Token selection log
- `events.jsonl` — Repetition event log
- `attention_trace.csv` — Attention mechanism trace
- `checkpoints/*.json` + `*.safetensors` — State snapshots at fork points

---

### 5. `legacy-run` — Fixed Token Comparison (V0.1.2)

```bash
python3 sr_qcache_lab_v14.py legacy-run \
  --model /path/to/model \
  --prompt "Short story about a key." \
  --lambdas "0,0.05,0.1,0.2" \
  --families "replay,attenuation" \
  --layers all \
  --max-new-tokens 64 \
  --max-context 512 \
  --prefill clean \
  --out results_qcache
```

**Function**: Fixed continuation token comparison with baseline. Uses the same fixed token sequence for all conditions.

**Families**:
- `replay`: Uses Q bank with readouts
- `attenuation`: Scales output without Q bank
- Both can be tested simultaneously

**Main Options**:

| Option | Default | Description |
|---|---|---|
| `--families` | `"replay,attenuation"` | Intervention families to test |
| `--lambdas` | `"0,0.05,0.1,0.2"` | Strength values (0.0 always included) |
| `--layers` | `all` | Intervention layers |
| `--prefill` | `clean` | `clean` / `stream` prefill |
| `--engine` | `online` | Readout engine |
| `--max-context` | `512` | Context length (hard limit) |

**Output Files**:
- `generations.md` — Markdown report
- `results.json` — All results
- `fixed_tokens.csv` — Fixed token comparison summary
- `attention_trace.csv` — Attention trace
- `paired_tokens.csv` / `paired_summary.csv` — Replay vs attenuation comparison
- `paired_generations.md` — Paired analysis report

**Note**: Uses fixed continuation tokens. Not fully free generation like V1.2+.

---

## Intervention Protocol

QCache intervention happens **before W_O** (output projection) in the attention mechanism:

```
Replay Mode (lambda > 0):
  history = mean(past_Q_readouts)  # Mean of all past Q readouts
  history = history.transpose(1, 2)  # Shape: [B, T, H, D]
  output = (1-lambda) * base + lambda * history

Attenuation Mode (lambda > 0):
  output = (1-lambda) * base  # No Q bank used

Parallel Mode (lambda > 0):
  # Per-head pre-W_O norm matching
  hypothetical = (1-lambda) * base + lambda * history
  output = norm_match(hypothetical, base)  # Match norms per head
```

**Core Properties**:
- **No KV rewrite**: K, V tensors are never modified (append-only)
- **Append-only**: New KV pairs are added, never replaced or evicted
- **Current Q excluded**: Current query is excluded from history mean calculation
- **Fresh state per trial**: Each experiment has independent KV and Q bank state

---

## Live Verification Status (2026-09-23)

| Command | Status | Details |
|---|---|---|
| `device-check` | PASS | CPU/MPS environment normal |
| `self-test` | 78/78 PASS | All PyTorch + toy tests passed |
| `hf-smoke` | PARTIAL | Llama/Mistral/Qwen2/Qwen3/Phi3/Olmo2/Mixtral PASS, Gemma2 FAIL |

> Gemma2 failure: `RuntimeError('Expected all tensors to be on the same device...')` - Environment-specific device placement issue (Spiralton + MPS)

---

## Requirements

- Python: 3.9+
- PyTorch: 2.0+
- Transformers: == 4.57.6 (strictly pinned)
- safetensors: Required for checkpoint loading
- sentencepiece: Required for some tokenizers

---

## Quick Start

```bash
# 1. Verify environment
python3 sr_qcache_lab_v14.py device-check --out check.json

# 2. Run internal tests
python3 sr_qcache_lab_v14.py self-test --out test.json

# 3. Test HF integration
python3 sr_qcache_lab_v14.py hf-smoke --out smoke.json --device cpu

# 4. Run a simple experiment
python3 sr_qcache_lab_v14.py run \
  --model meta-llama/Llama-2-7b-chat \
  --prompt "Tell me a short story." \
  --lambdas "0,0.2,0.5" \
  --branches "continue,cut,native" \
  --max-new-tokens 64 \
  --out my_first_experiment
```

---

## Links

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Attention Interface v4.57.6](https://huggingface.co/docs/transformers/v4.57.1/en/attention_interface)
