#!/usr/bin/env python3
"""QCache Lab V1.4 (1.4.0): prompt/model campaigns from sweep to free forks.

V1.4: capability-based HF adapters; auto/model context; independent Q gain and admission;
free-answer dev/test evaluation via assess. Reference HF version 4.57.6, with runtime feature checks.
Previous explore workflow: explore --model /path/to/model --prompt '...' --chat
Budgeted coarse sweep, multiple candidate intervals, refinement, genuine state
forks, reports, and receipt-checked resume. No teacher-forced future suffixes.
Use explore --dry-run to inspect the upper work budget without loading a model.
The V1.2 live engine and legacy commands remain available below.

Historical V1.2 documentation:

Default entry: run. Source and forked continuations are freely generated.
Checkpoint = all KV, Q/readout/logZ, original query positions, counters,
CPU sampler RNG and one selected but UNPROCESSED pending token.
Future generated token IDs are never forced into another trajectory.
Legacy v0.1.2 is retained below and can be invoked with legacy-run.
HF target remains transformers==4.57.6. Inference only, full DynamicCache.

The historical v0.1.2 documentation follows (legacy-run only):

    python sr_qcache_lab_v012.py self-test --out qcache_selftest.json
    python sr_qcache_lab_v012.py hf-smoke --out qcache_hf_smoke.json
    python sr_qcache_lab_v012.py run --model /path/to/hf-model --local-files-only \
        --device mps --chat --prompt 'Tell a short story.' --out results_qcache

v0.1.1 device fix: CPU scoring indices are explicitly placed on CPU even when
external startup hooks set the default device to MPS. Model and Q-bank placement
are NOT changed, and no external acceleration hooks are disabled.
    python sr_qcache_lab_v012.py device-check

v0.1.2 adds a matched attenuation-only control and direct paired comparisons.
    replay:      (1-lambda)*ordinary_out + lambda*history_readout
    attenuation: (1-lambda)*ordinary_out
Both intervene before W_O on exactly the same eligible positions. Attenuation
has no Q bank; its compute/memory budget and perturbation norm are NOT matched.
Each condition evolves its own KV, even during fixed-token evaluation. Pairwise
metrics describe this total rollout difference, not an isolated local effect.
    python sr_qcache_lab_v012.py run --model /path/to/model --families replay,attenuation

Core tests require only PyTorch. The HF adapter is pinned to transformers==4.57.6.
This is an intervention, not an output-preserving inference optimization.
Inference only; one unpadded sequence; append-only full KV; no beams or cache crop.
All experiment logic, an independent dense reference, tests, and the CLI live here.

Protocol (per layer / query head, positions are zero-based):
    r[i,t] = softmax(q[i] @ K[:t+1].T * scale) @ V[:t+1]
    out[t] = (1-lambda)*ordinary_out[t] + lambda*mean(r[:t,t])
The current query is EXCLUDED from that mean, then saved for the next step.
Post-RoPE Q and existing KV are never rewritten. No learned selector/gate is used.

Clean prefill: normal causal model outputs; Q-bank readouts are initialized against
all observed prompt KV but NOT fed into prefill. First generated token is unchanged.
Stream prefill: process prompt tokens singly and intervene from the second position.

Primary API references (implementation target, not claims of local HF execution):
https://huggingface.co/docs/transformers/v4.57.1/en/attention_interface
https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py
https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/masking_utils.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
import traceback
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:
    raise SystemExit('PyTorch is required. Install a build appropriate for your device.') from exc

VERSION = '0.1.2'
HF_VERSION = '4.57.6'
BACKEND = 'qcache_lab_v012'
SUPPORTED = {'llama', 'mistral', 'qwen2'}
TRACE_FIELDS = [
    'condition', 'branch', 'phase', 'layer', 'position', 'past_queries',
    'lambda', 'gate_mean', 'gate_max', 'readout_drift_rms',
    'history_current_cosine', 'applied_update_relative_l2',
    'bank_used_bytes', 'bank_allocated_bytes',
    'mode', 'observed_past_positions', 'base_output_l2', 'output_l2',
    'output_to_base_l2_ratio', 'history_to_base_l2_ratio',
]


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_devices() -> dict[str, Any]:
    """Observe ambient device behavior without changing defaults or unpatching hooks."""
    result: dict[str, Any] = {}
    try:
        result['declared_default_device'] = str(torch.get_default_device())
    except Exception as exc:
        result['declared_default_device_error'] = repr(exc)
    probes = {
        'implicit_empty': lambda: torch.empty(0),
        'explicit_cpu_empty': lambda: torch.empty(0, device='cpu'),
        'explicit_cpu_index': lambda: torch.arange(1, dtype=torch.long, device='cpu'),
    }
    for name, probe in probes.items():
        try:
            result[name] = str(probe().device)
        except Exception as exc:
            result[name + '_error'] = repr(exc)
    result['factory_origins'] = {}
    for name in ('tensor', 'arange', 'empty'):
        fn = getattr(torch, name)
        code = getattr(fn, '__code__', None)
        result['factory_origins'][name] = {
            'module': getattr(fn, '__module__', None),
            'qualname': getattr(fn, '__qualname__', None),
            'python_source': getattr(code, 'co_filename', None),
        }
    result['loaded_spiralton_modules'] = sorted(
        name for name in sys.modules if 'spiralton' in name.lower())
    return result


def environment() -> dict[str, Any]:
    return {
        'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'transformers': package_version('transformers'),
        'cuda_available': torch.cuda.is_available(),
        'mps_available': bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
        'script_version': VERSION,
        'device_runtime': runtime_devices(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def choose_device(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA was requested but is not available.')
    elif device.type == 'mps':
        require(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
                'MPS was requested but is not available.')
    else:
        require(device.type == 'cpu', 'Only cpu, cuda and mps are supported.')
    return device


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def gqa_scores(q: Tensor, k: Tensor, scale: float) -> Tensor:
    """[B,Hq,Q,D] x [B,Hkv,K,D], without materializing repeated full K."""
    require(q.ndim == k.ndim == 4, 'Q and K must be rank-4 tensors.')
    b, h, n, d = q.shape
    require(k.shape[0] == b and k.shape[-1] == d and h % k.shape[1] == 0,
            'Incompatible Q/K shapes or non-integral GQA grouping.')
    hk = k.shape[1]
    grouped = q.reshape(b, hk, h // hk, n, d)
    scores = torch.matmul(grouped, k.unsqueeze(2).transpose(-1, -2)) * scale
    return scores.reshape(b, h, n, k.shape[-2])


def gqa_values(p: Tensor, v: Tensor) -> Tensor:
    b, h, n, t = p.shape
    hk = v.shape[1]
    require(v.shape[0] == b and v.shape[-2] == t and h % hk == 0,
            'Incompatible probability/value shapes.')
    p = p.reshape(b, hk, h // hk, n, t)
    return torch.matmul(p, v.unsqueeze(2)).reshape(b, h, n, v.shape[-1])


def dense_read(q: Tensor, k: Tensor, v: Tensor, scale: float,
               dtype: torch.dtype = torch.float32, chunk: int = 64) -> tuple[Tensor, Tensor]:
    """Unmasked reread of OBSERVED KV. Returns weighted V and log partition.

    This is deliberately not the model's original causal attention function.
    Chunking bounds the score temporary; no keys or queries are discarded.
    """
    require(chunk > 0 and q.shape[-2] > 0, 'Read chunk and Q length must be positive.')
    require(k.shape[-2] == v.shape[-2] and k.shape[-2] > 0, 'KV must be aligned and nonempty.')
    kf, vf = k.to(dtype), v.to(dtype)
    rs, zs = [], []
    for start in range(0, q.shape[-2], chunk):
        scores = gqa_scores(q[..., start:start + chunk, :].to(dtype), kf, scale)
        z = torch.logsumexp(scores, dim=-1, keepdim=True)
        # softmax, rather than exp(scores-z), is more stable for very large offsets.
        p = torch.softmax(scores, dim=-1)
        rs.append(gqa_values(p, vf))
        zs.append(z)
    return torch.cat(rs, dim=-2), torch.cat(zs, dim=-2)


class QBank:
    """Append-only raw Q plus online readouts. KV remain owned by the model.

    Query storage uses the incoming dtype. Readouts and logZ use acc_dtype.
    Buffers grow geometrically; max_tokens is a hard stop, never an eviction rule.
    """
    def __init__(self, *, max_tokens: int, replay: bool = True, engine: str = 'online',
                 acc_dtype: torch.dtype = torch.float32, chunk: int = 64):
        require(max_tokens > 0 and chunk > 0, 'max_tokens and chunk must be positive.')
        require(engine in {'online', 'dense'}, 'Unknown readout engine.')
        self.max_tokens, self.replay, self.engine = max_tokens, replay, engine
        self.acc_dtype, self.chunk = acc_dtype, chunk
        self.n, self.capacity = 0, 0
        self.q: Tensor | None = None
        self.r: Tensor | None = None
        self.z: Tensor | None = None
        self.scale: float | None = None

    def _reserve(self, count: int, q: Tensor, dv: int) -> None:
        require(count <= self.max_tokens,
                f'Q bank capacity exceeded: {count} > {self.max_tokens}; no silent truncation.')
        if count <= self.capacity:
            return
        cap = min(self.max_tokens, max(count, 16, self.capacity * 2))
        b, h, _, d = q.shape
        newq = torch.empty((b, h, cap, d), device=q.device, dtype=q.dtype)
        if self.n:
            newq[..., :self.n, :].copy_(self.q[..., :self.n, :])
        self.q = newq
        if self.replay:
            newr = torch.empty((b, h, cap, dv), device=q.device, dtype=self.acc_dtype)
            newz = torch.empty((b, h, cap, 1), device=q.device, dtype=self.acc_dtype)
            if self.n:
                newr[..., :self.n, :].copy_(self.r[..., :self.n, :])
                newz[..., :self.n, :].copy_(self.z[..., :self.n, :])
            self.r, self.z = newr, newz
        self.capacity = cap

    @torch.no_grad()
    def bootstrap(self, q: Tensor, k: Tensor, v: Tensor, scale: float) -> None:
        require(self.n == 0, 'bootstrap requires an empty bank; reset between trials.')
        require(q.shape[-2] == k.shape[-2] == v.shape[-2],
                'Bootstrap requires Q for the entire observed prefix, without pre-existing KV.')
        require(q.shape[-2] > 0, 'Cannot bootstrap an empty prefix.')
        self._reserve(q.shape[-2], q, v.shape[-1])
        self.q[..., :q.shape[-2], :].copy_(q.detach())
        self.scale = float(scale)
        if self.replay:
            r, z = dense_read(q, k, v, scale, self.acc_dtype, self.chunk)
            self.r[..., :q.shape[-2], :].copy_(r)
            self.z[..., :q.shape[-2], :].copy_(z)
        self.n = q.shape[-2]

    @torch.no_grad()
    def step(self, q: Tensor, k: Tensor, v: Tensor, scale: float,
             diagnostics: bool = False) -> tuple[Tensor | None, dict[str, float]]:
        require(self.n > 0, 'Bootstrap the bank before a decode step.')
        require(q.shape[-2] == 1, 'After prefill only single-token append is supported.')
        require(k.shape[-2] == v.shape[-2] == self.n + 1,
                'KV/Q length mismatch: crop, eviction, reset, reused cache or speculative decode is unsupported.')
        require(float(scale) == self.scale, 'Attention scale changed during a trial.')
        require(q.shape[:2] == self.q.shape[:2] and q.shape[-1] == self.q.shape[-1]
                and q.device == self.q.device and q.dtype == self.q.dtype,
                'Batch, query heads, device or dtype changed during a trial.')
        self._reserve(self.n + 1, q, v.shape[-1])
        mean, metrics = None, {}
        if self.replay:
            oldr = self.r[..., :self.n, :]
            oldz = self.z[..., :self.n, :]
            s = gqa_scores(self.q[..., :self.n, :].to(self.acc_dtype),
                           k[..., -1:, :].to(self.acc_dtype), scale)
            # Pairwise max-rescaling keeps weights normalized even at large logits.
            # It is algebraically the online logaddexp recurrence, but computes the
            # normalized coefficients without subtracting two huge logZ values.
            m = torch.maximum(oldz, s)
            a, b = torch.exp(oldz - m), torch.exp(s - m)
            denom = a + b
            alpha, beta = a / denom, b / denom
            if self.engine == 'online':
                hk = v.shape[1]
                newest_v = v[..., -1:, :].to(self.acc_dtype).repeat_interleave(q.shape[1] // hk, dim=1)
                newr = alpha * oldr + beta * newest_v
                newz = m + torch.log(denom)
            else:
                newr, newz = dense_read(self.q[..., :self.n, :], k, v, scale,
                                        self.acc_dtype, self.chunk)
            if diagnostics:
                metrics = {
                    'gate_mean': float(beta.mean().item()),
                    'gate_max': float(beta.max().item()),
                    'readout_drift_rms': float((newr - oldr).square().mean().sqrt().item()),
                }
            oldr.copy_(newr)
            oldz.copy_(newz)
            # Exclude the current Q BEFORE appending its readout.
            mean = newr.mean(dim=-2, keepdim=True)
            current_r, current_z = dense_read(q, k, v, scale, self.acc_dtype, self.chunk)
            self.r[..., self.n:self.n + 1, :].copy_(current_r)
            self.z[..., self.n:self.n + 1, :].copy_(current_z)
        self.q[..., self.n:self.n + 1, :].copy_(q.detach())
        self.n += 1
        return mean, metrics

    @torch.no_grad()
    def verify(self, k: Tensor, v: Tensor, atol: float = 2e-4, rtol: float = 2e-4) -> dict[str, float]:
        require(self.replay and self.n > 0, 'Verification requires initialized readouts.')
        r, z = dense_read(self.q[..., :self.n, :], k, v, self.scale, self.acc_dtype, self.chunk)
        torch.testing.assert_close(self.r[..., :self.n, :], r, atol=atol, rtol=rtol)
        torch.testing.assert_close(self.z[..., :self.n, :], z, atol=atol, rtol=rtol)
        return {'r_max_abs': float((r - self.r[..., :self.n, :]).abs().max()),
                'logz_max_abs': float((z - self.z[..., :self.n, :]).abs().max())}

    def bytes(self, allocated: bool = False) -> int:
        n = self.capacity if allocated else self.n
        total = 0
        for tensor in (self.q, self.r, self.z):
            if tensor is not None:
                total += tensor.shape[0] * tensor.shape[1] * n * tensor.shape[-1] * tensor.element_size()
        return total


@dataclass(frozen=True)
class Condition:
    name: str
    mode: str = 'baseline'
    strength: float = 0.0

    def __post_init__(self) -> None:
        require(self.mode in {'baseline', 'store', 'replay', 'attenuation'}, 'Unknown condition mode.')
        require(math.isfinite(self.strength) and 0 <= self.strength <= 1,
                'lambda must be finite and within [0,1].')
        require(self.mode in {'replay', 'attenuation'} or self.strength == 0,
                'Only replay and attenuation have a nonzero lambda.')

    @property
    def is_null(self) -> bool:
        return self.mode in {'baseline', 'store'} or self.strength == 0


class Controller:
    def __init__(self, *, layers: set[int] | None = None, max_tokens: int = 512,
                 engine: str = 'online', chunk: int = 64, trace_every: int = 1,
                 verify_every: int = 0, acc_dtype: torch.dtype = torch.float32):
        self.layers, self.max_tokens = layers, max_tokens
        self.engine, self.chunk = engine, chunk
        self.trace_every, self.verify_every, self.acc_dtype = trace_every, verify_every, acc_dtype
        self.trace_sink: Callable[[dict[str, Any]], None] | None = None
        self.reset(Condition('baseline'))

    def reset(self, condition: Condition, branch: str = '') -> None:
        self.condition, self.branch, self.phase = condition, branch, 'unknown'
        self.banks: dict[int, QBank] = {}
        # Attenuation stores only shape/position metadata, never Q or readouts.
        self.attenuation_seen: dict[int, tuple[int, float, tuple[Any, ...]]] = {}
        self.eligible_positions: dict[int, list[int]] = {}
        self.trace: list[dict[str, Any]] = []
        self.calls = 0

    @torch.no_grad()
    def apply(self, layer: int, q: Tensor, k: Tensor, v: Tensor, base: Tensor,
              scale: float) -> Tensor:
        """HF contract: Q/K/V=[B,H,T,D], output=[B,T,H,D]."""
        c = self.condition
        if c.mode == 'baseline' or (self.layers is not None and layer not in self.layers):
            return base
        require(q.shape[0] == 1, 'v0.1 supports one unpadded sequence, not batches or beams.')
        require(base.shape == (q.shape[0], q.shape[-2], q.shape[1], v.shape[-1]),
                'Unexpected attention output layout.')
        self.calls += 1
        if c.mode == 'attenuation':
            return self._attenuate(layer, q, k, v, base, scale)
        bank = self.banks.get(layer)
        if bank is None:
            bank = QBank(max_tokens=self.max_tokens, replay=(c.mode == 'replay'),
                         engine=self.engine, acc_dtype=self.acc_dtype, chunk=self.chunk)
            bank.bootstrap(q, k, v, scale)
            self.banks[layer] = bank
            return base
        past = bank.n
        diag = bool(self.trace_every and (k.shape[-2] - 1) % self.trace_every == 0)
        mean, metrics = bank.step(q, k, v, scale, diagnostics=diag)
        self.eligible_positions.setdefault(layer, []).append(bank.n - 1)
        if self.verify_every and c.mode == 'replay' and bank.n % self.verify_every == 0:
            bank.verify(k, v)
        # Exact bypass, not multiply-by-one/add-zero. Lambda=0 is a real null test.
        if c.mode == 'store' or c.strength == 0:
            out = base
        else:
            history = mean.transpose(1, 2)
            out = ((1 - c.strength) * base.to(self.acc_dtype) + c.strength * history).to(base.dtype)
        if diag:
            self._record_trace(layer, bank.n - 1, past, base, out, mean, metrics,
                               bank.bytes(), bank.bytes(True), past)
        return out

    def _record_trace(self, layer: int, position: int, past_queries: int, base: Tensor,
                      out: Tensor, mean: Tensor | None, metrics: dict[str, float],
                      used: int, allocated: int, observed_past: int) -> None:
        a, result = base.to(self.acc_dtype), out.to(self.acc_dtype)
        eps = torch.finfo(self.acc_dtype).eps
        norm = a.norm()
        relative = float((result - a).norm() / norm.clamp_min(eps))
        cosine, history_ratio = None, None
        if mean is not None:
            b = mean.transpose(1, 2)
            cosine = float(((a * b).sum(-1) /
                            (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(eps)).mean())
            history_ratio = float(b.norm() / norm) if float(norm) > 0 else None
        row = {
            'condition': self.condition.name, 'branch': self.branch, 'phase': self.phase,
            'layer': layer, 'position': position, 'past_queries': past_queries,
            'lambda': self.condition.strength, 'gate_mean': metrics.get('gate_mean'),
            'gate_max': metrics.get('gate_max'), 'readout_drift_rms': metrics.get('readout_drift_rms'),
            'history_current_cosine': cosine, 'applied_update_relative_l2': relative,
            'bank_used_bytes': used, 'bank_allocated_bytes': allocated,
            'mode': self.condition.mode, 'observed_past_positions': observed_past,
            'base_output_l2': float(norm), 'output_l2': float(result.norm()),
            'output_to_base_l2_ratio': float(result.norm() / norm) if float(norm) > 0 else None,
            'history_to_base_l2_ratio': history_ratio,
        }
        if self.trace_sink is None:
            self.trace.append(row)
        else:
            self.trace_sink(row)

    def _attenuate(self, layer: int, q: Tensor, k: Tensor, v: Tensor,
                   base: Tensor, scale: float) -> Tensor:
        # Match replay's first-forward bypass, including a one-token clean prompt
        # and position zero of stream prefill. Never weaken clean prefill alone.
        total = k.shape[-2]
        require(total == v.shape[-2] and total <= self.max_tokens,
                'Attenuation needs aligned full KV within context capacity; no truncation.')
        shape = (tuple(q.shape[:2]), q.shape[-1], q.device, q.dtype, v.shape[-1])
        state = self.attenuation_seen.get(layer)
        if state is None:
            require(q.shape[-2] == total and total > 0,
                    'Attenuation bootstrap requires the entire prefix without pre-existing KV.')
            self.attenuation_seen[layer] = (total, float(scale), shape)
            return base
        past, old_scale, old_shape = state
        require(q.shape[-2] == 1 and total == past + 1,
                'Attenuation requires single-token append; crop, reset and reused KV are unsupported.')
        require(float(scale) == old_scale and shape == old_shape,
                'Attenuation scale, shape, device or dtype changed within a trial.')
        self.attenuation_seen[layer] = (total, old_scale, old_shape)
        self.eligible_positions.setdefault(layer, []).append(total - 1)
        strength = self.condition.strength
        # The zero control returns the EXACT original object. At lambda=1 the
        # pre-projection tensor is zero; residual/MLP and W_O bias are not removed.
        out = base if strength == 0 else ((1 - strength) * base.to(self.acc_dtype)).to(base.dtype)
        if self.trace_every and (total - 1) % self.trace_every == 0:
            self._record_trace(layer, total - 1, 0, base, out, None, {}, 0, 0, past)
        return out

    def schedule(self) -> dict[str, Any]:
        return {
            str(layer): {
                'eligible_count': len(pos), 'first_position': pos[0], 'last_position': pos[-1],
                'position_sequence_sha256': hashlib.sha256(
                    json.dumps(pos, separators=(',', ':')).encode('ascii')).hexdigest(),
            }
            for layer, pos in sorted(self.eligible_positions.items()) if pos
        }

    def memory(self) -> dict[str, Any]:
        return {'used_bytes': sum(x.bytes() for x in self.banks.values()),
                'allocated_bytes': sum(x.bytes(True) for x in self.banks.values()),
                'queries_per_layer': {str(i): x.n for i, x in self.banks.items()}}


def require_hf() -> Any:
    actual = package_version('transformers')
    require(actual == HF_VERSION,
            f'This adapter targets transformers=={HF_VERSION}; installed={actual!r}. '
            f'Use a separate environment: pip install "transformers=={HF_VERSION}" sentencepiece')
    return importlib.import_module('transformers')


def hf_attention(module: nn.Module, query: Tensor, key: Tensor, value: Tensor,
                 attention_mask: Tensor | None, **kwargs: Any) -> tuple[Tensor, Tensor | None]:
    """Delegate base attention to HF, preserve its mask, then intervene before W_O."""
    require(not module.training and not torch.is_grad_enabled(),
            'QCache Lab is inference-only: call eval() inside torch.inference_mode().')
    original = getattr(module, '_qcache_lab_original_eager', None)
    require(original is not None, 'QCache backend used without an attached adapter.')
    if query.shape[-2] > 1:
        require(attention_mask is not None, 'Missing causal prefill mask; refusing to run.')
    base, weights = original(module, query, key, value, attention_mask, **kwargs)
    controller = module._qcache_lab_controller
    out = controller.apply(int(module.layer_idx), query, key, value, base, float(kwargs['scaling']))
    # Base-only weights would misdescribe mixed output. Do not return them after mixing.
    return out, weights if out is base else None


def validate_config(config: Any) -> None:
    mt = config.model_type
    require(mt in SUPPORTED, f'Unsupported model_type={mt!r}; supported: {sorted(SUPPORTED)}')
    if mt == 'mistral':
        require(getattr(config, 'sliding_window', None) is None,
                'Active sliding-window Mistral is not supported. Do not silently change its attention.')
    if mt == 'qwen2':
        require(not getattr(config, 'use_sliding_window', False), 'Sliding-window Qwen2 is unsupported.')
    require(not any('sliding' in str(x) for x in (getattr(config, 'layer_types', None) or [])),
            'A sliding attention layer was found; this experiment requires full KV.')
    rope = getattr(config, 'rope_scaling', None) or {}
    rope_type = rope.get('rope_type', rope.get('type', 'default'))
    require(rope_type not in {'dynamic', 'longrope'},
            'Sequence-dependent RoPE frequency changes are outside the fixed-position protocol.')
    require(not getattr(config, 'quantization_config', None),
            'Quantized checkpoints are outside the v0.1 validation scope.')


class HFAdapter:
    """Official attention/mask registry adapter. No weight or module-global patching.

    There is one controller per model. Do not concurrently run unrelated sequences
    through a single adapter. Explicit close() restores the original backend.
    """
    def __init__(self, model: nn.Module, controller: Controller):
        hf = require_hf()
        validate_config(model.config)
        require(not model.training, 'Set model.eval() before attaching.')
        require(not hasattr(model, 'hf_device_map') or len(set(model.hf_device_map.values())) <= 1,
                'Sharded/device_map execution is outside the supported scope.')
        self.model, self.controller = model, controller
        self.previous = model.config._attn_implementation or 'eager'
        self.modules: list[nn.Module] = []
        from transformers.masking_utils import eager_mask
        # Both registrations are essential: a backend without a registered mask
        # can receive attention_mask=None in HF and accidentally remove causality.
        hf.AttentionInterface.register(BACKEND, hf_attention)
        hf.AttentionMaskInterface.register(BACKEND, eager_mask)
        try:
            for layer in model.model.layers:
                attn = layer.self_attn
                require(not hasattr(attn, '_qcache_lab_controller'), 'An adapter is already attached.')
                require(getattr(attn, 'sliding_window', None) is None, 'Sliding attention is unsupported.')
                src = importlib.import_module(type(attn).__module__)
                require(hasattr(src, 'eager_attention_forward'), 'Unsupported attention implementation.')
                attn._qcache_lab_original_eager = src.eager_attention_forward
                attn._qcache_lab_controller = controller
                self.modules.append(attn)
            model.set_attn_implementation(BACKEND)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if hasattr(self, 'model'):
            self.model.set_attn_implementation(self.previous)
        for module in self.modules:
            for attr in ('_qcache_lab_original_eager', '_qcache_lab_controller'):
                if hasattr(module, attr):
                    delattr(module, attr)
        self.modules.clear()

    def __enter__(self) -> 'HFAdapter':
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class HFRunner:
    def __init__(self, model: nn.Module, controller: Controller, device: torch.device):
        self.model, self.controller, self.device = model, controller, device
        self.cache: Any = None
        self.count = 0

    def reset(self, condition: Condition, branch: str) -> None:
        from transformers.cache_utils import DynamicCache
        self.controller.reset(condition, branch)
        # No config-driven eviction: validate_config has already rejected sliding models.
        self.cache, self.count = DynamicCache(), 0

    @torch.inference_mode()
    def forward(self, token_ids: list[int]) -> Tensor:
        require(bool(token_ids), 'Cannot forward an empty token sequence.')
        total = self.count + len(token_ids)
        require(total <= self.controller.max_tokens, 'Context capacity exceeded; no silent truncation.')
        ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        output = self.model(
            input_ids=ids, past_key_values=self.cache, use_cache=True,
            attention_mask=torch.ones((1, total), device=self.device, dtype=torch.long),
            cache_position=torch.arange(self.count, total, device=self.device),
            logits_to_keep=1, return_dict=True,
        )
        self.cache, self.count = output.past_key_values, total
        logits = output.logits[0, -1].detach().float().cpu()
        require(bool(torch.isfinite(logits).all()), 'Nonfinite model logits; trial aborted.')
        return logits


def prefill(runner: Any, prompt_ids: list[int], mode: str) -> Tensor:
    require(bool(prompt_ids), 'Prompt must contain at least one token.')
    if mode == 'clean':
        runner.controller.phase = 'clean_prefill'
        logits = runner.forward(prompt_ids)
    else:
        require(mode == 'stream', 'Unknown prefill mode.')
        runner.controller.phase = 'stream_prefill'
        for token in prompt_ids:
            logits = runner.forward([token])
    runner.controller.phase = 'continuation'
    return logits


def choose_token(logits: Tensor, generator: torch.Generator,
                 temperature: float = 0.0, top_p: float = 1.0) -> int:
    # The seeded generator belongs to CPU. Do not inherit a global MPS default.
    logits = logits.detach().to(device='cpu')
    if temperature == 0:
        return int(logits.argmax())
    p = torch.softmax(logits.double() / temperature, dim=-1)
    if top_p < 1:
        ordered, index = torch.sort(p, descending=True)
        remove = ordered.cumsum(-1) - ordered >= top_p
        ordered[remove] = 0
        ordered /= ordered.sum()
        return int(index[torch.multinomial(ordered, 1, generator=generator)])
    return int(torch.multinomial(p, 1, generator=generator))


@torch.inference_mode()
def trial(runner: Any, condition: Condition, prompt: list[int], *, prefill_mode: str,
          max_new: int, seed: int, forced: list[int] | None = None,
          eos_ids: set[int] | None = None, temperature: float = 0, top_p: float = 1,
          branch: str = 'free') -> dict[str, Any]:
    runner.reset(condition, branch)
    synchronize(runner.device)
    started = time.perf_counter()
    logits = prefill(runner, prompt, prefill_mode)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    tokens, logit_rows = [], []
    count = len(forced) if forced is not None else max_new
    require(count > 0, 'Continuation must be nonempty.')
    for i in range(count):
        logit_rows.append(logits)
        token = forced[i] if forced is not None else choose_token(logits, generator, temperature, top_p)
        tokens.append(token)
        if forced is None and eos_ids and token in eos_ids:
            break
        if i + 1 < count:
            logits = runner.forward([token])
    synchronize(runner.device)
    seconds = time.perf_counter() - started
    return {'token_ids': tokens, 'logits': torch.stack(logit_rows),
            'elapsed_seconds': seconds, 'qcache': runner.controller.memory(),
            'attention_calls': runner.controller.calls,
            'intervention_schedule': runner.controller.schedule(),
            'stop_reason': ('fixed_length' if forced is not None else
                            'eos' if eos_ids and tokens[-1] in eos_ids else 'max_new_tokens')}


def compare_logits(reference: Tensor, candidate: Tensor, target_ids: list[int],
                   condition: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(reference.shape == candidate.shape and len(target_ids) == reference.shape[0],
            'Logit comparison must use identical prefixes and target token IDs.')
    require(reference.ndim == 2 and reference.shape[0] > 0 and reference.shape[1] > 0,
            'Logits must be a nonempty [positions, vocabulary] matrix.')
    require(all(isinstance(t, int) and 0 <= t < reference.shape[1] for t in target_ids),
            'Target token IDs must be integer indices within the vocabulary.')
    # Scoring is CPU float64 regardless of the MODEL device. Transfer before
    # float64 conversion so callers can also provide MPS tensors safely.
    reference = reference.detach().to(device='cpu', dtype=torch.float64)
    candidate = candidate.detach().to(device='cpu', dtype=torch.float64)
    require(reference.device.type == candidate.device.type == 'cpu',
            'Metric tensors were not placed on CPU; check external tensor hooks.')
    require(bool(torch.isfinite(reference).all()) and bool(torch.isfinite(candidate).all()),
            'Cannot compare nonfinite logits.')
    p_log = torch.log_softmax(reference, dim=-1)
    q_log = torch.log_softmax(candidate, dim=-1)
    p, q = p_log.exp(), q_log.exp()
    kl = (p * (p_log - q_log)).sum(-1).clamp_min(0)
    tv = 0.5 * (p - q).abs().sum(-1)
    log_m = torch.logaddexp(p_log, q_log) - math.log(2)
    js = (0.5 * ((p * (p_log - log_m)).sum(-1) + (q * (q_log - log_m)).sum(-1))).clamp_min(0)
    targets = torch.tensor(target_ids, dtype=torch.long, device=p_log.device)
    idx = torch.arange(len(target_ids), dtype=torch.long, device=p_log.device)
    require(targets.device == idx.device == p_log.device == q_log.device,
            f'Metric device mismatch: logits={p_log.device}, targets={targets.device}, '
            f'rows={idx.device}. An external hook may be overriding explicit devices.')
    p_nll, q_nll = -p_log[idx, targets], -q_log[idx, targets]
    delta = (candidate.double() - reference.double()).abs().amax(-1)
    p_top, q_top = reference.argmax(-1), candidate.argmax(-1)
    rows = []
    for i, token in enumerate(target_ids):
        rows.append({
            'condition': condition, 'step': i, 'target_id': token,
            'reference_top1': int(p_top[i]), 'condition_top1': int(q_top[i]),
            'top1_same': bool(p_top[i] == q_top[i]),
            'kl_baseline_to_condition': float(kl[i]), 'js_divergence': float(js[i]),
            'total_variation': float(tv[i]), 'max_abs_logit_delta': float(delta[i]),
            'reference_nll': float(p_nll[i]), 'condition_nll': float(q_nll[i]),
            'nll_delta': float(q_nll[i] - p_nll[i]),
        })
    summary = {'mean_kl': float(kl.mean()), 'max_kl': float(kl.max()),
               'mean_js': float(js.mean()), 'mean_total_variation': float(tv.mean()),
               'max_abs_logit_delta': float(delta.max()),
               'top1_agreement': float((p_top == q_top).double().mean()),
               'mean_nll_delta': float((q_nll - p_nll).mean()),
               'reference_mean_nll': float(p_nll.mean()), 'condition_mean_nll': float(q_nll.mean())}
    return rows, summary



def build_conditions(lambdas: str, families: str = 'replay,attenuation') -> list[Condition]:
    modes = [m.strip() for m in families.split(',')]
    require(bool(modes) and len(set(modes)) == len(modes) and
            set(modes) <= {'replay', 'attenuation'},
            '--families must contain replay, attenuation, or both, without duplicates.')
    strengths = list(dict.fromkeys([0.] + [float(x.strip()) for x in lambdas.split(',')]))
    specs = [Condition('baseline'), Condition('store', 'store')]
    # Keep each lambda's pair adjacent: at most one prior fixed-logit matrix
    # is retained for pairing, not all lambda sweeps in host RAM.
    specs += [Condition(f'{m}_{v:g}', m, v) for v in strengths for m in modes]
    require(len({s.name for s in specs}) == len(specs),
            'Two lambda values have the same formatted label; choose more separated values.')
    return specs


def free_prefix_comparison(a: list[int], b: list[int]) -> dict[str, Any]:
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    equal = a == b
    return {'free_token_ids_equal': equal, 'free_common_prefix_tokens': common,
            'free_first_difference_step': None if equal else common,
            'free_difference_kind': ('none' if equal else
                                     'token' if common < min(len(a), len(b)) else 'length'),
            'attenuation_free_tokens': len(a), 'replay_free_tokens': len(b)}


def compare_pair(attenuation: Tensor, replay: Tensor, targets: list[int], strength: float,
                 tokenizer: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    '''Direct distributions on identical token prefixes, NOT subtraction of baseline KLs.

    CPU float64; process short chunks so two large vocabularies do not create
    an additional full-sequence float64 working set. Token-ID columns are exact;
    decoded token fragments are display aids and may have tokenizer artifacts.
    '''
    require(attenuation.shape == replay.shape and attenuation.ndim == 2 and
            len(targets) == attenuation.shape[0] and len(targets) > 0,
            'Pair comparison requires aligned nonempty fixed-token logit matrices.')
    name = f'lambda_{strength:g}'
    rows: list[dict[str, Any]] = []
    for start in range(0, len(targets), 8):
        end = min(start + 8, len(targets))
        forward, _ = compare_logits(attenuation[start:end], replay[start:end], targets[start:end], name)
        reverse, _ = compare_logits(replay[start:end], attenuation[start:end], targets[start:end], name)
        for local, (f, r) in enumerate(zip(forward, reverse)):
            step = start + local
            row = {
                'pair': name, 'lambda': strength, 'step': step, 'target_id': targets[step],
                'attenuation_top1': f['reference_top1'], 'replay_top1': f['condition_top1'],
                'top1_same': f['top1_same'],
                'kl_attenuation_to_replay': f['kl_baseline_to_condition'],
                'kl_replay_to_attenuation': r['kl_baseline_to_condition'],
                'js_divergence': f['js_divergence'], 'total_variation': f['total_variation'],
                'max_abs_logit_delta': f['max_abs_logit_delta'],
                'attenuation_nll': f['reference_nll'], 'replay_nll': f['condition_nll'],
                'replay_minus_attenuation_nll': f['nll_delta'],
            }
            if tokenizer is not None:
                row.update(target_token=tokenizer.decode([row['target_id']], skip_special_tokens=False),
                           attenuation_top1_token=tokenizer.decode([row['attenuation_top1']], skip_special_tokens=False),
                           replay_top1_token=tokenizer.decode([row['replay_top1']], skip_special_tokens=False),
                           fixed_prefix_tail=tokenizer.decode(targets[max(0, step - 24):step], skip_special_tokens=False))
            rows.append(row)
    def average(field: str) -> float:
        return math.fsum(r[field] for r in rows) / len(rows)
    mismatches = [r['step'] for r in rows if not r['top1_same']]
    summary = {
        'lambda': strength, 'fixed_positions': len(rows),
        'mean_js': average('js_divergence'), 'max_js': max(r['js_divergence'] for r in rows),
        'mean_kl_attenuation_to_replay': average('kl_attenuation_to_replay'),
        'mean_kl_replay_to_attenuation': average('kl_replay_to_attenuation'),
        'mean_total_variation': average('total_variation'),
        'max_abs_logit_delta': max(r['max_abs_logit_delta'] for r in rows),
        'top1_disagreements': len(mismatches),
        'top1_agreement': 1 - len(mismatches) / len(rows),
        'first_top1_difference_step': mismatches[0] if mismatches else None,
        'first_logit_difference_step': next((r['step'] for r in rows if r['max_abs_logit_delta'] > 0), None),
        'mean_replay_minus_attenuation_nll': average('replay_minus_attenuation_nll'),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_pair_outputs(out: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    pairs = report.get('pairs', {})
    write_csv(out / 'paired_tokens.csv', rows)
    write_csv(out / 'paired_summary.csv', [dict(pair=name, **item['summary']) for name, item in pairs.items()])
    lines = ['# Q replay vs attenuation', '', f"Status: {report.get('status')}", '',
             'Both use the same ordinary-attention coefficient (1-lambda), layer set and eligible positions.',
             'Fixed comparison: same token IDs, but each intervention has its OWN evolving KV and hidden states.',
             'Direct JS/KL are computed between the two conditions, not by subtracting their baseline distances.',
             'Free text after the first divergent token is not a matched-prefix comparison.',
             'Compute, perturbation norms and signal content are not matched by this control. No quality claim.', '',
             '| Lambda | Mean JS(A,R) | KL(A||R) | Top-1 differences | First fixed difference (0-based) | Free common prefix |',
             '|---|---:|---:|---:|---:|---:|']
    for item in pairs.values():
        x = item['summary']
        lines.append(f"| {x['lambda']:g} | {x['mean_js']:.8g} | {x['mean_kl_attenuation_to_replay']:.8g} | "
                     f"{x['top1_disagreements']}/{x['fixed_positions']} | {x['first_top1_difference_step']} | "
                     f"{x['free_common_prefix_tokens']} |")
    for name, item in pairs.items():
        lines.extend(['', f'## {name}', '', 'A = attenuation, R = Q replay.'])
        for mode in ('attenuation', 'replay'):
            result = report['conditions'][item[mode + '_condition']]
            lines.extend(['', f"### {mode} (stop: {result['free']['stop_reason']})", '',
                          '```text', result['free']['text'].replace('```', "'''"), '```'])
        differing = [r for r in rows if r['pair'] == name and not r['top1_same']]
        if differing:
            lines.extend(['', '### First fixed-token top-1 differences', '', '```json',
                          json.dumps([{key: r[key] for key in ('step', 'fixed_prefix_tail',
                                                              'attenuation_top1', 'replay_top1',
                                                              'attenuation_top1_token', 'replay_top1_token') if key in r}
                                      for r in differing[:12]], ensure_ascii=False, indent=2), '```'])
    (out / 'paired_generations.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def _cpu_metric_check() -> dict[str, Any]:
    """Exercise the formerly broken path, without loading a model or changing defaults."""
    logits = torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]],
                          device='cpu', dtype=torch.float32)
    require(logits.device.type == 'cpu', 'An explicit CPU tensor was redirected by an external hook.')
    targets = [2, 1]
    rows, summary = compare_logits(logits, logits.clone(), targets, 'boundary_control')
    assert summary['max_abs_logit_delta'] == 0.0
    assert summary['mean_kl'] == 0.0 and summary['mean_nll_delta'] == 0.0
    for row, values, target in zip(rows, logits.tolist(), targets):
        expected = math.log(sum(math.exp(x) for x in values)) - values[target]
        assert math.isclose(row['reference_nll'], expected, rel_tol=0.0, abs_tol=1e-12)
    candidate = logits.clone()
    candidate[0, 0] += 0.375
    _, changed = compare_logits(logits, candidate, targets, 'boundary_changed')
    assert changed['mean_kl'] > 0 and changed['mean_nll_delta'] > 0
    return {'metrics_device': 'cpu', 'metrics_dtype': 'float64',
            'zero_control_exact': True, 'nll_matches_independent_scalar_math': True,
            'nonzero_kl': changed['mean_kl'], 'reference_mean_nll': summary['reference_mean_nll']}


def _non_cpu_default_metric_check() -> dict[str, Any]:
    expected = _cpu_metric_check()
    # meta is a storage-free, non-CPU default stress test, NOT an MPS execution.
    with torch.device('meta'):
        observed = _cpu_metric_check()
    assert observed == expected
    return {'stress_context': 'meta', 'same_metrics_as_ambient_context': True,
            'real_mps_execution': False}


def _explicit_metric_factories_check() -> dict[str, Any]:
    from unittest.mock import patch
    seen: list[dict[str, str]] = []
    def guarded(name: str, original: Callable[..., Tensor]):
        def call(*args: Any, **kwargs: Any) -> Tensor:
            requested = kwargs.get('device')
            assert requested is not None, f'{name} must not inherit the global default device'
            assert torch.device(requested).type == 'cpu'
            result = original(*args, **kwargs)
            assert result.device.type == 'cpu'
            seen.append({'factory': name, 'device': str(result.device)})
            return result
        return call
    with patch.object(torch, 'tensor', guarded('tensor', torch.tensor)), \
         patch.object(torch, 'arange', guarded('arange', torch.arange)):
        _cpu_metric_check()
    assert sum(x['factory'] == 'arange' for x in seen) == 2
    return {'checked_calls': len(seen), 'every_index_factory_explicit_cpu': True}


def _non_cpu_default_sampler_check() -> dict[str, Any]:
    logits = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=torch.float32, device='cpu')
    def draw() -> list[int]:
        g = torch.Generator(device='cpu').manual_seed(311)
        return [choose_token(logits, g, 0.8, .9) for _ in range(20)]
    expected = draw()
    with torch.device('meta'):
        observed = draw()
    assert observed == expected
    return {'same_sampled_token_ids': True, 'draws': len(observed), 'real_mps_execution': False}


def device_check(out: Path) -> bool:
    report: dict[str, Any] = {'schema': 'qcache.device_check.v1', 'environment': environment(),
                              'scope': 'CPU metric boundary under the current process hooks; no model forward'}
    try:
        report['metrics'] = _cpu_metric_check()
        report['factories'] = _explicit_metric_factories_check()
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    json_write(out, report)
    snapshot = report['environment']['device_runtime']
    print(f'Device check: {report["status"]}; '
          f'default={snapshot.get("declared_default_device", "unknown")}, '
          f'implicit_factory={snapshot.get("implicit_empty", "failed")}, metrics=cpu', flush=True)
    print(f'Report: {out}', flush=True)
    return report['status'] == 'passed'


def repetition_stats(ids: list[int]) -> dict[str, Any]:
    if not ids:
        return {'tokens': 0}
    runs, run = 1, 1
    for i in range(1, len(ids)):
        run = run + 1 if ids[i] == ids[i - 1] else 1
        runs = max(runs, run)
    result: dict[str, Any] = {'tokens': len(ids), 'unique_token_fraction': len(set(ids)) / len(ids),
                              'max_identical_token_run': runs}
    for n in (2, 3):
        grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
        result[f'repeated_{n}gram_fraction'] = 1 - len(set(grams)) / len(grams) if grams else None
    return result


def parse_layers(text: str, count: int) -> set[int]:
    if text == 'all':
        return set(range(count))
    result = set()
    for item in text.split(','):
        if ':' in item:
            start, stop = item.split(':')
            result.update(range(int(start or 0), int(stop or count)))
        else:
            result.add(int(item))
    require(bool(result) and min(result) >= 0 and max(result) < count, 'Invalid layer selection.')
    return result


class _ToyLayer(nn.Module):
    """Small independent test decoder. Random weights, not a language model demo."""
    def __init__(self, hidden: int = 32, heads: int = 4, kvheads: int = 2):
        super().__init__()
        self.h, self.hk, self.d = heads, kvheads, hidden // heads
        self.norm = nn.LayerNorm(hidden, device='cpu')
        self.norm2 = nn.LayerNorm(hidden, device='cpu')
        self.qp, self.kp, self.vp = nn.Linear(hidden, hidden, device='cpu'), nn.Linear(hidden, kvheads * self.d, device='cpu'), nn.Linear(hidden, kvheads * self.d, device='cpu')
        self.op = nn.Linear(hidden, hidden, device='cpu')
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 2, device='cpu'), nn.SiLU(), nn.Linear(hidden * 2, hidden, device='cpu'))

    def forward(self, x: Tensor, kv: tuple[Tensor, Tensor] | None, controller: Controller | None,
                layer: int) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        b, t, _ = x.shape
        old = 0 if kv is None else kv[0].shape[-2]
        normalized = self.norm(x)
        q = self.qp(normalized).view(b, t, self.h, self.d).transpose(1, 2)
        k = self.kp(normalized).view(b, t, self.hk, self.d).transpose(1, 2)
        v = self.vp(normalized).view(b, t, self.hk, self.d).transpose(1, 2)
        pos = torch.arange(old, old + t, device=x.device, dtype=torch.float32)
        inv = 10000 ** (-torch.arange(0, self.d, 2, device=x.device, dtype=torch.float32) / self.d)
        angle = pos[:, None] * inv[None, :]
        angle = torch.cat([angle, angle], -1).to(x.dtype)
        cos, sin = angle.cos()[None, None], angle.sin()[None, None]
        def rotate(z: Tensor) -> Tensor:
            half = self.d // 2
            return z * cos + torch.cat([-z[..., half:], z[..., :half]], -1) * sin
        q, k = rotate(q), rotate(k)
        if kv is not None:
            k, v = torch.cat([kv[0], k], -2), torch.cat([kv[1], v], -2)
        kr, vr = k.repeat_interleave(self.h // self.hk, 1), v.repeat_interleave(self.h // self.hk, 1)
        scores = (q @ kr.transpose(-1, -2)) * self.d ** -0.5
        future = torch.arange(k.shape[-2], device=x.device)[None, :] > torch.arange(old, old + t, device=x.device)[:, None]
        scores = scores.masked_fill(future[None, None], -torch.inf)
        base = (scores.softmax(-1) @ vr).transpose(1, 2).contiguous()
        output = base if controller is None else controller.apply(layer, q, k, v, base, self.d ** -0.5)
        x = x + self.op(output.reshape(b, t, -1))
        x = x + self.mlp(self.norm2(x))
        return x, (k, v)


class _ToyLM(nn.Module):
    def __init__(self, vocab: int = 97, layers: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 32, device='cpu')
        self.layers = nn.ModuleList([_ToyLayer() for _ in range(layers)])
        self.norm = nn.LayerNorm(32, device='cpu')
        self.output = nn.Linear(32, vocab, bias=False, device='cpu')

    def forward(self, ids: Tensor, cache: list[Any] | None = None,
                controller: Controller | None = None) -> tuple[Tensor, list[Any]]:
        x = self.embedding(ids)
        new_cache = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, None if cache is None else cache[i], controller, i)
            new_cache.append(kv)
        return self.output(self.norm(x)), new_cache


class _ToyRunner:
    def __init__(self, model: _ToyLM, controller: Controller):
        self.model, self.controller, self.device = model, controller, next(model.parameters()).device
        self.cache = None

    def reset(self, condition: Condition, branch: str = '') -> None:
        self.cache = None
        self.controller.reset(condition, branch)

    @torch.inference_mode()
    def forward(self, ids: list[int]) -> Tensor:
        output, self.cache = self.model(torch.tensor([ids], device=self.device), self.cache, self.controller)
        return output[0, -1].float().cpu()


@contextmanager
def _manual_attenuation_hooks(projections: list[nn.Module], strength: float):
    """Independent TEST reference: scale the input to W_O, no Controller/QBank."""
    handles = []
    def make_hook():
        seen = False
        def hook(module: nn.Module, args: tuple[Any, ...]):
            nonlocal seen
            x = args[0]
            if not seen:
                seen = True
                return None
            require(x.shape[1] == 1, 'Independent reference expects single-token decode.')
            if strength == 0:
                return None
            return (((1-strength)*x.float()).to(x.dtype),) + args[1:]
        return hook
    try:
        for projection in projections:
            handles.append(projection.register_forward_pre_hook(make_hook()))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _offline_pair_pipeline_test() -> dict[str, Any]:
    """Exercise the REAL suite writer/loop via a toy runner; NOT an HF integration test."""
    import io
    import tempfile
    from contextlib import redirect_stdout
    from types import SimpleNamespace
    from unittest.mock import patch

    class Config(SimpleNamespace):
        def to_dict(self):
            return vars(self).copy()
    config = Config(model_type='llama', num_hidden_layers=3, num_attention_heads=4,
                    num_key_value_heads=2, hidden_size=32, max_position_embeddings=128)
    class Tokenizer:
        chat_template = 'toy-test-template'
        eos_token_id = None
        def encode(self, text, add_special_tokens=True):
            return [3,8,1,27] if add_special_tokens else [5,6,21,9]
        def apply_chat_template(self, *args, **kwargs):
            return [3,8,1,27]
        def decode(self, ids, **kwargs):
            return ' '.join(f't{i}' for i in ids)
    tokenizer = Tokenizer()
    def make_model(*args, **kwargs):
        m = _ToyLM().eval()
        m.config = config
        m.generation_config = SimpleNamespace(eos_token_id=None)
        return m
    hf = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=make_model))
    class Adapter:
        def __init__(self, model, ctrl):
            pass
        def close(self):
            pass
    details = []
    with tempfile.TemporaryDirectory(prefix='qcache_pipeline_') as tmp:
        for mode, families in [('clean','replay,attenuation'), ('stream','attenuation,replay'),
                               ('clean','replay')]:
            path = Path(tmp) / f'{mode}_{families}'
            args = argparse.Namespace(model='random-toy-not-pretrained', device='cpu', dtype='float32',
                families=families, lambdas='0,.2,1', layers='all', prefill=mode, engine='online',
                max_new_tokens=4, max_context=32, chunk=4, temperature=0., top_p=1., seed=42,
                threads=0, trace_every=1, verify_every=2, control_atol=1e-6, save_logits=False,
                out=str(path), prompt='toy', prompt_file=None, chat=False, continuation_file=None,
                local_files_only=True, revision='main')
            with patch.dict(globals(), {'require_hf': lambda: hf,
                                        'HFRunner': lambda m,c,d: _ToyRunner(m,c), 'HFAdapter': Adapter}):
                with torch.inference_mode(False), redirect_stdout(io.StringIO()):
                    run_experiment(args)
                    try:
                        run_experiment(args)
                    except ValueError as exc:
                        assert 'not empty' in str(exc)
                    else:
                        raise AssertionError('Existing output was overwritten')
            report = json.loads((path/'results.json').read_text())
            assert report['status'] == 'completed'
            if 'attenuation' in families:
                assert len(report['pairs']) == 3
                assert report['pairs']['lambda_0']['summary']['max_abs_logit_delta'] == 0
                assert report['pairs']['lambda_0.2']['summary']['fixed_schedule_match']
                assert report['conditions']['attenuation_1']['null_control_passed'] is None
                assert report['conditions']['attenuation_1']['fixed']['qcache']['used_bytes'] == 0
                with (path/'paired_tokens.csv').open(newline='') as f:
                    rows = list(csv.DictReader(f))
                assert len(rows) == 12 and 'attenuation_top1_token' in rows[0]
                assert 'kl_attenuation_to_replay' in rows[0] and 'kl_baseline_to_condition' not in rows[0]
                assert (path/'paired_summary.csv').exists()
            else:
                assert report['pairs'] == {} and not (path/'paired_tokens.csv').exists()
            details.append({'prefill': mode, 'families': families, 'conditions': len(report['conditions']),
                            'pairs': len(report['pairs']), 'report_and_CSV_valid': True})
    return {'runs': details, 'existing_output_protected': True,
            'scope': 'real suite loop/artifact writers with a random toy runner and stub loaders; not HF'}

def self_test(out: Path) -> bool:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    tests: list[dict[str, Any]] = []
    def test(name: str, fn: Callable[[], Any]) -> None:
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                detail = fn()
            tests.append({'name': name, 'status': 'passed', 'seconds': time.perf_counter() - start,
                          'detail': detail})
            print(f'PASS {name}', flush=True)
        except Exception as exc:
            tests.append({'name': name, 'status': 'failed', 'seconds': time.perf_counter() - start,
                          'error': repr(exc), 'traceback': traceback.format_exc()})
            print(f'FAIL {name}: {exc}', flush=True)

    def tensors(seed: int = 1, n: int = 23, dtype: torch.dtype = torch.float64):
        gen = torch.Generator(device='cpu').manual_seed(seed)
        return (torch.randn(1, 6, n, 8, generator=gen, dtype=dtype, device='cpu'),
                torch.randn(1, 2, n, 8, generator=gen, dtype=dtype, device='cpu'),
                torch.randn(1, 2, n, 5, generator=gen, dtype=dtype, device='cpu'))

    def gqa_reference():
        q, k, v = tensors()
        r, z = dense_read(q, k, v, 0.4, torch.float64, 5)
        s = q @ k.repeat_interleave(3, 1).transpose(-1, -2) * 0.4
        reference = s.softmax(-1) @ v.repeat_interleave(3, 1)
        torch.testing.assert_close(r, reference, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(z, s.logsumexp(-1, keepdim=True), atol=1e-12, rtol=1e-12)
        return {'max_abs': float((r - reference).abs().max())}
    test('grouped_attention_matches_explicit_head_repeat', gqa_reference)

    def recurrence():
        err_r, err_z, err_mean = 0., 0., 0.
        steps = 0
        for seed in range(4):
            q, k, v = tensors(seed, n=31)
            for prefix in (1, 4, 9):
                bank = QBank(max_tokens=31, acc_dtype=torch.float64, chunk=5)
                bank.bootstrap(q[..., :prefix, :], k[..., :prefix, :], v[..., :prefix, :], 8 ** -0.5)
                for t in range(prefix, 31):
                    mean, _ = bank.step(q[..., t:t+1, :], k[..., :t+1, :], v[..., :t+1, :], 8 ** -0.5)
                    expected, _ = dense_read(q[..., :t, :], k[..., :t+1, :], v[..., :t+1, :], 8 ** -0.5, torch.float64, 7)
                    torch.testing.assert_close(mean, expected.mean(-2, keepdim=True), atol=1e-12, rtol=1e-12)
                    e = bank.verify(k[..., :t+1, :], v[..., :t+1, :], atol=1e-12, rtol=1e-12)
                    err_r, err_z = max(err_r, e['r_max_abs']), max(err_z, e['logz_max_abs'])
                    err_mean = max(err_mean, float((mean - expected.mean(-2, keepdim=True)).abs().max()))
                    steps += 1
        return {'decode_steps': steps, 'r_max_abs': err_r, 'logz_max_abs': err_z, 'mean_max_abs': err_mean}
    test('online_recurrence_vs_dense_float64_multiple_prefixes', recurrence)

    def float32_long():
        q, k, v = tensors(23, 160, torch.float32)
        bank = QBank(max_tokens=160, chunk=32)
        bank.bootstrap(q[..., :7, :], k[..., :7, :], v[..., :7, :], 0.3535533905932738)
        for t in range(7, 160):
            bank.step(q[..., t:t+1, :], k[..., :t+1, :], v[..., :t+1, :], 0.3535533905932738)
        return bank.verify(k, v, atol=1e-5, rtol=1e-5)
    test('float32_160_position_recurrence', float32_long)

    def mixed_precision_bank():
        results = {}
        for dtype in (torch.float16, torch.bfloat16):
            q, k, v = tensors(7, 33, torch.float32)
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
            bank = QBank(max_tokens=33, chunk=9)
            bank.bootstrap(q[..., :5, :], k[..., :5, :], v[..., :5, :], .4)
            for t in range(5, 33):
                bank.step(q[..., t:t+1, :], k[..., :t+1, :], v[..., :t+1, :], .4)
            assert bank.q.dtype == dtype and bank.r.dtype == torch.float32 and bank.z.dtype == torch.float32
            results[str(dtype)] = bank.verify(k, v, atol=1e-5, rtol=1e-5)
        return results
    test('half_and_bfloat_queries_keep_float32_accumulators', mixed_precision_bank)

    def raw_immutable():
        q, k, v = tensors()
        bank = QBank(max_tokens=23, acc_dtype=torch.float64)
        bank.bootstrap(q[..., :2, :], k[..., :2, :], v[..., :2, :], 0.4)
        saved = bank.q[..., :2, :].clone()
        for t in range(2, 23):
            bank.step(q[..., t:t+1, :], k[..., :t+1, :], v[..., :t+1, :], 0.4)
        assert torch.equal(saved, bank.q[..., :2, :])
        assert torch.equal(q, bank.q[..., :23, :])
        return {'queries_preserved_bitwise': True, 'capacity': bank.capacity}
    test('raw_queries_survive_growth_without_mutation', raw_immutable)

    def bootstrap_full():
        q, k, v = tensors(n=8)
        bank = QBank(max_tokens=8, acc_dtype=torch.float64)
        bank.bootstrap(q, k, v, 0.5)
        bank.verify(k, v, atol=1e-12, rtol=1e-12)
        r_first, _ = dense_read(q[..., :1, :], k, v, 0.5, torch.float64)
        torch.testing.assert_close(bank.r[..., :1, :], r_first, atol=1e-12, rtol=1e-12)
        return {'includes_later_observed_prompt_keys': True}
    test('prefill_bank_initialized_against_entire_observed_prompt', bootstrap_full)

    def exclude_current():
        q, k, v = tensors(n=5)
        bank = QBank(max_tokens=5, acc_dtype=torch.float64)
        bank.bootstrap(q[..., :4, :], k[..., :4, :], v[..., :4, :], 0.8)
        mean, _ = bank.step(q[..., 4:, :], k, v, 0.8)
        allr, _ = dense_read(q, k, v, 0.8, torch.float64)
        torch.testing.assert_close(mean, allr[..., :4, :].mean(-2, keepdim=True), atol=1e-12, rtol=1e-12)
        assert not torch.allclose(mean, allr.mean(-2, keepdim=True))
        return {'current_query_excluded': True}
    test('history_mean_excludes_current_query', exclude_current)

    def stable_logits():
        q = torch.tensor([[[[1.], [1.], [-1.], [1.]]]], dtype=torch.float64, device='cpu')
        k = torch.tensor([[[[10000.], [-10000.], [10000.], [9999.]]]], dtype=torch.float64, device='cpu')
        v = torch.tensor([[[[2.], [-3.], [7.], [4.]]]], dtype=torch.float64, device='cpu')
        bank = QBank(max_tokens=4, acc_dtype=torch.float64)
        bank.bootstrap(q[..., :1, :], k[..., :1, :], v[..., :1, :], 1.)
        for t in range(1, 4):
            bank.step(q[..., t:t+1, :], k[..., :t+1, :], v[..., :t+1, :], 1.)
        return bank.verify(k, v, atol=1e-10, rtol=1e-10)
    test('large_positive_and_negative_scores_stay_finite', stable_logits)

    def direct_control(mode: str):
        q, k, v = tensors(n=5, dtype=torch.float32)
        ctrl = Controller(max_tokens=5, trace_every=1)
        ctrl.reset(Condition(mode, mode, 0.))
        base = torch.randn(1, 4, 6, 5, device='cpu')
        assert ctrl.apply(0, q[..., :4, :], k[..., :4, :], v[..., :4, :], base, .4) is base
        nextbase = torch.randn(1, 1, 6, 5, device='cpu')
        assert ctrl.apply(0, q[..., 4:, :], k, v, nextbase, .4) is nextbase
        if mode == 'store':
            assert ctrl.banks[0].r is None and ctrl.banks[0].z is None
        return {'exact_output_object_bypass': True, 'bank_length': ctrl.banks[0].n}
    test('store_only_is_exact_and_has_no_readout_state', lambda: direct_control('store'))
    test('lambda_zero_updates_bank_but_exactly_bypasses_output', lambda: direct_control('replay'))

    def first_prefill_untouched():
        q, k, v = tensors(n=5, dtype=torch.float32)
        ctrl = Controller(max_tokens=5)
        ctrl.reset(Condition('r1', 'replay', 1.))
        base = torch.randn(1, 5, 6, 5, device='cpu')
        assert ctrl.apply(0, q, k, v, base, .4) is base
    test('clean_prefill_unchanged_even_at_lambda_one', first_prefill_untouched)

    def reject_case(case: str):
        q, k, v = tensors(n=5)
        bank = QBank(max_tokens=4, acc_dtype=torch.float64)
        bank.bootstrap(q[..., :3, :], k[..., :3, :], v[..., :3, :], .4)
        try:
            if case == 'chunk':
                bank.step(q[..., 3:, :], k, v, .4)
            elif case == 'rollback':
                bank.step(q[..., 3:4, :], k[..., :2, :], v[..., :2, :], .4)
            elif case == 'overflow':
                bank.step(q[..., 3:4, :], k[..., :4, :], v[..., :4, :], .4)
                bank.step(q[..., 4:, :], k, v, .4)
            elif case == 'scale':
                bank.step(q[..., 3:4, :], k[..., :4, :], v[..., :4, :], .5)
            else:
                raise AssertionError('bad test')
        except ValueError as exc:
            return {'rejected': str(exc)}
        raise AssertionError(f'{case} was not rejected')
    for name in ('chunk', 'rollback', 'overflow', 'scale'):
        test(f'reject_{name}_rather_than_silently_change_protocol', lambda n=name: reject_case(n))

    torch.manual_seed(901)
    model = _ToyLM().eval()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    prompt, forced = [3, 8, 1, 27, 44, 12], [5, 6, 21, 9, 22, 31, 71, 13]
    def toy_run(condition: Condition, engine: str = 'online', mode: str = 'clean',
                layers: set[int] | None = None):
        ctrl = Controller(max_tokens=64, engine=engine, layers=layers, trace_every=1, verify_every=3)
        runner = _ToyRunner(model, ctrl)
        result = trial(runner, condition, prompt, prefill_mode=mode, max_new=len(forced),
                       seed=17, forced=forced, branch='test')
        return result, ctrl

    def toy_controls():
        baseline, _ = toy_run(Condition('baseline'))
        for spec in [Condition('store', 'store'), Condition('r0', 'replay', 0.)]:
            result, _ = toy_run(spec)
            assert torch.equal(result['logits'], baseline['logits'])
        return {'max_abs_logit_delta': 0., 'decoder_layers': 3}
    test('toy_multilayer_store_and_lambda_zero_match_bitwise', toy_controls)

    def toy_dense(mode: str):
        online, _ = toy_run(Condition('r', 'replay', .2), mode=mode)
        dense, _ = toy_run(Condition('r', 'replay', .2), engine='dense', mode=mode)
        torch.testing.assert_close(online['logits'], dense['logits'], atol=2e-6, rtol=2e-6)
        return {'max_abs_logit_delta': float((online['logits'] - dense['logits']).abs().max())}
    test('toy_feedback_online_vs_dense_clean_prefill', lambda: toy_dense('clean'))
    test('toy_feedback_online_vs_dense_stream_prefill', lambda: toy_dense('stream'))

    def toy_effect():
        b, _ = toy_run(Condition('baseline'))
        r, ctrl = toy_run(Condition('r', 'replay', .2))
        assert torch.equal(b['logits'][0], r['logits'][0])
        difference = float((r['logits'][1:] - b['logits'][1:]).abs().max())
        assert difference > 1e-4 and torch.isfinite(r['logits']).all()
        return {'first_prediction_exactly_unchanged': True, 'later_max_abs_logit_delta': difference,
                'note': 'random toy weights; not evidence of a linguistic effect', 'trace_rows': len(ctrl.trace)}
    test('toy_nonzero_intervention_changes_later_logits', toy_effect)

    def toy_reset():
        ctrl = Controller(max_tokens=64, trace_every=0)
        runner = _ToyRunner(model, ctrl)
        spec = Condition('r', 'replay', .1)
        kwargs = dict(prefill_mode='clean', max_new=8, seed=19, forced=forced)
        a = trial(runner, spec, prompt, **kwargs)
        trial(runner, Condition('store', 'store'), [80, 5, 5], **kwargs)
        b = trial(runner, spec, prompt, **kwargs)
        assert torch.equal(a['logits'], b['logits'])
        return {'trial_isolation_bitwise': True}
    test('toy_state_reset_prevents_cross_trial_contamination', toy_reset)

    def toy_causal():
        a = torch.tensor([[3, 5, 8, 9, 20, 21]], device='cpu')
        b = torch.tensor([[3, 5, 8, 9, 78, 79]], device='cpu')
        outa, _ = model(a)
        outb, _ = model(b)
        assert torch.equal(outa[:, :4], outb[:, :4])
        def stream(ids):
            ctrl = Controller(max_tokens=16, trace_every=0)
            ctrl.reset(Condition('r', 'replay', .3))
            cache, outputs = None, []
            for token in ids[0].tolist():
                logits, cache = model(torch.tensor([[token]], device='cpu'), cache, ctrl)
                outputs.append(logits)
            return torch.cat(outputs, 1)
        sa, sb = stream(a), stream(b)
        assert torch.equal(sa[:, :4], sb[:, :4])
        return {'future_token_changes_do_not_affect_prior_outputs': True}
    test('toy_native_and_stream_intervention_are_prefix_causal', toy_causal)

    def selected_layers():
        _, ctrl = toy_run(Condition('r', 'replay', .2), layers={1})
        assert set(ctrl.banks) == {1}
        assert all(row['layer'] == 1 for row in ctrl.trace)
    test('only_explicitly_selected_layers_allocate_query_banks', selected_layers)

    def null_stream():
        a, _ = toy_run(Condition('baseline'), mode='clean')
        b, _ = toy_run(Condition('baseline'), mode='stream')
        torch.testing.assert_close(a['logits'], b['logits'], atol=2e-6, rtol=2e-6)
        return {'max_abs_logit_delta': float((a['logits'] - b['logits']).abs().max())}
    test('toy_native_batch_and_stream_prefill_agree_numerically', null_stream)

    def weights_unchanged():
        assert all(torch.equal(before[name], param) for name, param in model.named_parameters())
        return {'all_parameters_bitwise_unchanged': True}
    test('toy_model_weights_unchanged_after_all_trials', weights_unchanged)

    def callback_contract():
        from types import SimpleNamespace
        q, k, v = tensors(12, 4, torch.float32)
        ctrl = Controller(max_tokens=8, trace_every=0)
        ctrl.reset(Condition('replay', 'replay', .2))
        seen_masks = []
        def native(module, query, key, value, mask, **kwargs):
            seen_masks.append(mask)
            scores = gqa_scores(query, key, kwargs['scaling']) + mask
            probs = scores.softmax(-1)
            return gqa_values(probs, value).transpose(1, 2).contiguous(), probs
        module = SimpleNamespace(training=False, layer_idx=0,
                                 _qcache_lab_original_eager=native, _qcache_lab_controller=ctrl)
        mask = torch.zeros(1, 1, 3, 3, device='cpu').masked_fill(torch.ones(3, 3, dtype=torch.bool, device='cpu').triu(1), -torch.inf)
        out, weights = hf_attention(module, q[..., :3, :], k[..., :3, :], v[..., :3, :], mask, scaling=.4)
        assert seen_masks[-1] is mask and weights is not None
        mask2 = torch.zeros(1, 1, 1, 4, device='cpu')
        _, mixed_weights = hf_attention(module, q[..., 3:, :], k, v, mask2, scaling=.4)
        assert seen_masks[-1] is mask2 and mixed_weights is None
        try:
            hf_attention(module, q, k, v, None, scaling=.4)
        except ValueError:
            return {'native_mask_preserved': True, 'missing_prefill_mask_rejected': True,
                    'note': 'Callback contract tested with a stub native function, NOT HF registration.'}
        raise AssertionError('Missing prefill mask not rejected')
    test('callback_contract_preserves_mask_and_rejects_unmasked_prefill', callback_contract)

    def metric_test():
        torch.manual_seed(73)
        logits = torch.randn(5, 97, device='cpu')
        rows, summary = compare_logits(logits, logits.clone(), [0, 1, 2, 3, 4], 'self')
        assert summary['max_abs_logit_delta'] == 0 and summary['mean_kl'] == 0
        assert summary['mean_total_variation'] == 0 and summary['top1_agreement'] == 1
        assert all(row['nll_delta'] == 0 for row in rows)
    test('fixed_token_comparison_metrics_have_exact_zero_control', metric_test)

    test('metric_boundary_current_device_hooks', _cpu_metric_check)
    test('metric_boundary_non_cpu_default_context', _non_cpu_default_metric_check)
    test('metric_index_factories_explicit_cpu', _explicit_metric_factories_check)
    test('cpu_sampling_non_cpu_default_context', _non_cpu_default_sampler_check)

    # Attenuation uses no Q bank and must match replay's eligibility schedule.
    def attenuation_formula(dtype: torch.dtype, strength: float):
        q, k, v = tensors(87, n=7, dtype=torch.float32)
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        originals = [x.clone() for x in (q, k, v)]
        ctrl = Controller(max_tokens=7, trace_every=1)
        ctrl.reset(Condition('a', 'attenuation', strength))
        init = torch.randn(1, 3, 6, 5, device='cpu').to(dtype)
        assert ctrl.apply(0, q[..., :3, :], k[..., :3, :], v[..., :3, :], init, .4) is init
        for t in range(3, 7):
            base = torch.randn(1, 1, 6, 5, device='cpu').to(dtype)
            saved = base.clone()
            out_tensor = ctrl.apply(0, q[..., t:t+1, :], k[..., :t+1, :], v[..., :t+1, :], base, .4)
            assert torch.equal(base, saved)
            if strength == 0:
                assert out_tensor is base
            else:
                expected = ((1-strength)*base.float()).to(dtype)
                assert torch.equal(out_tensor, expected)
            if strength == 1:
                assert int(torch.count_nonzero(out_tensor)) == 0
        assert not ctrl.banks and ctrl.memory()['used_bytes'] == 0
        assert ctrl.schedule()['0']['eligible_count'] == 4
        assert all(torch.equal(a, b) for a, b in zip(originals, (q, k, v)))
        assert all(row['past_queries'] == 0 and row['gate_max'] is None and
                   row['history_current_cosine'] is None for row in ctrl.trace)
        return {'dtype': str(dtype), 'lambda': strength, 'q_bank_allocated': False,
                'qkv_and_base_unchanged': True, 'eligible_positions': ctrl.eligible_positions[0]}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for strength in (0., .2, 1.):
            test(f'attenuation_formula_{dtype}_{strength:g}',
                 lambda d=dtype, v=strength: attenuation_formula(d, v))

    def attenuation_no_hidden_readout():
        from unittest.mock import patch
        with patch.object(QBank, '__init__', side_effect=AssertionError('QBank must not be allocated')):
            r, ctrl = toy_run(Condition('a', 'attenuation', .3))
        assert torch.isfinite(r['logits']).all() and not ctrl.banks
        return {'QBank_constructor_never_called': True}
    test('attenuation_never_calls_qbank', attenuation_no_hidden_readout)

    def independent_attenuation(mode: str):
        results = {}
        for strength in (0., .2, 1.):
            for selected in (None, {1}):
                active, ctrl = toy_run(Condition('a', 'attenuation', strength), mode=mode, layers=selected)
                layers_list = model.layers if selected is None else [model.layers[i] for i in sorted(selected)]
                with _manual_attenuation_hooks([layer.op for layer in layers_list], strength):
                    manual, _ = toy_run(Condition('baseline'), mode=mode)
                assert torch.equal(active['logits'], manual['logits'])
                assert set(ctrl.attenuation_seen) == (set(range(3)) if selected is None else selected)
                results[f'{strength:g}:{selected}'] = float((active['logits']-manual['logits']).abs().max())
        return {'independent_projection_pre_hook_max_abs_errors': results}
    test('toy_attenuation_vs_independent_WO_hook_clean', lambda: independent_attenuation('clean'))
    test('toy_attenuation_vs_independent_WO_hook_stream', lambda: independent_attenuation('stream'))

    def schedule_match(mode: str):
        for input_ids in ([5], prompt):
            schedules = []
            for family in ('replay', 'attenuation'):
                ctrl = Controller(max_tokens=64, trace_every=1)
                result = trial(_ToyRunner(model, ctrl), Condition(family, family, .3), input_ids,
                               prefill_mode=mode, max_new=len(forced), forced=forced, seed=42)
                schedules.append(result['intervention_schedule'])
                first = len(input_ids) if mode == 'clean' else 1
                assert all(row['first_position'] == first for row in schedules[-1].values())
            assert schedules[0] == schedules[1]
        return {'single_and_multi_token_prompt_schedules_match': True, 'prefill': mode}
    test('matched_intervention_schedule_clean', lambda: schedule_match('clean'))
    test('matched_intervention_schedule_stream', lambda: schedule_match('stream'))

    def atten_invalid(kind: str):
        q, k, v = tensors(33, n=5, dtype=torch.float32)
        ctrl = Controller(max_tokens=4, trace_every=0)
        ctrl.reset(Condition('a', 'attenuation', .3))
        ctrl.apply(0, q[..., :3, :], k[..., :3, :], v[..., :3, :], torch.zeros(1, 3, 6, 5, device='cpu'), .4)
        base = torch.zeros(1, 1, 6, 5, device='cpu')
        try:
            if kind == 'rollback':
                ctrl.apply(0, q[..., 3:4, :], k[..., :2, :], v[..., :2, :], base, .4)
            elif kind == 'overflow':
                ctrl.apply(0, q[..., 3:4, :], k[..., :4, :], v[..., :4, :], base, .4)
                ctrl.apply(0, q[..., 4:, :], k, v, base, .4)
            elif kind == 'scale':
                ctrl.apply(0, q[..., 3:4, :], k[..., :4, :], v[..., :4, :], base, .5)
            elif kind == 'dtype':
                ctrl.apply(0, q[..., 3:4, :].half(), k[..., :4, :], v[..., :4, :], base, .4)
            elif kind == 'chunk':
                ctrl.apply(0, q[..., 3:, :], k, v, torch.zeros(1, 2, 6, 5, device='cpu'), .4)
            else:
                raise AssertionError('unknown case')
        except ValueError as exc:
            return {'rejected': str(exc)}
        raise AssertionError('Invalid attenuation state was accepted')
    for kind in ('rollback', 'overflow', 'scale', 'dtype', 'chunk'):
        test('attenuation_reject_' + kind, lambda k=kind: atten_invalid(k))

    def atten_reset():
        ctrl = Controller(max_tokens=64, trace_every=1)
        runner = _ToyRunner(model, ctrl)
        kwargs = dict(prefill_mode='stream', max_new=len(forced), forced=forced, seed=21)
        a = trial(runner, Condition('a', 'attenuation', .3), prompt, **kwargs)
        trial(runner, Condition('r', 'replay', .5), [1, 7, 3], **kwargs)
        b = trial(runner, Condition('a', 'attenuation', .3), prompt, **kwargs)
        assert torch.equal(a['logits'], b['logits'])
        assert a['intervention_schedule'] == b['intervention_schedule']
        assert not ctrl.banks
        return {'cross_family_trial_isolation': True}
    test('attenuation_reset_across_replay_trials', atten_reset)

    def pair_metrics():
        a = torch.tensor([[1., 0., -1.], [1., 2., 0.]], device='cpu').repeat(6, 1)
        r = a.clone()
        r[1, 0] = 3.
        targets = [0, 1]*6
        rows, summary = compare_pair(a, r, targets, .3)
        def probs(vals):
            ex = [math.exp(x) for x in vals]
            return [x/sum(ex) for x in ex]
        p, q = probs(a[1].tolist()), probs(r[1].tolist())
        m = [(x+y)/2 for x, y in zip(p, q)]
        kl = sum(x*math.log(x/y) for x, y in zip(p, q))
        js = .5*sum(x*math.log(x/z)+y*math.log(y/z) for x, y, z in zip(p, q, m))
        assert math.isclose(rows[1]['kl_attenuation_to_replay'], kl, abs_tol=1e-12)
        assert math.isclose(rows[1]['js_divergence'], js, abs_tol=1e-12)
        assert summary['top1_disagreements'] == 1 and summary['first_top1_difference_step'] == 1
        reversed_rows, reversed_summary = compare_pair(r, a, targets, .3)
        assert math.isclose(summary['mean_js'], reversed_summary['mean_js'], abs_tol=1e-15)
        assert math.isclose(summary['mean_kl_attenuation_to_replay'],
                            reversed_summary['mean_kl_replay_to_attenuation'], abs_tol=1e-15)
        _, zero = compare_pair(a, a, targets, 0.)
        assert zero['max_abs_logit_delta'] == 0 and zero['mean_kl_attenuation_to_replay'] == 0
        assert zero['first_logit_difference_step'] is None and zero['first_top1_difference_step'] is None
        with torch.device('meta'):
            meta_rows, meta_summary = compare_pair(a, r, targets, .3)
        assert meta_rows == rows and meta_summary == summary
        return {'scalar_KL_and_JS_verified': True, 'meta_default_verified': True,
                'chunk_boundary_covered': True, 'direct_mean_js': summary['mean_js']}
    test('direct_pair_metrics_scalar_symmetry_and_non_cpu_default', pair_metrics)

    def same_base_distance_is_not_pair_distance():
        base = torch.zeros(1, 2, device='cpu')
        a = torch.tensor([[2., 0.]], device='cpu')
        r = torch.tensor([[0., 2.]], device='cpu')
        _, bs_a = compare_logits(base, a, [0], 'a')
        _, bs_r = compare_logits(base, r, [0], 'r')
        assert bs_a['mean_kl'] == bs_r['mean_kl']
        _, pair = compare_pair(a, r, [0], .5)
        assert pair['mean_js'] > .1 and pair['top1_disagreements'] == 1
        return {'equal_baseline_KL': bs_a['mean_kl'], 'nonzero_direct_pair_JS': pair['mean_js']}
    test('direct_pair_distance_not_difference_of_baseline_distances', same_base_distance_is_not_pair_distance)

    def grid_checks():
        specs = build_conditions('0,.3,1,.3')
        assert [x.name for x in specs] == ['baseline','store','replay_0','attenuation_0',
                                         'replay_0.3','attenuation_0.3','replay_1','attenuation_1']
        assert sum(s.is_null for s in specs) == 4
        assert [x.mode for x in build_conditions('.3','replay')] == ['baseline','store','replay','replay']
        for vals, families in [('nan','replay'), ('inf','attenuation'), ('-0.1','attenuation'),
                               ('1.1','replay'), ('.2','bad'), ('.2','replay,replay'),
                               ('.12345671,.12345672','replay')]:
            try:
                build_conditions(vals, families)
            except ValueError:
                pass
            else:
                raise AssertionError((vals, families))
        return {'zero_controls_first': True, 'name_collisions_and_invalid_modes_rejected': True}
    test('condition_grid_and_null_classification', grid_checks)

    def prefix_and_stop():
        assert free_prefix_comparison([1, 2], [1, 2])['free_first_difference_step'] is None
        assert free_prefix_comparison([1, 2], [1, 3])['free_difference_kind'] == 'token'
        assert free_prefix_comparison([1], [1, 2])['free_difference_kind'] == 'length'
        runner = _ToyRunner(model, Controller(max_tokens=64, trace_every=0))
        spec = Condition('baseline')
        ordinary = trial(runner, spec, prompt, prefill_mode='clean', max_new=2, seed=42)
        early = trial(runner, spec, prompt, prefill_mode='clean', max_new=2, seed=42,
                      eos_ids={ordinary['token_ids'][0]})
        fixed = trial(runner, spec, prompt, prefill_mode='clean', max_new=2, seed=42,
                      eos_ids={ordinary['token_ids'][0]}, forced=ordinary['token_ids'])
        assert ordinary['stop_reason'] == 'max_new_tokens'
        assert early['stop_reason'] == 'eos' and len(early['token_ids']) == 1
        assert fixed['stop_reason'] == 'fixed_length' and len(fixed['token_ids']) == 2
        return {'eos_vs_token_limit_vs_forced_length': True}
    test('free_prefix_and_explicit_stop_reason', prefix_and_stop)

    test('offline_toy_end_to_end_paired_artifacts', _offline_pair_pipeline_test)

    passed = sum(t['status'] == 'passed' for t in tests)
    report = {'schema': 'qcache.selftest.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'environment': environment(),
              'scope': 'PyTorch mathematical core and independent random-weight RoPE/GQA toy decoder; NOT HF or a pretrained model',
              'passed': passed, 'failed': len(tests) - passed, 'tests': tests}
    json_write(out, report)
    print(f'{passed}/{len(tests)} passed; report: {out}', flush=True)
    return passed == len(tests)


def hf_smoke(out: Path, device_name: str = 'cpu') -> bool:
    """No checkpoint download. Verify the actual HF adapter with random tiny models."""
    report: dict[str, Any] = {'schema': 'qcache.hf_smoke.v1', 'environment': environment(),
                              'scope': 'Actual HF Llama/Mistral/Qwen2 classes with random weights; NOT pretrained language behavior',
                              'models': []}
    try:
        hf = require_hf()
    except (ValueError, ImportError) as exc:
        report.update(status='not_run', reason=str(exc))
        json_write(out, report)
        print(f'HF smoke NOT RUN: {exc}', file=sys.stderr)
        return False
    device = choose_device(device_name)
    try:
        for mt, config_cls, model_cls in [
            ('llama', hf.LlamaConfig, hf.LlamaForCausalLM),
            ('mistral', hf.MistralConfig, hf.MistralForCausalLM),
            ('qwen2', hf.Qwen2Config, hf.Qwen2ForCausalLM),
        ]:
            torch.manual_seed(191)
            opts = dict(vocab_size=97, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=128, attention_dropout=0.0,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0)
            if mt == 'mistral':
                opts['sliding_window'] = None
            if mt == 'qwen2':
                opts.update(use_sliding_window=False, sliding_window=None)
            config = config_cls(**opts)
            config._attn_implementation = 'eager'
            with torch.device('cpu'):
                model = model_cls(config)
            model = model.to(device).eval()
            before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            ctrl = Controller(max_tokens=64, trace_every=0, verify_every=2)
            runner = HFRunner(model, ctrl, device)
            prompt, forced = [1, 8, 23, 11, 5], [7, 21, 3, 19, 44, 37]
            kwargs = dict(prefill_mode='clean', max_new=6, seed=0, forced=forced)
            with torch.inference_mode():
                native = trial(runner, Condition('native'), prompt, **kwargs)
                x = torch.tensor([[1, 4, 8, 12, 3]], device=device)
                y = torch.tensor([[1, 4, 8, 55, 62]], device=device)
                nx = model(x, use_cache=False).logits[:, :3].cpu()
                ny = model(y, use_cache=False).logits[:, :3].cpu()
                torch.testing.assert_close(nx, ny, atol=1e-6, rtol=1e-6)
                with HFAdapter(model, ctrl):
                    native_delta = 0.0
                    for spec in [Condition('baseline'), Condition('store', 'store'), Condition('r0', 'replay', 0.)]:
                        check = trial(runner, spec, prompt, **kwargs)
                        torch.testing.assert_close(native['logits'], check['logits'], atol=1e-6, rtol=1e-6)
                        _, metric_summary = compare_logits(native['logits'], check['logits'], forced, spec.name)
                        require(metric_summary['max_abs_logit_delta'] <= 1e-6,
                                'HF smoke CPU metric control failed.')
                        native_delta = max(native_delta, float((native['logits'] - check['logits']).abs().max()))
                    active = trial(runner, Condition('r', 'replay', .2), prompt, **kwargs)
                    ctrl.engine = 'dense'
                    dense = trial(runner, Condition('r', 'replay', .2), prompt, **kwargs)
                    torch.testing.assert_close(active['logits'], dense['logits'], atol=2e-5, rtol=2e-5)
                    assert torch.equal(active['logits'][0], native['logits'][0])
                    changed = float((active['logits'][1:] - native['logits'][1:]).abs().max())
                    assert changed > 0
                    compare_logits(native['logits'], active['logits'], forced, 'active_metric_smoke')
                    ctrl.engine = 'online'
                    stream = trial(runner, Condition('s', 'replay', .2), prompt,
                                   **dict(kwargs, prefill_mode='stream'))
                    assert torch.isfinite(stream['logits']).all()
                    attenuation_checks = []
                    for pre_mode in ('clean', 'stream'):
                        for strength in (0., .2, 1.):
                            opts = dict(kwargs, prefill_mode=pre_mode)
                            att = trial(runner, Condition('a', 'attenuation', strength), prompt, **opts)
                            assert att['qcache']['used_bytes'] == 0
                            with _manual_attenuation_hooks(
                                    [layer.self_attn.o_proj for layer in model.model.layers], strength):
                                independent = trial(runner, Condition('manual'), prompt, **opts)
                            torch.testing.assert_close(att['logits'], independent['logits'], atol=1e-6, rtol=1e-6)
                            rep = trial(runner, Condition('r', 'replay', strength), prompt, **opts)
                            assert att['intervention_schedule'] == rep['intervention_schedule']
                            _, paired = compare_pair(att['logits'], rep['logits'], forced, strength)
                            if strength == 0:
                                assert paired['max_abs_logit_delta'] == 0
                            attenuation_checks.append({
                                'prefill': pre_mode, 'lambda': strength,
                                'independent_WO_hook_max_abs': float((att['logits']-independent['logits']).abs().max()),
                                'schedule_match': True, 'direct_pair_JS': paired['mean_js']})
                    ctrl.reset(Condition('baseline'))
                    px = model(x, use_cache=False).logits[:, :3].cpu()
                    py = model(y, use_cache=False).logits[:, :3].cpu()
                    torch.testing.assert_close(px, py, atol=1e-6, rtol=1e-6)
                assert all(torch.equal(before[n], p.detach().cpu()) for n, p in model.named_parameters())
                restored = trial(runner, Condition('restored'), prompt, **kwargs)
                torch.testing.assert_close(restored['logits'], native['logits'], atol=1e-6, rtol=1e-6)
            item = {'model_type': mt, 'status': 'passed', 'native_vs_controls_max_abs': native_delta,
                    'online_vs_dense_max_abs': float((active['logits'] - dense['logits']).abs().max()),
                    'active_vs_native_later_max_abs': changed, 'weights_unchanged': True,
                    'causal_mask_and_backend_restoration_checked': True,
                    'cpu_fixed_token_metrics_exercised': True,
                    'attenuation_and_pair_checks': attenuation_checks}
            report['models'].append(item)
            print(f'HF PASS {mt}: {item}', flush=True)
            del runner, ctrl, model
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
    json_write(out, report)
    return report['status'] == 'passed'


def estimate_extra_bytes(config: Any, layers: int, tokens: int, query_bytes: int = 2) -> dict[str, int]:
    h = config.num_attention_heads
    d = getattr(config, 'head_dim', None) or config.hidden_size // h
    # readout is D_v=D for supported architectures; accumulators are float32.
    return {'q_only_bytes': layers * tokens * h * d * query_bytes,
            'q_readout_logz_bytes': layers * tokens * h * (d * query_bytes + d * 4 + 4),
            'ordinary_kv_bytes': config.num_hidden_layers * tokens * config.num_key_value_heads * d * query_bytes * 2}


def write_suite_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ['# QCache Lab run', '', f"Status: {report.get('status')}",
             '', 'Fixed-token metrics use the same token-ID continuation in every condition.',
             'Free generations are separate trajectories, not matched-prefix comparisons.',
             'NLL on a baseline-generated continuation is NOT a held-out language-quality benchmark.', '',
             '| Condition | Mean KL(base || condition) | Top-1 agreement | Max logit delta |',
             '|---|---:|---:|---:|']
    for name, result in report.get('conditions', {}).items():
        s = result['fixed_summary']
        lines.append(f"| {name} | {s['mean_kl']:.8g} | {s['top1_agreement']:.5f} | {s['max_abs_logit_delta']:.8g} |")
    for name, result in report.get('conditions', {}).items():
        lines.extend(['', f'## {name}', '', 'Free generation:', '',
                      '```text', result['free']['text'].replace('```', "'''"), '```'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_experiment(args: argparse.Namespace) -> None:
    hf = require_hf()
    device = choose_device(args.device)
    require(args.max_new_tokens >= 2, 'Use at least 2 new tokens: clean prefill leaves the first unchanged.')
    require(args.max_context > 0 and args.chunk > 0, 'Context/chunk sizes must be positive.')
    require(args.temperature >= 0 and math.isfinite(args.temperature), 'Invalid temperature.')
    require(0 < args.top_p <= 1, 'top_p must lie in (0,1].')
    require(args.control_atol >= 0 and math.isfinite(args.control_atol), 'Invalid control tolerance.')
    require(args.trace_every >= 0 and args.verify_every >= 0, 'Trace/verification intervals cannot be negative.')
    specs = build_conditions(args.lambdas, args.families)
    out = Path(args.out)
    require(not out.exists() or not any(out.iterdir()),
            f'Output directory is not empty: {out}. Choose a new path; existing experiments are not overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        'schema': 'qcache.run.v2', 'status': 'started', 'environment': environment(),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'arguments': vars(args),
        'derived_from': {'script_version': '0.1.1',
                         'script_sha256': 'e487d201d4f33fa4de4de28f136abbec246d0414ecb13bd8cc311e953d88e891'},
        'protocol': {
            'query_space': 'post-RoPE, original positions', 'selection': 'all positions, all heads in selected layers',
            'pool': 'uniform mean of past query readouts, excludes current query',
            'kv': 'append-only full context, no rewrite, eviction or unknown future tokens',
            'prefill': args.prefill, 'accumulator_dtype': 'float32',
            'readout_engine': args.engine, 'weights': 'frozen, no optimizer',
            'run_isolation': 'fresh KV and Q banks for every trial',
            'base_attention': 'original HF eager with its causal mask',
            'intervention_site': 'attention output before W_O; residual, MLP and projection bias are preserved',
            'replay_formula': '(1-lambda)*ordinary_out + lambda*mean(past_Q_readouts)',
            'attenuation_formula': '(1-lambda)*ordinary_out; no Q bank or readout',
            'matched_schedule': 'first forward bypassed; then all single-token steps in selected layers',
            'pair_scope': 'same token prefixes; separate evolving KV and hidden states for each condition',
            'not_matched': ['compute budget', 'output perturbation norm', 'injected signal content'],
            'metric_units': 'natural-log KL/JS (nats); metrics are not language-quality benchmarks',
            'condition_order': [s.name for s in specs],
        },
    }
    json_write(out / 'manifest.json', manifest)
    report: dict[str, Any] = {'schema': 'qcache.results.v2', 'status': 'started', 'conditions': {}, 'pairs': {}}
    trace_file = None
    adapter = None
    pair_rows: list[dict[str, Any]] = []
    try:
        manifest['metric_device_check'] = _cpu_metric_check()
        snapshot = manifest['environment']['device_runtime']
        print(f'[QCache] model device={device}; '
              f'implicit factory={snapshot.get("implicit_empty", "unknown")}; metrics=cpu', flush=True)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if args.threads:
            torch.set_num_threads(args.threads)
        load_opts = dict(local_files_only=args.local_files_only, trust_remote_code=False, revision=args.revision)
        config = hf.AutoConfig.from_pretrained(args.model, **load_opts)
        _legacy_validate_config(config)
        layers = parse_layers(args.layers, config.num_hidden_layers)
        dtype_name = args.dtype
        if dtype_name == 'auto':
            dtype_name = 'float32' if device.type == 'cpu' else (
                'bfloat16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'float16')
        dtype = getattr(torch, dtype_name)
        require(not (device.type == 'mps' and dtype == torch.bfloat16),
                'Use float16 or float32 for the initial MPS validation.')
        tokenizer = hf.AutoTokenizer.from_pretrained(args.model, **load_opts)
        prompt_text = Path(args.prompt_file).read_text(encoding='utf-8') if args.prompt_file else args.prompt
        require(bool(prompt_text.strip()), 'Prompt is empty.')
        if args.chat:
            require(bool(tokenizer.chat_template), 'This tokenizer has no chat template. Omit --chat.')
            prompt_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt_text}],
                                                       tokenize=True, add_generation_prompt=True)
        else:
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        require(isinstance(prompt_ids, list) and len(prompt_ids) > 0, 'Unexpected/empty tokenizer output.')
        fixed_explicit = None
        if args.continuation_file:
            continuation = Path(args.continuation_file).read_text(encoding='utf-8')
            fixed_explicit = tokenizer.encode(continuation, add_special_tokens=False)
            require(bool(fixed_explicit), 'Fixed continuation is empty.')
        longest = max(args.max_new_tokens, len(fixed_explicit) if fixed_explicit else 0)
        total = len(prompt_ids) + longest
        require(total <= args.max_context,
                f'Prompt ({len(prompt_ids)}) + continuation ({longest}) exceeds --max-context {args.max_context}. No truncation occurred.')
        require(total <= config.max_position_embeddings,
                'Requested sequence exceeds the checkpoint position limit; length extrapolation is a different experiment.')
        memory = estimate_extra_bytes(config, len(layers), total, torch.tensor([], dtype=dtype, device='cpu').element_size())
        print(f'Loading {args.model}; device={device}, dtype={dtype_name}, selected_layers={sorted(layers)}', flush=True)
        print(f'Prompt={len(prompt_ids)} tokens. Extra replay state at {total} positions: '
              f'{memory["q_readout_logz_bytes"] / 2**20:.1f} MiB (used tensors, excludes capacity slack and temporaries).', flush=True)
        # Only safe tensor checkpoint loading; no remote model Python, no quantization.
        model = hf.AutoModelForCausalLM.from_pretrained(
            args.model, config=config, attn_implementation='eager', dtype=dtype,
            use_safetensors=True, **load_opts,
        ).to(device).eval()
        model.requires_grad_(False)
        placement = {'parameter_devices': sorted({str(p.device) for p in model.parameters()}),
                     'buffer_devices': sorted({str(b.device) for b in model.buffers()})}
        mismatches = [(name, str(t.device)) for name, t in
                      list(model.named_parameters()) + list(model.named_buffers())
                      if t.device.type != device.type or
                      (device.type == 'cuda' and t.device.index !=
                       (device.index if device.index is not None else torch.cuda.current_device()))]
        require(not mismatches, f'Model device mismatch after .to({device}): {mismatches[:8]}. '
                'Check external load/device hooks; no silent relocation is performed.')
        manifest['model_placement'] = placement
        print(f'[QCache] loaded parameters={placement["parameter_devices"]}; '
              f'buffers={placement["buffer_devices"]}', flush=True)
        parameter_versions = {name: p._version for name, p in model.named_parameters()}
        ctrl = Controller(layers=layers, max_tokens=args.max_context, engine=args.engine,
                          chunk=args.chunk, trace_every=args.trace_every, verify_every=args.verify_every)
        runner = HFRunner(model, ctrl, device)
        trace_file = (out / 'attention_trace.csv').open('w', encoding='utf-8', newline='')
        trace_writer = csv.DictWriter(trace_file, fieldnames=TRACE_FIELDS)
        trace_writer.writeheader()
        ctrl.trace_sink = trace_writer.writerow
        eos_raw = getattr(model.generation_config, 'eos_token_id', None)
        eos_raw = eos_raw if eos_raw is not None else tokenizer.eos_token_id
        eos_ids = set(eos_raw if isinstance(eos_raw, list) else ([] if eos_raw is None else [eos_raw]))
        manifest.update(model_config=config.to_dict(), resolved_revision=getattr(config, '_commit_hash', None),
                        device=str(device), dtype=dtype_name, selected_layers=sorted(layers),
                        prompt_text=prompt_text, prompt_token_ids=prompt_ids, memory_estimate=memory,
                        eos_token_ids=sorted(eos_ids),
                        sampling={'temperature': args.temperature, 'top_p': args.top_p, 'seed': args.seed,
                                  'sampler_device': 'cpu', 'sampler_reset_each_trial': True})
        json_write(out / 'manifest.json', manifest)
        gen_kwargs = dict(prefill_mode=args.prefill, max_new=args.max_new_tokens, seed=args.seed,
                          temperature=args.temperature, top_p=args.top_p, eos_ids=eos_ids)
        with torch.inference_mode():
            print('Running native baseline before adapter attachment', flush=True)
            # Native baseline is collected BEFORE installing the custom backend.
            baseline_free = trial(runner, specs[0], prompt_ids, **gen_kwargs, branch='free_native')
            fixed_ids = fixed_explicit if fixed_explicit is not None else baseline_free['token_ids']
            baseline_fixed = (trial(runner, specs[0], prompt_ids, **gen_kwargs,
                                    forced=fixed_ids, branch='fixed_native')
                              if fixed_explicit is not None else baseline_free)
            require(len(fixed_ids) >= 2 or args.prefill == 'stream',
                    'Baseline ended at the first token. With clean prefill no intervention is measurable; try another prompt.')
            manifest.update(fixed_token_ids=fixed_ids,
                            fixed_source='continuation_file_tokenized_separately' if fixed_explicit is not None else 'native_baseline_free_generation',
                            fixed_text=tokenizer.decode(fixed_ids, skip_special_tokens=False))
            json_write(out / 'manifest.json', manifest)
            adapter = HFAdapter(model, ctrl)
            all_rows: list[dict[str, Any]] = []
            pending_pair: tuple[Condition, Tensor] | None = None
            paired_enabled = {s.mode for s in specs} >= {'replay', 'attenuation'}
            for spec in specs:
                print(f'Running {spec.name}: fixed token replay + independent free generation', flush=True)
                # Even baseline is re-run through the registered backend to catch mask changes.
                fixed = trial(runner, spec, prompt_ids, **gen_kwargs, forced=fixed_ids, branch='fixed')
                free = trial(runner, spec, prompt_ids, **gen_kwargs, branch='free')
                print(f'  {spec.name}: scoring fixed-token logits on CPU', flush=True)
                rows, summary = compare_logits(baseline_fixed['logits'], fixed['logits'], fixed_ids, spec.name)
                all_rows.extend(rows)
                null = spec.is_null
                control_pass = summary['max_abs_logit_delta'] <= args.control_atol if null else None
                same_free = free['token_ids'] == baseline_free['token_ids']
                result = {
                    'mode': spec.mode, 'lambda': spec.strength, 'fixed_summary': summary,
                    'null_control_passed': control_pass,
                    'free_token_ids_match_native': same_free,
                    'fixed': {'token_ids': fixed['token_ids'], 'elapsed_seconds': fixed['elapsed_seconds'], 'qcache': fixed['qcache'],
                              'intervention_schedule': fixed['intervention_schedule']},
                    'free': {'text': tokenizer.decode(free['token_ids'], skip_special_tokens=True),
                             'text_with_special_tokens': tokenizer.decode(free['token_ids'], skip_special_tokens=False),
                             'token_ids': free['token_ids'], 'elapsed_seconds': free['elapsed_seconds'],
                             'qcache': free['qcache'], 'repetition': repetition_stats(free['token_ids']),
                             'stop_reason': free['stop_reason'],
                             'intervention_schedule': free['intervention_schedule']},
                }
                report['conditions'][spec.name] = result
                if paired_enabled and spec.mode in {'replay', 'attenuation'}:
                    if pending_pair is None:
                        pending_pair = (spec, fixed['logits'])
                    else:
                        other_spec, other_logits = pending_pair
                        require(other_spec.strength == spec.strength and other_spec.mode != spec.mode,
                                'Pair ordering mismatch; refusing an unpaired comparison.')
                        by_mode = {spec.mode: (spec, fixed['logits']),
                                   other_spec.mode: (other_spec, other_logits)}
                        a_spec, a_logits = by_mode['attenuation']
                        r_spec, r_logits = by_mode['replay']
                        a_result, r_result = report['conditions'][a_spec.name], report['conditions'][r_spec.name]
                        require(a_result['fixed']['intervention_schedule'] == r_result['fixed']['intervention_schedule'],
                                'Replay and attenuation intervention schedules differ.')
                        new_rows, pair_summary = compare_pair(a_logits, r_logits, fixed_ids, spec.strength, tokenizer)
                        pair_summary.update(free_prefix_comparison(a_result['free']['token_ids'], r_result['free']['token_ids']))
                        pair_summary.update(attenuation_stop_reason=a_result['free']['stop_reason'],
                                            replay_stop_reason=r_result['free']['stop_reason'],
                                            fixed_schedule_match=True)
                        report['pairs'][f'lambda_{spec.strength:g}'] = {
                            'attenuation_condition': a_spec.name, 'replay_condition': r_spec.name,
                            'summary': pair_summary,
                        }
                        pair_rows.extend(new_rows)
                        write_pair_outputs(out, report, pair_rows)
                        print(f'  PAIR JS={pair_summary["mean_js"]:.6g}; '
                              f'top1 differences={pair_summary["top1_disagreements"]}/{len(fixed_ids)}', flush=True)
                        pending_pair = None
                        del other_logits, by_mode, a_logits, r_logits
                json_write(out / 'results.json', report)
                with (out / 'fixed_tokens.csv').open('w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
                    writer.writeheader()
                    writer.writerows(all_rows)
                trace_file.flush()
                write_suite_markdown(out / 'generations.md', report)
                if args.save_logits:
                    torch.save({'condition': spec.name, 'fixed_logits': fixed['logits'],
                                'free_logits': free['logits'], 'fixed_token_ids': fixed_ids,
                                'free_token_ids': free['token_ids']}, out / f'{spec.name}_logits.pt')
                require(control_pass is not False,
                        f'Null control failed for {spec.name}; max logit delta={summary["max_abs_logit_delta"]}. '
                        'Results were saved but cannot be interpreted as a clean intervention.')
                # Greedy controls should be identical. With stochastic decoding and
                # nonzero numeric tolerance, rare boundary flips are only recorded.
                if null and args.temperature == 0:
                    require(same_free, f'Greedy null-control output changed for {spec.name}.')
                print(f'  mean KL={summary["mean_kl"]:.6g}, top1 agreement={summary["top1_agreement"]:.3f}, '
                      f'max delta={summary["max_abs_logit_delta"]:.6g}', flush=True)
            require(pending_pair is None, 'An incomplete replay/attenuation pair remains.')
            if args.save_logits:
                torch.save({'fixed_logits': baseline_fixed['logits'], 'free_logits': baseline_free['logits'],
                            'fixed_token_ids': fixed_ids, 'free_token_ids': baseline_free['token_ids']}, out / 'native_logits.pt')
        require(all(parameter_versions[name] == p._version for name, p in model.named_parameters()),
                'A model parameter version changed during the run.')
        report.update(status='completed', parameter_version_counters_unchanged=True,
                      caution='Version counters are a mutation check, not a cryptographic weight comparison. No quality gain is established by this suite.')
        manifest.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        error_traceback = traceback.format_exc()
        manifest.update(status='failed', error=repr(exc), traceback=error_traceback,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        report.update(status='failed', error=repr(exc), traceback=error_traceback)
        (out / 'error_traceback.txt').write_text(error_traceback, encoding='utf-8')
        raise
    finally:
        if adapter is not None:
            adapter.close()
        if trace_file is not None:
            trace_file.close()
        json_write(out / 'manifest.json', manifest)
        json_write(out / 'results.json', report)
        write_suite_markdown(out / 'generations.md', report)
        write_pair_outputs(out, report, pair_rows)
    print(f'Completed. Read {out / "paired_generations.md"}, {out / "generations.md"} and {out / "fixed_tokens.csv"}.', flush=True)


def legacy_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version=f'QCache Lab {VERSION}')
    sub = parser.add_subparsers(dest='command', required=True)
    p_device = sub.add_parser('device-check', help='Check CPU scoring under current device hooks; no model needed.')
    p_device.add_argument('--out', default='qcache_device_check.json', type=Path)
    p_test = sub.add_parser('self-test', help='PyTorch-only core and independent random toy decoder tests.')
    p_test.add_argument('--out', default='qcache_selftest.json', type=Path)
    p_hf = sub.add_parser('hf-smoke', help='Actual HF random tiny Llama/Mistral/Qwen2 adapter tests; no model downloads.')
    p_hf.add_argument('--out', default='qcache_hf_smoke.json', type=Path)
    p_hf.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda', 'auto'])
    p = sub.add_parser('run', help='Fixed-token probes and independent generations on a pretrained HF model.')
    p.add_argument('--model', required=True, help='Local HF checkpoint directory or Hub model ID; safetensors required.')
    p.add_argument('--revision', default='main', help='Hub revision; resolved commit is recorded when available.')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--dtype', default='auto', choices=['auto', 'float32', 'float16', 'bfloat16'])
    group = p.add_mutually_exclusive_group()
    group.add_argument('--prompt', default='Tell a short story about a brass key that opens the wrong door.')
    group.add_argument('--prompt-file', help='UTF-8 prompt file.')
    p.add_argument('--chat', action='store_true', help='Apply the checkpoint tokenizer chat template once.')
    p.add_argument('--continuation-file', help='Optional UTF-8 fixed continuation, tokenized separately without special tokens.')
    p.add_argument('--lambdas', default='0,0.05,0.1,0.2', help='Comma-separated strengths. Zero control is always included.')
    p.add_argument('--families', default='replay,attenuation',
                   help='replay,attenuation (default) runs matched pairs; replay keeps the old sweep only.')
    p.add_argument('--layers', default='all', help='all, comma-separated indices, or Python-style ranges such as 0:8,16.')
    p.add_argument('--prefill', default='clean', choices=['clean', 'stream'])
    p.add_argument('--engine', default='online', choices=['online', 'dense'], help='dense is the slow full-reread reference.')
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--max-context', type=int, default=512, help='Hard stop, never an eviction policy.')
    p.add_argument('--chunk', type=int, default=64, help='Readout query chunk size; no pruning.')
    p.add_argument('--temperature', type=float, default=0, help='0=greedy; otherwise CPU-seeded sampling.')
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=0, help='0 leaves PyTorch thread count unchanged.')
    p.add_argument('--trace-every', type=int, default=1, help='0 disables attention metrics; otherwise every N positions.')
    p.add_argument('--verify-every', type=int, default=0, help='Expensive dense readout assertion every N positions; 0=off.')
    p.add_argument('--control-atol', type=float, default=1e-6)
    p.add_argument('--save-logits', action='store_true', help='Save full CPU logits for later re-analysis; may be large.')
    p.add_argument('--out', default='results_qcache', help='New or empty output directory; existing runs are never overwritten.')
    args = parser.parse_args()
    try:
        if args.command == 'device-check':
            return 0 if device_check(args.out) else 2
        if args.command == 'self-test':
            return 0 if self_test(args.out) else 1
        if args.command == 'hf-smoke':
            return 0 if hf_smoke(args.out, args.device) else 2
        run_experiment(args)
        return 0
    except (ValueError, ImportError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        traceback.print_exc()
        return 2




# ============================================================================
# V1.2: free trajectories and same-state forks. Legacy v0.1.2 remains above.
# ============================================================================
import copy
import html
import itertools
import tempfile
from dataclasses import asdict

VERSION = '1.2.0'
BACKEND = 'qcache_lab_v120'
LEGACY_SOURCE_SHA256 = 'd2f326bb9c5cb70246c5dc488759b26732d67838ea3d6e1582a3dce895c68858'
LIVE_MODES = {'baseline', 'store', 'replay', 'attenuation', 'parallel'}


@dataclass(frozen=True)
class LivePolicy:
    name: str = 'replay'
    mode: str = 'replay'
    strength: float = .55
    admit: bool = True

    def __post_init__(self):
        require(self.mode in LIVE_MODES, f'Unsupported live mode: {self.mode}')
        require(math.isfinite(self.strength) and 0 <= self.strength <= 1, 'lambda must be in [0, 1].')
        require(self.mode not in {'baseline', 'store'} or self.strength == 0,
                'baseline/store require lambda=0.')


class LiveBank(QBank):
    """Separate query membership from observed KV length; no eviction or Q rewrite.

    n: admitted queries; seen: observed KV positions. Positions are kept explicitly.
    Stopping admission does NOT freeze existing readouts: all admitted Q continue
    to read every newly observed K/V. This is not an output-preserving cache.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen = 0
        self.positions: list[int] = []

    def bootstrap(self, q, k, v, scale):
        super().bootstrap(q, k, v, scale)
        self.seen = k.shape[-2]
        self.positions = list(range(self.n))

    @torch.no_grad()
    def step(self, q, k, v, scale, diagnostics=False, admit=True):
        require(self.n > 0 and q.shape[-2] == 1, 'Initialized bank and single-token append required.')
        require(k.shape[-2] == v.shape[-2] == self.seen + 1, 'Observed KV position mismatch.')
        require(self.seen + 1 <= self.max_tokens, 'Context limit reached; no truncation.')
        require(float(scale) == self.scale, 'Attention scale changed.')
        require(q.device == self.q.device and q.dtype == self.q.dtype and
                q.shape[:2] == self.q.shape[:2] and q.shape[-1] == self.q.shape[-1],
                'Q layout/device/dtype changed.')
        if admit:
            self._reserve(self.n + 1, q, v.shape[-1])
        mean, metrics = None, {}
        if self.replay:
            oldr, oldz = self.r[..., :self.n, :], self.z[..., :self.n, :]
            s = gqa_scores(self.q[..., :self.n, :].to(self.acc_dtype),
                           k[..., -1:, :].to(self.acc_dtype), scale)
            m = torch.maximum(oldz, s)
            a, b = torch.exp(oldz - m), torch.exp(s - m)
            alpha, beta = a / (a + b), b / (a + b)
            if self.engine == 'online':
                newest = v[..., -1:, :].to(self.acc_dtype).repeat_interleave(q.shape[1] // v.shape[1], 1)
                newr, newz = alpha * oldr + beta * newest, m + torch.log(a + b)
            else:
                newr, newz = dense_read(self.q[..., :self.n, :], k, v, scale, self.acc_dtype, self.chunk)
            if diagnostics:
                metrics = {'gate_mean': float(beta.mean()), 'gate_max': float(beta.max()),
                           'readout_drift_rms': float((newr - oldr).square().mean().sqrt())}
            oldr.copy_(newr)
            oldz.copy_(newz)
            # This mean is computed BEFORE optionally admitting the current Q.
            mean = newr.mean(-2, keepdim=True)
            if admit:
                r, z = dense_read(q, k, v, scale, self.acc_dtype, self.chunk)
                self.r[..., self.n:self.n + 1, :].copy_(r)
                self.z[..., self.n:self.n + 1, :].copy_(z)
        if admit:
            self.q[..., self.n:self.n + 1, :].copy_(q.detach())
            self.positions.append(self.seen)
            self.n += 1
        self.seen += 1
        return mean, metrics

    def snapshot(self):
        # Capacity is retained so restored matmul inputs retain the same strides.
        return {'n': self.n, 'capacity': self.capacity, 'seen': self.seen,
                'positions': list(self.positions), 'scale': self.scale,
                'max_tokens': self.max_tokens, 'replay': self.replay, 'engine': self.engine,
                'acc_dtype': str(self.acc_dtype).split('.')[-1], 'chunk': self.chunk,
                'q': self.q[..., :self.n, :].detach().to('cpu').clone(),
                'r': None if self.r is None else self.r[..., :self.n, :].detach().to('cpu').clone(),
                'z': None if self.z is None else self.z[..., :self.n, :].detach().to('cpu').clone()}

    @classmethod
    def restore(cls, data, device):
        b = cls(max_tokens=data['max_tokens'], replay=data['replay'], engine=data['engine'],
                acc_dtype=getattr(torch, data['acc_dtype']), chunk=data['chunk'])
        b.n, b.seen, b.capacity, b.scale = data['n'], data['seen'], data['capacity'], data['scale']
        b.positions = list(data['positions'])
        require(0 < b.n <= b.capacity <= b.max_tokens and b.seen <= b.max_tokens,
                'Invalid bank snapshot counts.')
        require(len(b.positions) == b.n and b.positions == sorted(set(b.positions))
                and b.positions[0] >= 0 and b.positions[-1] < b.seen, 'Invalid query positions.')
        for name in ('q', 'r', 'z'):
            src = data[name]
            if src is None:
                setattr(b, name, None)
            else:
                require(src.shape[-2] == b.n, 'Invalid snapshot tensor length.')
                shape = list(src.shape); shape[-2] = b.capacity
                dst = torch.empty(shape, dtype=src.dtype, device=device)
                dst[..., :b.n, :].copy_(src.to(device))
                setattr(b, name, dst)
        require(not b.replay or (b.r is not None and b.z is not None), 'Replay snapshot lacks readouts.')
        return b


def parallel_output(base: Tensor, replay: Tensor, dtype=torch.float32):
    """Per B,T,head L2 match BEFORE W_O, relative to LOCAL hypothetical replay.

    This is not a cross-branch global norm match. A zero ordinary vector cannot
    supply a direction for a nonzero target; reject it instead of choosing one.
    """
    a, r = base.to(dtype), replay.to(dtype)
    an, rn = a.norm(dim=-1, keepdim=True), r.norm(dim=-1, keepdim=True)
    require(not bool(((an == 0) & (rn != 0)).any()),
            'parallel is undefined: zero base head with nonzero replay target.')
    safe = torch.where(an == 0, torch.ones_like(an), an)
    out = (a * (rn / safe)).to(base.dtype)
    require(bool(torch.isfinite(out).all()), 'Nonfinite parallel output; refusing a silent clamp.')
    return out


class LiveController:
    """Sequential, frozen-weight, single-stream controller with serial fork restore.

    All free-trajectory arms keep shadow Q/readout state. Thus cut/native branches
    can return to replay without skipped observations; only their output differs.
    Pure attenuation's arithmetic is identical to the legacy no-bank arm, but its
    compute/memory is not. The separate legacy controller above remains unchanged.
    """
    def __init__(self, *, layers=None, max_tokens=512, engine='online', chunk=64,
                 trace_every=8, verify_every=0, acc_dtype=torch.float32):
        self.layers, self.max_tokens, self.engine, self.chunk = layers, max_tokens, engine, chunk
        self.trace_every, self.verify_every, self.acc_dtype = trace_every, verify_every, acc_dtype
        self.trace_sink = None
        self.projectors = {}
        self.reset(LivePolicy('baseline', 'baseline', 0))

    def reset(self, condition, branch=''):
        if not isinstance(condition, LivePolicy):
            condition = LivePolicy(condition.name, condition.mode, condition.strength)
        self.condition, self.branch, self.phase = condition, branch, 'unknown'
        self.banks = {}
        self.calls = 0
        self.eligible_positions = {}
        self.trace = []

    def set_policy(self, policy):
        self.condition = policy

    @torch.no_grad()
    def apply(self, layer, q, k, v, base, scale):
        if self.layers is not None and layer not in self.layers:
            return base
        require(q.shape[0] == 1 and base.shape == (1, q.shape[-2], q.shape[1], v.shape[-1]),
                'Only one unpadded sequence and B,T,H,D output are supported.')
        self.calls += 1
        bank = self.banks.get(layer)
        if bank is None:
            bank = LiveBank(max_tokens=self.max_tokens, replay=True, engine=self.engine,
                            acc_dtype=self.acc_dtype, chunk=self.chunk)
            bank.bootstrap(q, k, v, scale)
            self.banks[layer] = bank
            return base
        policy, past = self.condition, bank.n
        position = k.shape[-2] - 1
        diag = bool(self.trace_every and position % self.trace_every == 0)
        mean, detail = bank.step(q, k, v, scale, diagnostics=diag, admit=policy.admit)
        self.eligible_positions.setdefault(layer, []).append(position)
        if self.verify_every and bank.seen % self.verify_every == 0:
            bank.verify(k, v)
        strength = policy.strength
        hypothetical = None
        if policy.mode in {'baseline', 'store'} or strength == 0:
            out = base
        elif policy.mode == 'attenuation':
            out = ((1 - strength) * base.to(self.acc_dtype)).to(base.dtype)
        else:
            hypothetical = ((1 - strength) * base.to(self.acc_dtype) +
                            strength * mean.transpose(1, 2)).to(base.dtype)
            out = hypothetical if policy.mode == 'replay' else parallel_output(base, hypothetical, self.acc_dtype)
        if diag:
            a, h, y = base.to(self.acc_dtype), mean.transpose(1, 2), out.to(self.acc_dtype)
            eps = torch.finfo(self.acc_dtype).eps
            row = dict(branch=self.branch, phase=self.phase, mode=policy.mode,
                       lambda_value=strength, layer=layer, position=position,
                       admitted_queries_before=past, admitted_queries_after=bank.n,
                       observed_kv=bank.seen, admission_enabled=policy.admit,
                       applied=out is not base, base_l2=float(a.norm()), output_l2=float(y.norm()),
                       history_l2=float(h.norm()),
                       relative_update=float((y-a).norm()/a.norm().clamp_min(eps)),
                       history_current_cosine=float(((a*h).sum(-1)/
                           (a.norm(dim=-1)*h.norm(dim=-1)).clamp_min(eps)).mean()),
                       bank_used_bytes=bank.bytes(), **detail)
            if policy.mode == 'parallel' and hypothetical is not None:
                r = hypothetical.to(self.acc_dtype)
                row['local_head_norm_match_max_abs'] = float((y.norm(dim=-1)-r.norm(dim=-1)).abs().max())
                row['norm_target_scope'] = 'this arm, this layer/position/head, before W_O'
            projector = self.projectors.get(layer)
            if projector is not None:
                # Observe using the actual projection weights; do not invoke module hooks.
                w, bias = projector.weight, projector.bias
                def projected(z):
                    return torch.nn.functional.linear(z.reshape(z.shape[0], z.shape[1], -1).to(w.dtype), w, bias)
                row['projected_output_l2'] = float(projected(out).float().norm())
                if hypothetical is not None:
                    row['projected_local_replay_l2'] = float(projected(hypothetical).float().norm())
            if self.trace_sink is None:
                self.trace.append(row)
            else:
                self.trace_sink(row)
        return out

    def memory(self):
        return {'used_bytes': sum(b.bytes() for b in self.banks.values()),
                'allocated_bytes': sum(b.bytes(True) for b in self.banks.values()),
                'queries_per_layer': {str(i): b.n for i,b in self.banks.items()},
                'observed_kv_per_layer': {str(i): b.seen for i,b in self.banks.items()}}

    def schedule(self):
        return {str(i): {'eligible_count': len(p), 'first_position': p[0], 'last_position': p[-1]}
                for i,p in self.eligible_positions.items() if p}

    def snapshot(self):
        return {'condition': asdict(self.condition), 'phase': self.phase, 'calls': self.calls,
                'layers': None if self.layers is None else sorted(self.layers),
                'max_tokens': self.max_tokens, 'engine': self.engine, 'chunk': self.chunk,
                'acc_dtype': str(self.acc_dtype).split('.')[-1],
                'eligible_positions': copy.deepcopy(self.eligible_positions),
                'banks': {str(i): b.snapshot() for i,b in self.banks.items()}}

    def restore(self, data, device, branch):
        require(data['layers'] == (None if self.layers is None else sorted(self.layers)), 'Layer set changed on restore.')
        require(data['max_tokens'] == self.max_tokens and data['engine'] == self.engine and
                data['chunk'] == self.chunk and data['acc_dtype'] == str(self.acc_dtype).split('.')[-1],
                'Controller configuration changed on restore.')
        self.condition = LivePolicy(**data['condition'])
        self.phase, self.calls, self.branch = data['phase'], data['calls'], branch
        self.eligible_positions = {int(i): list(p) for i,p in data['eligible_positions'].items()}
        self.banks = {int(i): LiveBank.restore(b, device) for i,b in data['banks'].items()}
        self.trace = []


def _clone_cpu(t):
    return t.detach().to('cpu').clone(memory_format=torch.contiguous_format)


def _tensor_digest(t):
    a = t.detach().to('cpu').contiguous()
    payload = a.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(str((str(a.dtype), tuple(a.shape))).encode()+payload).hexdigest()


def pack_state(data):
    """JSON tree + flat tensors, no pickle. Deterministic traversal and key checks."""
    tensors = {}
    def walk(obj):
        if isinstance(obj, Tensor):
            key = f'tensor_{len(tensors):05d}'
            tensors[key] = _clone_cpu(obj)
            return {'__tensor__': key}
        if isinstance(obj, dict):
            require('__tensor__' not in obj, 'Reserved snapshot key.')
            return {str(k): walk(v) for k,v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
        if isinstance(obj, (tuple, list)):
            return [walk(x) for x in obj]
        require(obj is None or isinstance(obj, (str, bool, int, float)), f'Unsupported snapshot object: {type(obj)}')
        return obj
    tree = walk(data)
    return tree, tensors


def unpack_state(tree, tensors):
    if isinstance(tree, dict):
        if set(tree) == {'__tensor__'}:
            return tensors[tree['__tensor__']].clone()
        return {k: unpack_state(v, tensors) for k,v in tree.items()}
    if isinstance(tree, list):
        return [unpack_state(x, tensors) for x in tree]
    return tree


def state_digest(data):
    tree, tensors = pack_state(data)
    h = hashlib.sha256(json.dumps(tree, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
    for key in sorted(tensors):
        h.update(key.encode()); h.update(_tensor_digest(tensors[key]).encode())
    return h.hexdigest()


def save_checkpoint(path: Path, data):
    from safetensors.torch import save_file
    tree, tensors = pack_state(data)
    tensor_path = path.with_suffix('.safetensors')
    temp = tensor_path.with_suffix('.safetensors.tmp')
    save_file(tensors, str(temp))
    temp.replace(tensor_path)
    record = {'schema': 'qcache.checkpoint.1.2', 'tree': tree,
              'tensor_file': tensor_path.name,
              'tensor_sha256': hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
              'state_sha256': state_digest(data)}
    json_write(path, record)
    return {'path': str(path), 'state_sha256': record['state_sha256'],
            'tensor_bytes': tensor_path.stat().st_size}


def load_checkpoint(path: Path):
    from safetensors.torch import load_file
    doc = json.loads(path.read_text(encoding='utf-8'))
    require(doc['schema'] == 'qcache.checkpoint.1.2', 'Unsupported checkpoint schema.')
    require(Path(doc['tensor_file']).name == doc['tensor_file'], 'Checkpoint tensor path must be a basename.')
    p = path.parent/doc['tensor_file']
    require(hashlib.sha256(p.read_bytes()).hexdigest() == doc['tensor_sha256'], 'Checkpoint tensor hash mismatch.')
    data = unpack_state(doc['tree'], load_file(str(p), device='cpu'))
    require(state_digest(data) == doc['state_sha256'], 'Checkpoint logical state hash mismatch.')
    return data


class LiveRunner(HFRunner):
    def kv_snapshot(self):
        from transformers.cache_utils import DynamicCache, DynamicLayer
        require(type(self.cache) is DynamicCache and not self.cache.offloading,
                'Snapshot supports only non-offloaded DynamicCache.')
        require(all(type(x) is DynamicLayer and x.is_initialized for x in self.cache.layers),
                'Only full initialized DynamicLayer cache snapshots are supported.')
        return [(_clone_cpu(x.keys), _clone_cpu(x.values)) for x in self.cache.layers]

    def kv_restore(self, state):
        from transformers.cache_utils import DynamicCache
        require(len(state) == self.model.config.num_hidden_layers, 'KV layer count mismatch.')
        self.cache = DynamicCache(ddp_cache_data=[(k.to(self.device).clone(),v.to(self.device).clone()) for k,v in state])


class LiveToyRunner(_ToyRunner):
    def reset(self, condition, branch=''):
        super().reset(condition, branch); self.count = 0

    def forward(self, ids):
        require(self.count + len(ids) <= self.controller.max_tokens, 'Toy context overflow.')
        y = super().forward(ids)
        self.count += len(ids)
        return y

    def kv_snapshot(self):
        return [(_clone_cpu(k),_clone_cpu(v)) for k,v in self.cache]

    def kv_restore(self, state):
        self.cache = [(k.to(self.device).clone(),v.to(self.device).clone()) for k,v in state]


def snapshot_boundary(runner, prompt, generated, rng):
    require(bool(generated), 'At least one generated, pending token is required for a fork.')
    require(runner.count == len(prompt)+len(generated)-1, 'Fork boundary is off by one.')
    require(all(b.seen == runner.count for b in runner.controller.banks.values()), 'Q/KV snapshot not aligned.')
    kv = runner.kv_snapshot()
    require(all(k.shape[-2] == v.shape[-2] == runner.count for k,v in kv), 'KV snapshot length mismatch.')
    return {'schema': 'qcache.boundary.1.2', 'boundary': 'after selecting pending token; before its model forward',
            'count': runner.count, 'prompt_ids': list(prompt), 'generated_ids': list(generated),
            'pending_token_id': generated[-1], 'generator_state': _clone_cpu(rng.get_state()),
            'kv': kv, 'controller': runner.controller.snapshot()}


def restore_boundary(runner, state, branch):
    require(state['schema'] == 'qcache.boundary.1.2', 'Wrong boundary schema.')
    require(state['count'] == len(state['prompt_ids'])+len(state['generated_ids'])-1,
            'Invalid checkpoint pending-token boundary.')
    require(state['pending_token_id'] == state['generated_ids'][-1], 'Pending token mismatch.')
    require(all(k.shape[-2] == v.shape[-2] == state['count'] for k,v in state['kv']), 'Bad checkpoint KV length.')
    runner.kv_restore(state['kv']); runner.count = state['count']
    runner.controller.restore(state['controller'], runner.device, branch)
    require(all(b.seen == runner.count for b in runner.controller.banks.values()), 'Bad checkpoint bank length.')
    gen = torch.Generator(device='cpu')
    gen.set_state(state['generator_state'].to('cpu'))
    return list(state['generated_ids']), state['pending_token_id'], gen

@dataclass(frozen=True)
class RepeatConfig:
    min_period: int = 4
    max_period: int = 64
    copies: int = 3
    min_tokens: int = 24

    def __post_init__(self):
        require(1 <= self.min_period <= self.max_period and self.copies >= 2 and self.min_tokens >= 2,
                'Invalid repeat detector parameters.')


def periodic_suffix(ids: list[int], config: RepeatConfig):
    """Exact finite token repetition, NOT hidden-state recurrence or semantic failure.

    A primitive period below min_period is ignored, rather than rediscovered at
    its multiples. All evidence is from the prefix available at this step.
    """
    n = len(ids)
    for period in range(1, min(config.max_period, n//config.copies)+1):
        width = max(period*config.copies, config.min_tokens)
        if width > n:
            continue
        if any(ids[j] != ids[j-period] for j in range(n-width+period, n)):
            continue
        start = n-width
        while start > 0 and ids[start-1] == ids[start-1+period]:
            start -= 1
        if period < config.min_period:
            return None
        unit = ids[n-period:n]
        canonical = min(tuple(unit[j:]+unit[:j]) for j in range(period))
        return {'period': period, 'start': start, 'end_exclusive': n,
                'matched_tokens': n-start, 'complete_copies': (n-start)//period,
                'unit_token_ids': unit, 'canonical_unit': list(canonical),
                'expected_next_token_id': ids[n-period]}
    return None


class RepeatMonitor:
    def __init__(self, config, prefix=None):
        self.config = config
        self.ids = list(prefix or [])
        self.active = periodic_suffix(self.ids, config)
        self.seen_units = set()
        if self.active:
            self.seen_units.add(tuple(self.active['canonical_unit']))

    def step(self, token):
        previous = self.active
        self.ids.append(token)
        active = periodic_suffix(self.ids, self.config)
        old = None if previous is None else tuple(previous['canonical_unit'])
        new = None if active is None else tuple(active['canonical_unit'])
        events = []
        if old != new:
            if previous:
                events.append({'kind': 'repeat_exit', 'detected_after_tokens': len(self.ids),
                               'pattern': previous, 'note': 'lost exact periodic suffix; not a quality judgment'})
            if active:
                events.append({'kind': 'repeat_return' if new in self.seen_units else 'repeat_enter',
                               'detected_after_tokens': len(self.ids), 'pattern': active})
                self.seen_units.add(new)
        elif active and previous and active['complete_copies'] != previous['complete_copies']:
            events.append({'kind': 'repeat_cycle', 'detected_after_tokens': len(self.ids), 'pattern': active})
        self.active = active
        return events


def distribution_row(logits, selected, eos_ids, expected=None, top_k=5):
    x = logits.detach().to(device='cpu', dtype=torch.float64)
    require(x.ndim == 1 and bool(torch.isfinite(x).all()), 'Invalid free-generation logits.')
    lp = torch.log_softmax(x, -1); p = lp.exp()
    values, ids = torch.topk(x, min(max(2, top_k), x.numel()))
    row = {'selected_token_id': int(selected), 'selected_raw_probability': float(p[selected]),
           'top1_id': int(ids[0]), 'top1_raw_probability': float(p[ids[0]]),
           'top1_top2_logit_margin': float(values[0]-values[1]),
           'entropy_nats': float(-(p*lp).sum()),
           'eos_probability': float(p[torch.tensor(sorted(eos_ids), device='cpu', dtype=torch.long)].sum()) if eos_ids else 0.,
           'top_candidates': [{'id': int(i), 'probability': float(p[i]), 'logit': float(x[i])} for i in ids[:top_k]]}
    if expected is not None:
        require(0 <= expected < x.numel(), 'Repeat target is outside the vocabulary.')
        competitors = x.clone(); competitors[expected] = -torch.inf
        row.update(repeat_expected_id=int(expected), repeat_probability=float(p[expected]),
                   repeat_margin_vs_best_other=float(x[expected]-competitors.max()),
                   chose_repeat_token=selected == expected)
    else:
        row.update(repeat_expected_id=None, repeat_probability=None,
                   repeat_margin_vs_best_other=None, chose_repeat_token=None)
    return row


def _emit(sink, row):
    if sink is not None:
        sink(row)


def free_source(runner, policy, prompt, *, max_new, prefill_mode, seed, temperature=0., top_p=1.,
                eos_ids=None, repeat_config=None, fork_at=(), event_copies=(3,6), max_forks=3,
                checkpoint_sink=None, token_sink=None, event_sink=None, branch='source'):
    """Entire source is sampled freely. Forks are prospective prefix-only events.

    checkpoint_sink receives a pending-token boundary. The source itself is not
    interrupted or resumed at that point, furnishing an untouched continuation
    control for each eventual restored branch.
    """
    eos_ids, repeat_config = eos_ids or set(), repeat_config or RepeatConfig()
    runner.reset(policy, branch)
    synchronize(runner.device); trajectory_started=time.perf_counter()
    with torch.inference_mode():
        logits = prefill(runner, prompt, prefill_mode)
        gen = torch.Generator(device='cpu').manual_seed(seed)
        ids, rows, events, checkpoints, logits_all = [], [], [], [], make_logit_tape(runner.controller)
        monitor = RepeatMonitor(repeat_config)
        manual = set(fork_at); triggered = set(); skipped = []
        stop = 'max_new_tokens'
        for step in range(max_new):
            token = choose_token(logits, gen, temperature, top_p)
            expected = None if monitor.active is None else monitor.active['expected_next_token_id']
            row = dict(branch=branch, generated_step=step, generated_count=step+1,
                       **distribution_row(logits, token, eos_ids, expected))
            ids.append(token); rows.append(row); logits_all.append(logits.clone())
            _emit(token_sink, row)
            current_events = monitor.step(token)
            for event in current_events:
                event = dict(event, branch=branch)
                events.append(event); _emit(event_sink, event)
            reasons = []
            if len(ids) in manual:
                reasons.append({'kind': 'manual', 'generated_count': len(ids)})
            active = monitor.active
            if active and active['complete_copies'] in event_copies:
                key = (tuple(active['canonical_unit']), active['start'], active['complete_copies'])
                if key not in triggered:
                    triggered.add(key)
                    reasons.append({'kind': 'periodic_suffix', 'pattern': copy.deepcopy(active),
                                    'selection': 'prospective prefix-only detector'})
            if reasons:
                if token in eos_ids or step+1 == max_new or len(checkpoints) >= max_forks or checkpoint_sink is None:
                    skipped.append({'generated_count': len(ids), 'reasons': reasons,
                                    'why': 'eos' if token in eos_ids else 'source_horizon' if step+1 == max_new
                                           else 'max_forks' if len(checkpoints) >= max_forks else 'forks_disabled'})
                else:
                    state = snapshot_boundary(runner, prompt, ids, gen)
                    info = checkpoint_sink(state, reasons)
                    checkpoints.append(dict(info, generated_count=len(ids), pending_token_id=token,
                                            first_affected_prediction_step=len(ids),
                                            intervention_forward_position=runner.count,
                                            reasons=reasons, repeat_pattern=copy.deepcopy(active)))
            if token in eos_ids:
                stop = 'eos'; break
            if step+1 < max_new:
                logits = runner.forward([token])
        synchronize(runner.device)
        return {'elapsed_seconds_with_observation':time.perf_counter()-trajectory_started,
                'policy': asdict(policy), 'token_ids': ids, 'stop_reason': stop, 'events': events,
                'repetition': repetition_stats(ids), 'checkpoints': checkpoints,
                'forks_skipped': skipped, 'manual_forks_not_reached': sorted(n for n in manual if n > len(ids)),
                'qcache': runner.controller.memory(), '_logits': logits_all, '_rows': rows}


def branch_policy(name, strength, step):
    if name == 'continue': return LivePolicy(name, 'replay', strength)
    if name == 'cut': return LivePolicy(name, 'attenuation', strength)
    if name == 'native': return LivePolicy(name, 'baseline', 0)
    if name == 'parallel': return LivePolicy(name, 'parallel', strength)
    if name == 'freeze': return LivePolicy(name, 'replay', strength, False)
    if name.startswith('pulse_cut_'):
        n = int(name.removeprefix('pulse_cut_')); require(n > 0, 'Pulse length must be positive.')
        return LivePolicy(name, 'attenuation' if step < n else 'replay', strength)
    raise ValueError(f'Unknown branch policy {name}')


def free_branch(runner, checkpoint, name, *, max_new, temperature=0., top_p=1., eos_ids=None,
                repeat_config=None, token_sink=None, event_sink=None, trace_label=None, probe_period=0):
    eos_ids, repeat_config = eos_ids or set(), repeat_config or RepeatConfig()
    with torch.inference_mode():
        prefix, pending, gen = restore_boundary(runner, checkpoint, trace_label or name)
        require(pending not in eos_ids, 'Cannot branch after an EOS token.')
        require(runner.count + max_new <= runner.controller.max_tokens,
                'Branch exceeds context capacity; increase max-context or reduce branch-new-tokens.')
        strength = checkpoint['controller']['condition']['strength']
        initial = len(prefix)
        monitor = RepeatMonitor(repeat_config, prefix)
        # Fixed motif ONLY for observation of exact continuation; never forced.
        pattern = copy.deepcopy(monitor.active)
        if probe_period:
            require(0 < probe_period <= len(prefix), 'Probe period exceeds observed generated prefix.')
            pattern = {'period': probe_period, 'unit_token_ids': prefix[-probe_period:],
                       'selection': 'explicit suffix motif; repetition need not be established',
                       'start': len(prefix)-probe_period, 'end_exclusive': len(prefix)}
        motif = None if pattern is None else pattern['unit_token_ids']
        first_departure = None
        suffix, rows, events, logits_all = [], [], [], make_logit_tape(runner.controller)
        stop = 'max_new_tokens'
        runner.controller.phase = 'branch'
        for step in range(max_new):
            runner.controller.set_policy(branch_policy(name, strength, step))
            logits = runner.forward([pending])
            expected = None if motif is None else motif[step % len(motif)]
            token = choose_token(logits, gen, temperature, top_p)
            row = dict(branch=name, branch_step=step, generated_step=initial+step,
                       generated_count=initial+step+1, **distribution_row(logits, token, eos_ids, expected))
            rows.append(row); logits_all.append(logits.clone()); suffix.append(token)
            _emit(token_sink, row)
            if expected is not None and first_departure is None and token != expected:
                first_departure = step
            for e in monitor.step(token):
                event = dict(e, branch=name, branch_step=step)
                events.append(event); _emit(event_sink, event)
            if token in eos_ids:
                stop = 'eos'; break
            pending = token
        no_repeat_windows = []
        # A mechanical window diagnostic, not semantic recovery or permanent escape.
        window = max(repeat_config.min_tokens, repeat_config.max_period)
        for end in range(window, len(suffix)+1):
            if periodic_suffix(suffix[end-window:end], repeat_config) is None:
                no_repeat_windows.append(end-window)
        return {'branch': name, 'source_prefix_tokens': initial, 'suffix_token_ids': suffix,
                'token_ids': prefix+suffix, 'stop_reason': stop, 'events': events,
                'first_exact_motif_departure_step': first_departure,
                'motif_observed': pattern, 'motif_departure_is_not_semantic_recovery': True,
                'initial_motif_followed_for_tokens': len(suffix) if motif is not None and first_departure is None else first_departure,
                'first_window_without_periodic_suffix': no_repeat_windows[0] if no_repeat_windows else None,
                'repeat_observation_window': window, 'repetition': repetition_stats(suffix),
                'qcache': runner.controller.memory(), '_logits': logits_all, '_rows': rows}


def boundary_comparison(a, b, eos_ids, target=None):
    """Direct distribution comparison for the first prediction after a shared fork."""
    target = int(a.argmax()) if target is None else target
    _, s = compare_logits(a[None], b[None], [target], 'boundary')
    _, reverse = compare_logits(b[None], a[None], [target], 'boundary_reverse')
    return {'js_nats': s['mean_js'], 'kl_continue_to_branch': s['mean_kl'],
            'kl_branch_to_continue': reverse['mean_kl'], 'total_variation': s['mean_total_variation'],
            'max_abs_logit_delta': s['max_abs_logit_delta'],
            'continue_top1': int(a.argmax()), 'branch_top1': int(b.argmax()),
            'scope': 'one forward from shared pre-forward checkpoint; no comparison of diverged future prefixes'}


def continuation_control(source, checkpoint, continued, atol=1e-6):
    n = len(checkpoint['generated_ids'])
    expected = source['token_ids'][n:]
    actual = continued['suffix_token_ids']
    overlap = min(len(expected),len(actual))
    require(overlap > 0, 'No source overlap to validate restoration.')
    require(expected[:overlap] == actual[:overlap], 'Restored continuation diverged from unbranched source tokens.')
    maximum = max(float((source['_logits'][n+i]-continued['_logits'][i]).abs().max()) for i in range(overlap))
    require(maximum <= atol, f'Restored continuation logit mismatch {maximum} > {atol}.')
    if source['stop_reason'] == 'eos' and len(expected) <= len(actual):
        require(expected == actual and continued['stop_reason'] == 'eos', 'Restored EOS changed.')
    return {'status': 'passed', 'overlap_tokens': overlap, 'max_abs_logit_delta': maximum,
            'token_ids_identical_on_overlap': True,
            'scope': 'restoration vs untouched source tail, not comparison to a canonical answer'}


def public_trajectory(result, tokenizer):
    public = {k:v for k,v in result.items() if not k.startswith('_')}
    ids = result.get('suffix_token_ids', result['token_ids'])
    public['text'] = tokenizer.decode(ids, skip_special_tokens=True)
    public['text_with_special_tokens'] = tokenizer.decode(ids, skip_special_tokens=False)
    return public


def write_csv_rows(path, rows):
    if not rows:
        path.write_text('',encoding='utf-8'); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def write_live_reports(out, report):
    branches, boundaries = [], []
    lines=['# QCache Lab V1.4: free trajectories', '', f"Status: {report.get('status')}", '',
           'All future tokens are freely generated. No baseline continuation is forced.',
           'Repeat detection is exact token periodicity, not a semantic or dynamical proof.', '']
    for key, item in report.get('runs',{}).items():
        source=item['source']
        lines += [f'## {key}: source', '', '```text', source['text'].replace('```',"'''"), '```', '']
        for f in item.get('forks',[]):
            ck=f['checkpoint']
            lines += [f"### Fork after {ck['generated_count']} generated tokens", '',
                      f"Restoration: {f.get('restoration_control',{}).get('status','not_checked')}", '',
                      '**Shared generated prefix**', '', '```text', f['prefix_text'].replace('```',"'''"), '```']
            for name,b in f.get('branches',{}).items():
                lines += ['',f'#### {name}', '', '```text', b['text'].replace('```',"'''"), '```',
                          f"Stop: {b['stop_reason']}; suffix tokens: {len(b['suffix_token_ids'])}; "
                          f"first exact motif departure: {b['first_exact_motif_departure_step']}."]
                row={'run':key,'fork_after':ck['generated_count'],'branch':name,
                     'suffix_tokens':len(b['suffix_token_ids']),'stop_reason':b['stop_reason'],
                     'first_exact_motif_departure_step':b['first_exact_motif_departure_step'],
                     'repeated_3gram_fraction':b['repetition'].get('repeated_3gram_fraction'),
                     'first_window_without_periodic_suffix':b['first_window_without_periodic_suffix']}
                branches.append(row)
                if 'boundary_vs_continue' in b:
                    boundaries.append(dict(run=key,fork_after=ck['generated_count'],branch=name,**b['boundary_vs_continue']))
    md='\n'.join(lines)+'\n'
    (out/'branches.md').write_text(md,encoding='utf-8')
    write_csv_rows(out/'branch_summary.csv',branches)
    write_csv_rows(out/'boundary_pairs.csv',boundaries)
    # The reading room is self-contained; never interpret model output as HTML.
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
          '<title>QCache Lab V1.2</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}',
          'pre{white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid;padding:1rem}',
          'summary{cursor:pointer;font-weight:600;padding:.7rem}section{margin-bottom:2rem}</style>',
          '<h1>QCache Lab V1.2</h1><p>Free continuations from shared state. No forced future tokens.</p>']
    for key,item in report.get('runs',{}).items():
        page += ['<section><h2>'+html.escape(key)+'</h2><details><summary>Source</summary><pre>'+html.escape(item['source']['text'])+'</pre></details>']
        for f in item.get('forks',[]):
            page += [f"<h3>Fork after {f['checkpoint']['generated_count']} generated tokens</h3>",
                     '<details><summary>Shared prefix</summary><pre>'+html.escape(f['prefix_text'])+'</pre></details>']
            for name,b in f.get('branches',{}).items():
                page += ['<details open><summary>'+html.escape(name)+' / '+html.escape(b['stop_reason'])+'</summary><pre>'+html.escape(b['text'])+'</pre></details>']
        page.append('</section>')
    page.append('</html>'); (out/'reading_room.html').write_text(''.join(page),encoding='utf-8')
    json_write(out/'results.json',report)
    # A small, single upload packet avoids requiring multi-megabyte traces.
    packet={'schema':'qcache.share_packet.1.2','status':report.get('status'),
            'configuration':report.get('configuration'),'controls':report.get('controls'),
            'runs':{}}
    for key,item in report.get('runs',{}).items():
        packet['runs'][key]={'source_text':item['source']['text'], 'source_repetition':item['source']['repetition'],
                            'source_stop_reason':item['source']['stop_reason'],
                            'forks':[{'checkpoint':f['checkpoint'],'prefix_text':f['prefix_text'],
                                      'restoration_control':f.get('restoration_control'),
                                      'branches':{name:{k:v for k,v in b.items() if k not in {'token_ids','events','qcache'}}
                                                  for name,b in f.get('branches',{}).items()}}
                                     for f in item.get('forks',[])]}
    json_write(out/'share_packet.json',packet)


def null_free_controls(runner, prompt, *, prefill_mode, max_new, seed, eos_ids, atol, native=None, temperature=0., top_p=1.):
    kw=dict(max_new=max_new,prefill_mode=prefill_mode,seed=seed,eos_ids=eos_ids,max_forks=0,temperature=temperature,top_p=top_p)
    if native is None:
        native=free_source(runner,LivePolicy('baseline','baseline',0),prompt,**kw)
    results=[]
    for mode in ('baseline','replay','attenuation','parallel'):
        check=free_source(runner,LivePolicy('null_'+mode,mode,0),prompt,**kw)
        require(native['token_ids']==check['token_ids'],f'Free null control tokens changed: {mode}')
        delta=max(float((a-b).abs().max()) for a,b in zip(native['_logits'],check['_logits']))
        require(delta<=atol,f'Free null control logits changed: {mode}: {delta}')
        results.append({'mode':mode,'tokens':len(check['token_ids']),'max_abs_logit_delta':delta,'status':'passed'})
    return results


def run_live(args):
    # Parse and validate before loading potentially large model weights.
    strengths=list(dict.fromkeys(float(x) for x in args.lambdas.split(',')))
    require(bool(strengths) and all(math.isfinite(x) and 0<=x<=1 for x in strengths),'Invalid lambdas.')
    seeds=list(dict.fromkeys(int(x) for x in args.seeds.split(',')))
    branches=list(dict.fromkeys(['continue']+[s.strip() for s in args.branches.split(',') if s.strip()]))
    pulses=[int(x) for x in args.pulse_steps.split(',') if x.strip()]
    branches+=['pulse_cut_'+str(x) for x in pulses if 'pulse_cut_'+str(x) not in branches]
    for name in branches: branch_policy(name,.55,0)
    forks=[int(x) for x in args.fork_at.split(',') if x.strip()]
    require(args.probe_period >= 0, 'Probe period cannot be negative.')
    require(all(n>0 for n in forks),'fork-at uses positive counts of already selected generated tokens.')
    stages=tuple(int(x) for x in args.event_copies.split(',') if x.strip())
    rc=RepeatConfig(args.min_period,args.max_period,args.repeat_copies,args.min_repeat_tokens)
    require(all(n>=rc.copies for n in stages),'event-copies must be >= repeat-copies.')
    require(args.max_new_tokens>=2 and args.branch_new_tokens>0 and args.max_forks>=0,'Invalid trajectory budgets.')
    require(args.chunk>0 and args.max_context>0 and args.threads>=0,'Invalid resource settings.')
    require(args.trace_every>=0 and args.verify_every>=0,'Invalid trace/verification interval.')
    require(math.isfinite(args.temperature) and args.temperature>=0 and 0<args.top_p<=1,'Invalid sampling settings.')
    require(math.isfinite(args.control_atol) and args.control_atol>=0,'Invalid tolerance.')
    out=Path(args.out)
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())),f'Output exists and is not empty: {out}')
    out.mkdir(parents=True,exist_ok=True); (out/'checkpoints').mkdir()
    report={'schema':'qcache.free_tree.1.2','status':'started','runs':{},'controls':[]}
    manifest={'schema':'qcache.free_manifest.1.2','status':'started','environment':environment(),
              'arguments':vars(args),'started_utc':datetime.now(timezone.utc).isoformat(),
              'protocol':{'source':'free generation under continuous replay',
                'branches':'freely sampled future suffixes from exact pre-forward state clones',
                'fixed_future_tokens':False,'source_trajectory_not_interrupted':True,
                'checkpoint_boundary':'pending token selected; not yet forwarded',
                'state_scope':'full DynamicCache KV, admitted Q/readout/logZ, positions, counters, CPU sampler RNG',
                'scope_limit':'single sequence, frozen eval model, fixed-frequency RoPE, no offload, no quantization',
                'parallel':'per-head pre-W_O norm of local hypothetical replay; not global cross-branch match',
                'shadow_state':'all live modes update Q/readouts; freeze alone disables new-Q admission',
                'event_selection':'prospective finite token periodicity; manual points are explicitly labeled',
                'not_claimed':['quality improvement','semantic escape','hidden-state periodic orbit','population effect']}}
    streams=[]; adapter=None
    try:
        hf=require_hf(); device=choose_device(args.device)
        _cpu_metric_check()
        if args.threads: torch.set_num_threads(args.threads)
        torch.manual_seed(seeds[0]); random.seed(seeds[0])
        load=dict(local_files_only=args.local_files_only,trust_remote_code=False,revision=args.revision)
        config=hf.AutoConfig.from_pretrained(args.model,**load); validate_config(config)
        layers=parse_layers(args.layers,config.num_hidden_layers)
        dtype_name=args.dtype
        if dtype_name=='auto': dtype_name='float32' if device.type=='cpu' else 'float16'
        dtype=getattr(torch,dtype_name)
        tokenizer=hf.AutoTokenizer.from_pretrained(args.model,**load)
        text=Path(args.prompt_file).read_text(encoding='utf-8') if args.prompt_file else args.prompt
        if args.prompt_token_ids:
            raw=json.loads(Path(args.prompt_token_ids).read_text(encoding='utf-8'))
            prompt=raw.get('prompt_token_ids') if isinstance(raw,dict) else raw
            text=tokenizer.decode(prompt,skip_special_tokens=False)
            manifest['prompt_input_source']='explicit token IDs; template bypassed'
            manifest['prompt_input_sha256']=hashlib.sha256(Path(args.prompt_token_ids).read_bytes()).hexdigest()
        elif args.chat:
            require(bool(tokenizer.chat_template),'Tokenizer has no chat template.')
            extra={} if args.chat_date is None else {'date_string':args.chat_date}
            prompt=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,
                                                 add_generation_prompt=True,**extra)
        else:
            prompt=tokenizer.encode(text,add_special_tokens=True)
        require(isinstance(prompt,list) and prompt and all(isinstance(x,int) and 0<=x<config.vocab_size for x in prompt),
                'Invalid or empty prompt IDs.')
        total=len(prompt)+args.max_new_tokens+(args.branch_new_tokens if args.max_forks else 0)
        resolution=resolve_context(getattr(args,'context_request',args.max_context),config,len(prompt),args.max_new_tokens,
                                   args.branch_new_tokens if args.max_forks else 0)
        args.max_context=resolution['effective_context']
        manifest['length_resolution']=resolution
        manifest['research']=asdict(RESEARCH_DEFAULTS)
        print('[lengths] '+json.dumps(resolution),flush=True)
        manifest.update(prompt_text=text,prompt_token_ids=prompt,device=str(device),dtype=dtype_name,
                        model_config=config.to_dict(),selected_layers=sorted(layers),
                        resolved_revision=getattr(config,'_commit_hash',None),detector=asdict(rc),branch_order=branches)
        json_write(out/'manifest.json',manifest)
        print(f'Loading {args.model}; {device}/{dtype_name}; free source + {len(branches)} fork laws.',flush=True)
        model=hf.AutoModelForCausalLM.from_pretrained(args.model,config=config,dtype=dtype,
                        attn_implementation='eager',use_safetensors=True,**load).to(device).eval()
        model.requires_grad_(False)
        wrong=[(n,str(t.device)) for n,t in itertools.chain(model.named_parameters(),model.named_buffers())
               if t.device.type!=device.type or (device.type=='cuda' and t.device.index!=
                   (torch.cuda.current_device() if device.index is None else device.index))]
        require(not wrong,f'Model placement differs from requested device: {wrong[:5]}')
        parameter_versions={n:p._version for n,p in model.named_parameters()}
        ctrl=LiveController(layers=layers,max_tokens=args.max_context,engine=args.engine,chunk=args.chunk,
                            trace_every=args.trace_every,verify_every=args.verify_every)
        runner=LiveRunner(model,ctrl,device)
        def jsonl(name):
            f=(out/name).open('w',encoding='utf-8'); streams.append(f)
            def write(row): f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            return write
        token_sink,event_sink,attention_sink=jsonl('token_trace.jsonl'),jsonl('events.jsonl'),jsonl('attention_trace.jsonl')
        eos=getattr(model.generation_config,'eos_token_id',None)
        if eos is None: eos=tokenizer.eos_token_id
        eos=set(eos if isinstance(eos,list) else [] if eos is None else [eos])
        manifest['eos_token_ids']=sorted(eos)
        manifest['model_placement']={'parameters':sorted({str(t.device) for t in model.parameters()}),
                                     'buffers':sorted({str(t.device) for t in model.buffers()})}
        print('Checking native and zero-strength free trajectories before starting forks.',flush=True)
        native=free_source(runner,LivePolicy('native','baseline',0),prompt,max_new=min(24,args.max_new_tokens),
                          prefill_mode=args.prefill,seed=seeds[0],eos_ids=eos,max_forks=0,temperature=args.temperature,top_p=args.top_p)
        adapter=HFAdapter(model,ctrl)
        report['controls']=null_free_controls(runner,prompt,prefill_mode=args.prefill,
                    max_new=min(24,args.max_new_tokens),seed=seeds[0],eos_ids=eos,atol=args.control_atol,native=native,temperature=args.temperature,top_p=args.top_p)
        del native
        ctrl.trace_sink=attention_sink
        if args.trace_projection:
            ctrl.projectors=attention_projectors(model,layers)
        report['configuration']={'model':args.model,'device':str(device),'dtype':dtype_name,
            'prompt_text':text,'prompt_token_ids':prompt,'prefill':args.prefill,'lambdas':strengths,
            'seeds':seeds,'temperature':args.temperature,'top_p':args.top_p,'selected_layers':sorted(layers),
            'detector':asdict(rc),'max_new_tokens':args.max_new_tokens,'branch_new_tokens':args.branch_new_tokens,
            'fork_at':forks,'probe_period':args.probe_period,'event_copies':stages,'max_forks':args.max_forks,'branches':branches,
            'script_version':VERSION,'script_sha256':manifest['environment']['script_sha256'],
            'protocol':manifest['protocol']}
        for strength,seed in itertools.product(strengths,seeds):
            key=f'lambda_{strength:.12g}_seed_{seed}'
            print(f'{key}: freely generating source.',flush=True)
            checkpoint_index=0
            def capture(state,reasons):
                nonlocal checkpoint_index
                checkpoint_index+=1
                path=out/'checkpoints'/f'{key}_g{len(state["generated_ids"]):05d}.json'
                info=save_checkpoint(path,state)
                print(f'  fork saved after {len(state["generated_ids"])} generated tokens ({reasons[0]["kind"]})',flush=True)
                return info
            source=free_source(runner,LivePolicy(key,'replay',strength),prompt,
                    max_new=args.max_new_tokens,prefill_mode=args.prefill,seed=seed,
                    temperature=args.temperature,top_p=args.top_p,eos_ids=eos,repeat_config=rc,
                    fork_at=forks,event_copies=stages,max_forks=args.max_forks,checkpoint_sink=capture,
                    token_sink=token_sink,event_sink=event_sink,branch=key+'/source')
            item={'source':public_trajectory(source,tokenizer),'forks':[]}; report['runs'][key]=item
            write_live_reports(out,report)
            if args.save_logits:
                from safetensors.torch import save_file
                save_file({'logits':torch.stack(list(source['_logits']))},str(out/f'{key}_source_logits.safetensors'))
            for ck in source['checkpoints']:
                state=load_checkpoint(Path(ck['path']))
                digest=state_digest(state)
                fork={'checkpoint':ck,'prefix_text':tokenizer.decode(state['generated_ids'],skip_special_tokens=False),'branches':{}}
                item['forks'].append(fork)
                continued=None
                for name in branches:
                    print(f'  g{ck["generated_count"]}: {name}, free suffix.',flush=True)
                    tag=key+f'/g{ck["generated_count"]}/'+name
                    def wrapped_token(row): token_sink(dict(row,trajectory=tag))
                    def wrapped_event(row): event_sink(dict(row,trajectory=tag))
                    # Controller attention rows include the path set after restore.
                    result=free_branch(runner,state,name,max_new=args.branch_new_tokens,
                            temperature=args.temperature,top_p=args.top_p,eos_ids=eos,repeat_config=rc,
                            token_sink=wrapped_token,event_sink=wrapped_event,trace_label=tag,probe_period=args.probe_period)
                    require(state_digest(state)==digest,'Source checkpoint mutated by a branch.')
                    if name=='continue':
                        continued=result
                        fork['restoration_control']=continuation_control(source,state,continued,args.control_atol)
                    else:
                        result['boundary_vs_continue']=boundary_comparison(continued['_logits'][0],result['_logits'][0],eos)
                        if name=='freeze':
                            maximum=float((continued['_logits'][0]-result['_logits'][0]).abs().max())
                            require(maximum<=args.control_atol,'Freeze changed first readout before new-Q admission could act.')
                            result['freeze_first_prediction_control']={'status':'passed','max_abs_logit_delta':maximum}
                    first=result['_rows'][0]
                    result['first_prediction']={**first,
                        'top1_token':tokenizer.decode([first['top1_id']],skip_special_tokens=False)}
                    fork['branches'][name]=public_trajectory(result,tokenizer)
                    write_live_reports(out,report)
                    for stream in streams: stream.flush()
                    if args.save_logits:
                        from safetensors.torch import save_file
                        save_file({'logits':torch.stack(list(result['_logits']))},
                                  str(out/f'{key}_g{ck["generated_count"]}_{name}_logits.safetensors'))
                del continued,state
            del source
        require(all(parameter_versions[n]==p._version for n,p in model.named_parameters()),'Model parameter version changed.')
        report.update(status='completed',parameter_version_counters_unchanged=True,
                      caution='Version counters are not cryptographic weight verification. This is not a quality benchmark.')
        manifest.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        manifest.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        try:
            if adapter is not None: adapter.close()
        finally:
            for f in streams: f.close()
            json_write(out/'manifest.json',manifest)
            write_live_reports(out,report)
    print(f'Completed: {out / "reading_room.html"}; upload packet: {out / "share_packet.json"}',flush=True)


def inspect_old_results(args):
    data=json.loads(Path(args.results).read_text(encoding='utf-8'))
    config=RepeatConfig(args.min_period,args.max_period,args.repeat_copies,args.min_repeat_tokens)
    report={'schema':'qcache.offline_repeat_inspection.1.2',
            'source_sha256':hashlib.sha256(Path(args.results).read_bytes()).hexdigest(),
            'scope':'saved token IDs only, no model execution or reconstructed KV','conditions':{}}
    for name,c in data.get('conditions',{}).items():
        ids=c.get('free',{}).get('token_ids',[])
        m=RepeatMonitor(config); events=[]
        for token in ids: events.extend(m.step(token))
        report['conditions'][name]={'tokens':len(ids),'events':events,'repetition':repetition_stats(ids)}
    json_write(Path(args.out),report)
    print(f'Wrote {args.out}; token periodicity only, no inferred input settings.',flush=True)



def live_tests(out: Path, include_legacy=True, device='cpu'):
    torch.set_num_threads(min(2,torch.get_num_threads()))
    tests=[]
    def test(name,fn):
        try:
            with torch.inference_mode(): value=fn()
            tests.append({'name':name,'status':'passed','detail':value}); print('PASS '+name,flush=True)
        except Exception as exc:
            tests.append({'name':name,'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
            print('FAIL '+name+': '+repr(exc),flush=True)
    if include_legacy:
        with tempfile.TemporaryDirectory() as folder:
            legacy_path=Path(folder)/'legacy.json'
            self_test(legacy_path)
            tests.extend(dict(t,name='legacy/'+t['name']) for t in json.loads(legacy_path.read_text())['tests'])
    def tensors(n=20,dtype=torch.float64):
        g=torch.Generator(device='cpu').manual_seed(300)
        return [torch.randn(1,h,n,d,generator=g,device='cpu',dtype=dtype) for h,d in [(6,8),(2,8),(2,5)]]
    def recurrence():
        cases=0; error=0.
        for dtype in (torch.float64,torch.float32,torch.float16):
            q,k,v=tensors(dtype=dtype)
            acc=torch.float64 if dtype==torch.float64 else torch.float32
            for prefix in (1,5):
                old=QBank(max_tokens=20,acc_dtype=acc)
                new=LiveBank(max_tokens=20,acc_dtype=acc)
                for b in (old,new): b.bootstrap(q[...,:prefix,:],k[...,:prefix,:],v[...,:prefix,:],.4)
                for i in range(prefix,20):
                    a,_=old.step(q[...,i:i+1,:],k[...,:i+1,:],v[...,:i+1,:],.4)
                    b,_=new.step(q[...,i:i+1,:],k[...,:i+1,:],v[...,:i+1,:],.4)
                    assert torch.equal(a,b)
                    assert torch.equal(old.q[...,:old.n,:],new.q[...,:new.n,:])
                    assert torch.equal(old.r[...,:old.n,:],new.r[...,:new.n,:])
                    assert torch.equal(old.z[...,:old.n,:],new.z[...,:new.n,:])
                    error=max(error,float((a-b).abs().max()));cases+=1
        return {'steps':cases,'max_abs':error,'bitwise_equal_to_legacy':True}
    test('live_bank_admit_all_preserves_legacy_recurrence_bitwise',recurrence)
    def freeze_bank():
        q,k,v=tensors(); b=LiveBank(max_tokens=20,acc_dtype=torch.float64)
        b.bootstrap(q[...,:4,:],k[...,:4,:],v[...,:4,:],.4)
        before=b.q[...,:4,:].clone()
        maximum=0.
        for i in range(4,15):
            r,_=b.step(q[...,i:i+1,:],k[...,:i+1,:],v[...,:i+1,:],.4,admit=False)
            expected,_=dense_read(before,k[...,:i+1,:],v[...,:i+1,:],.4,torch.float64)
            torch.testing.assert_close(r,expected.mean(-2,keepdim=True),atol=1e-12,rtol=1e-12)
            b.verify(k[...,:i+1,:],v[...,:i+1,:],atol=1e-12,rtol=1e-12)
            assert b.n==4 and b.seen==i+1 and torch.equal(before,b.q[...,:4,:])
            maximum=max(maximum,float((r-expected.mean(-2,keepdim=True)).abs().max()))
        b.step(q[...,15:16,:],k[...,:16,:],v[...,:16,:],.4,admit=True)
        assert b.n==5 and b.seen==16 and b.positions==[0,1,2,3,15]
        b.verify(k[...,:16,:],v[...,:16,:],atol=1e-12,rtol=1e-12)
        return {'max_abs_vs_dense':maximum,'readouts_update_without_admission':True,'resume_skips_missing_Q':True}
    test('freeze_query_membership_preserves_online_readout_and_resumes',freeze_bank)
    def snapshot_bank():
        q,k,v=tensors(); b=LiveBank(max_tokens=20,acc_dtype=torch.float64)
        b.bootstrap(q[...,:5,:],k[...,:5,:],v[...,:5,:],.4)
        data=b.snapshot(); r=LiveBank.restore(data,torch.device('cpu'))
        assert r.capacity==b.capacity and r.q.stride()==b.q.stride()
        for i in range(5,20):
            a,_=b.step(q[...,i:i+1,:],k[...,:i+1,:],v[...,:i+1,:],.4)
            z,_=r.step(q[...,i:i+1,:],k[...,:i+1,:],v[...,:i+1,:],.4)
            assert torch.equal(a,z)
        original=data['q'].clone(); r.q.zero_(); assert torch.equal(data['q'],original)
        return {'capacity_and_stride_preserved':True,'no_tensor_aliasing':True}
    test('bank_snapshot_preserves_stride_and_never_aliases',snapshot_bank)
    def norm_control():
        for dtype in (torch.float32,torch.float16):
            a=torch.randn(1,1,6,8,device='cpu',dtype=dtype)
            b=torch.randn_like(a)
            p=parallel_output(a,b)
            eps=2e-3 if dtype==torch.float16 else 1e-6
            torch.testing.assert_close(p.float().norm(dim=-1),b.float().norm(dim=-1),atol=eps,rtol=eps)
            cosine=torch.nn.functional.cosine_similarity(p.float(),a.float(),dim=-1)
            torch.testing.assert_close(cosine,torch.ones_like(cosine),atol=eps,rtol=eps)
        zero=torch.zeros(1,1,2,4,device='cpu')
        assert torch.equal(parallel_output(zero,zero),zero)
        try: parallel_output(zero,torch.ones_like(zero))
        except ValueError: return {'pre_WO_per_head_norm_checked':True,'undefined_direction_rejected':True}
        raise AssertionError('Zero base direction was not rejected.')
    test('parallel_norm_direction_dtype_and_zero_guard',norm_control)
    def detector():
        conf=RepeatConfig(4,64,3,24)
        pre=list(range(100,138)); unit=list(range(14))
        m=RepeatMonitor(conf); events=[]
        for t in pre+unit*11: events+=m.step(t)
        enters=[x for x in events if x['kind']=='repeat_enter']
        assert enters[0]['detected_after_tokens']==80
        assert enters[0]['pattern']['start']==38 and enters[0]['pattern']['period']==14
        assert m.active['complete_copies']==11
        assert periodic_suffix([1]*100,conf) is None
        assert periodic_suffix(list(range(100)),conf) is None
        exit_events=m.step(444); assert exit_events[0]['kind']=='repeat_exit'
        again=[]
        for t in unit*3: again+=m.step(t)
        assert any(x['kind']=='repeat_return' for x in again)
        return {'period':14,'first_detection_after':80,'no_future_lookahead':True,'single_token_loop_filtered':True}
    test('prospective_period_detector_rotations_exit_return_and_min_period',detector)
    def detector_prefix_only():
        conf=RepeatConfig(2,12,3,6)
        pre=[8,5,7,2]*4; a=pre+[10]*20;b=pre+[90]*20
        def scan(seq):
            m=RepeatMonitor(conf);return [m.step(t) for t in seq]
        assert scan(a)[:len(pre)]==scan(b)[:len(pre)]
        return {'events_identical_before_different_future':True}
    test('event_detection_is_prefix_causal',detector_prefix_only)
    def checkpoint_disk():
        q,k,v=tensors(n=6)
        data={'tensor':q,'kv':[(k,v)],'counter':7,'meta':{'seed':22}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'c.json';saved=save_checkpoint(path,data);restored=load_checkpoint(path)
            assert state_digest(data)==state_digest(restored)
            restored['tensor'].zero_(); assert not torch.equal(data['tensor'],restored['tensor'])
            payload=bytearray(path.with_suffix('.safetensors').read_bytes());payload[-1]^=1
            path.with_suffix('.safetensors').write_bytes(payload)
            try: load_checkpoint(path)
            except ValueError: return {'roundtrip':True,'tampered_tensor_rejected':True,'pickle_used':False}
            raise AssertionError('Hash change was accepted.')
    test('safe_snapshot_disk_roundtrip_and_tamper_rejection',checkpoint_disk)

    torch.manual_seed(600)
    model=_ToyLM().eval()
    model.requires_grad_(False)
    prompt=[3,8,1,27,44,12]
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    def root_and_states(temp=0.,mode='clean',forks=(5,9)):
        c=LiveController(max_tokens=96,trace_every=1,verify_every=3)
        runner=LiveToyRunner(model,c);states=[]
        def capture(state,reasons):
            states.append(state);return {'memory_index':len(states)-1,'state_sha256':state_digest(state)}
        root=free_source(runner,LivePolicy('source','replay',.55),prompt,max_new=24,prefill_mode=mode,
                         seed=19,temperature=temp,top_p=.95 if temp else 1.,fork_at=forks,event_copies=(),
                         max_forks=4,checkpoint_sink=capture)
        return runner,root,states
    def legacy_matrix():
        checks=0
        for dtype in (torch.float32,torch.float16):
            m=copy.deepcopy(model).to(dtype)
            for mode,layers,strength,family in itertools.product(('clean','stream'),(None,{1}),(0.,.2,.55,1.),('replay','attenuation')):
                lc=Controller(layers=layers,max_tokens=64,trace_every=0)
                old=_ToyRunner(m,lc)
                spec=Condition(family,family,strength)
                a=trial(old,spec,prompt,prefill_mode=mode,max_new=10,seed=3)
                nc=LiveController(layers=layers,max_tokens=64,trace_every=0)
                new=LiveToyRunner(m,nc)
                b=free_source(new,LivePolicy(family,family,strength),prompt,
                              prefill_mode=mode,max_new=10,seed=3,max_forks=0)
                assert a['token_ids']==b['token_ids']
                assert torch.equal(a['logits'],torch.stack(list(b['_logits'])))
                checks+=1
        return {'free_trajectory_cases':checks,'all_token_ids_and_logits_bitwise_equal':True}
    test('legacy_free_generation_regression_across_precisions_layers_prefills_strengths',legacy_matrix)
    def restored_greedy():
        reports=[]
        for mode in ('clean','stream'):
            runner,root,states=root_and_states(mode=mode)
            for state in states:
                continued=free_branch(runner,state,'continue',max_new=12)
                reports.append(continuation_control(root,state,continued,atol=0))
                assert state['count']==len(prompt)+len(state['generated_ids'])-1
        return reports
    test('restored_free_continuations_match_untouched_sources_bitwise',restored_greedy)
    def restored_sampling():
        runner,root,states=root_and_states(temp=.8)
        state=states[0]
        continued=free_branch(runner,state,'continue',max_new=15,temperature=.8,top_p=.95)
        return continuation_control(root,state,continued,atol=0)
    test('sampler_rng_snapshot_reproduces_stochastic_free_suffix',restored_sampling)
    def fork_isolation():
        runner,root,states=root_and_states()
        state=states[0];digest=state_digest(state)
        saved={}
        for name in ('continue','cut','native','parallel','freeze','pulse_cut_1','pulse_cut_4'):
            result=free_branch(runner,state,name,max_new=10)
            saved[name]=result
            assert state_digest(state)==digest
        assert torch.equal(saved['freeze']['_logits'][0],saved['continue']['_logits'][0])
        assert not torch.equal(saved['cut']['_logits'][0],saved['continue']['_logits'][0])
        assert not torch.equal(saved['native']['_logits'][0],saved['continue']['_logits'][0])
        assert saved['freeze']['qcache']['queries_per_layer']['0']==state['controller']['banks']['0']['n']
        assert saved['freeze']['qcache']['observed_kv_per_layer']['0']==state['count']+10
        for name in reversed(tuple(saved)):
            a=free_branch(runner,state,name,max_new=10)
            assert a['suffix_token_ids']==saved[name]['suffix_token_ids']
            assert torch.equal(torch.stack(list(a['_logits'])),torch.stack(saved[name]['_logits']))
        return {'branch_count':len(saved),'branch_order_independent':True,'snapshot_immutable':True,
                'freeze_first_prediction_exact':True,'cut_changes_immediate_next_prediction':True}
    test('fork_branches_independent_and_no_stale_logits',fork_isolation)
    def pulse():
        runner,_,states=root_and_states();s=states[0]
        a=free_branch(runner,s,'cut',max_new=8)
        p=free_branch(runner,s,'pulse_cut_4',max_new=8)
        assert torch.equal(torch.stack(a['_logits'][:4]),torch.stack(p['_logits'][:4]))
        assert not torch.equal(a['_logits'][4],p['_logits'][4])
        return {'pulse_forward_count':4,'resumption_not_one_token_late':True}
    test('pulse_cut_exact_forward_timing_and_shadow_readout_continuity',pulse)
    def no_forced_tokens():
        runner,root,states=root_and_states()
        for name in ('continue','cut','native','parallel','freeze'):
            a=free_branch(runner,states[0],name,max_new=10)
            assert all(t==int(logit.argmax()) for t,logit in zip(a['suffix_token_ids'],a['_logits']))
        assert all(t==int(x.argmax()) for t,x in zip(root['token_ids'],root['_logits']))
        return {'every_generated_token_selected_from_own_logits':True}
    test('live_sources_and_forks_never_force_future_tokens',no_forced_tokens)
    def context_guard():
        runner,root,states=root_and_states();state=states[0]
        try: free_branch(runner,state,'cut',max_new=96)
        except ValueError: return {'oversized_branch_rejected_without_crop':True}
        raise AssertionError('Context overflow was ignored.')
    test('fork_context_capacity_rejected_not_truncated',context_guard)
    def disk_runner():
        runner,root,states=root_and_states();s=states[0]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'fork.json';save_checkpoint(p,s)
            restored=load_checkpoint(p)
            a=free_branch(runner,restored,'continue',max_new=12)
            return continuation_control(root,s,a,atol=0)
    test('whole_runner_safe_disk_roundtrip_preserves_free_continuation',disk_runner)
    def scope_metrics():
        runner,root,states=root_and_states();s=states[0]
        a=free_branch(runner,s,'continue',max_new=8);b=free_branch(runner,s,'cut',max_new=8)
        d=boundary_comparison(a['_logits'][0],b['_logits'][0],set())
        assert d['js_nats']>0 and d['max_abs_logit_delta']>0
        z=boundary_comparison(a['_logits'][0],a['_logits'][0],set())
        assert z['max_abs_logit_delta']==0 and z['kl_continue_to_branch']==0
        return {'only_first_prediction_compared_across_arms':True}
    test('boundary_direct_metrics_without_canonical_continuation',scope_metrics)
    def automatic_checkpoint():
        # A deterministic synthetic source controls only the fixture logits, not the generator.
        class ScriptedRunner:
            def __init__(self):
                self.controller=LiveController(max_tokens=200,trace_every=0);self.device=torch.device('cpu')
            def reset(self,p,branch): self.controller.reset(p,branch);self.count=0
            def forward(self,ids):
                self.count+=len(ids)
                y=torch.zeros(97,device='cpu');y[(self.count-len(prompt))%14]=10;return y
            def kv_snapshot(self):
                return [(torch.zeros(1,1,self.count,2,device='cpu'),torch.zeros(1,1,self.count,2,device='cpu'))]
        runner=ScriptedRunner();seen=[]
        def capture(s,r):seen.append(s);return {'count':s['count']}
        a=free_source(runner,LivePolicy(),prompt,max_new=100,prefill_mode='clean',seed=1,
                      checkpoint_sink=capture,repeat_config=RepeatConfig(),event_copies=(3,6),max_forks=2)
        assert [len(s['generated_ids']) for s in seen]==[42,84]
        assert a['token_ids']==[i%14 for i in range(100)]
        assert all(s['count']==len(prompt)+len(s['generated_ids'])-1 for s in seen)
        return {'fork_counts':[42,84],'source_uninterrupted':True,'synthetic_fixture_not_LM_behavior':True}
    test('automatic_event_forks_are_prefix_selected_and_source_uninterrupted',automatic_checkpoint)
    def reporting():
        runner,root,states=root_and_states();s=states[0]
        class Tokens:
            def decode(self,ids,skip_special_tokens=False): return '<script> not html '+str(ids)
        tok=Tokens();a=free_branch(runner,s,'continue',max_new=8)
        r={'status':'completed','runs':{'tiny':{'source':public_trajectory(root,tok),'forks':[
            {'checkpoint':{'generated_count':5},'prefix_text':'<script>',
             'restoration_control':continuation_control(root,s,a,0),'branches':{'continue':public_trajectory(a,tok)}}]}}}
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);write_live_reports(d,r)
            assert '&lt;script&gt;' in (d/'reading_room.html').read_text()
            assert '<script>' not in (d/'reading_room.html').read_text()
            assert (d/'branch_summary.csv').exists() and (d/'share_packet.json').exists()
            assert not (d/'fixed_tokens.csv').exists()
            json.loads((d/'results.json').read_text())
        return {'free_only_outputs':True,'model_text_html_escaped':True}
    test('free_report_json_csv_html_and_upload_packet',reporting)
    def metadata_snapshot_rejection():
        runner,root,states=root_and_states();state=copy.deepcopy(states[0]);state['count']+=1
        try: restore_boundary(runner,state,'bad')
        except ValueError:return {'corrupted_pending_boundary_rejected':True}
        raise AssertionError('Wrong count accepted')
    test('invalid_boundary_metadata_rejected',metadata_snapshot_rejection)
    def manual_probe():
        runner,root,states=root_and_states();state=states[0]
        a=free_branch(runner,state,'continue',max_new=8,probe_period=4,trace_label='special/path')
        assert a['motif_observed']['unit_token_ids']==state['generated_ids'][-4:]
        assert a['_rows'][0]['repeat_expected_id']==state['generated_ids'][-4]
        assert all(r['branch']=='special/path' for r in runner.controller.trace)
        return {'manual_suffix_probe_observational_only':True,'attention_rows_keep_full_trajectory_id':True}
    test('manual_pre_onset_motif_probe_and_unique_trace_labels',manual_probe)
    def lambda_zero_branches():
        ctrl=LiveController(max_tokens=96,trace_every=0);runner=LiveToyRunner(model,ctrl);states=[]
        def cap(s,r):states.append(s);return {}
        root=free_source(runner,LivePolicy('r0','replay',0),prompt,max_new=20,prefill_mode='clean',seed=8,
                         fork_at=(5,),event_copies=(),checkpoint_sink=cap,max_forks=1)
        for name in ('continue','cut','native','parallel','freeze','pulse_cut_1'):
            a=free_branch(runner,states[0],name,max_new=10)
            continuation_control(root,states[0],a,atol=0)
        return {'all_six_arms_null_at_lambda_zero':True}
    test('lambda_zero_all_free_fork_laws_are_exact_controls',lambda_zero_branches)
    def live_pipeline():
        from types import SimpleNamespace
        from unittest.mock import patch
        class Config(SimpleNamespace):
            def to_dict(self):return dict(vars(self))
        cfg=Config(model_type='llama',num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,
                   hidden_size=32,vocab_size=97,max_position_embeddings=512)
        with torch.inference_mode(False):
            m=copy.deepcopy(model).eval()
        m.config=cfg;m.generation_config=SimpleNamespace(eos_token_id=None)
        m.adapter_active=False
        class Tok:
            eos_token_id=None
            chat_template=None
            def encode(self,text,add_special_tokens=True):return [3,8,1,27,44,12]
            def decode(self,ids,skip_special_tokens=False):return 'toy tokens '+str(ids)
        fake=SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *a,**kw:cfg),
             AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:Tok()),
             AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a,**kw:m))
        class Adapt:
            def __init__(self,model,controller):self.model=model;model.adapter_active=True
            def close(self):self.model.adapter_active=False
        class Runner(LiveToyRunner):
            def __init__(self,model,controller,device):super().__init__(model,controller)
            def forward(self,ids):
                with torch.inference_mode():
                    out,self.cache=self.model(torch.tensor([ids],device=self.device),self.cache,
                                               self.controller if self.model.adapter_active else None)
                    self.count+=len(ids)
                    return out[0,-1].float().cpu()
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(model='random-toy-fixture',revision='main',local_files_only=True,device='cpu',
                dtype='float32',prompt='fixture',prompt_file=None,prompt_token_ids=None,chat=False,chat_date=None,
                lambdas='0.55',seeds='42',branches='continue,cut,native,parallel,freeze',pulse_steps='1',
                fork_at='5',probe_period=4,event_copies='',max_forks=1,max_new_tokens=16,branch_new_tokens=8,
                max_context=64,layers='all',prefill='clean',engine='online',chunk=8,temperature=0.,top_p=1.,
                threads=1,trace_every=2,verify_every=3,control_atol=1e-6,save_logits=False,
                trace_projection=False,min_period=4,max_period=16,repeat_copies=3,min_repeat_tokens=12,
                out=str(Path(tmp)/'run'))
            module=sys.modules[__name__]
            with patch.object(module,'require_hf',return_value=fake), \
                 patch.object(module,'LiveRunner',Runner),patch.object(module,'HFAdapter',Adapt):
                run_live(args)
            folder=Path(args.out);r=json.loads((folder/'results.json').read_text())
            assert r['status']=='completed' and len(r['controls'])==4
            fork=r['runs']['lambda_0.55_seed_42']['forks'][0]
            assert len(fork['branches'])==6 and fork['restoration_control']['status']=='passed'
            assert fork['branches']['freeze']['freeze_first_prediction_control']['status']=='passed'
            assert all((folder/name).exists() for name in ['share_packet.json','reading_room.html','branches.md',
                       'boundary_pairs.csv','branch_summary.csv','manifest.json','token_trace.jsonl','events.jsonl'])
            assert not (folder/'fixed_tokens.csv').exists()
            traces=[json.loads(l) for l in (folder/'attention_trace.jsonl').read_text().splitlines()]
            assert any(t['branch']=='lambda_0.55_seed_42/g5/cut' for t in traces)
        return {'free_run_loading_and_adapter_stubbed':True,'actual_toy_forward_executed':True,
                'checkpoint_and_branch_artifacts_checked':True,'not_HF_validation':True}
    test('live_run_end_to_end_cli_pipeline_with_toy_adapter_stub',live_pipeline)
    def model_weights():
        assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
        return {'all_toy_weights_bitwise_unchanged':True}
    test('live_toy_parameters_remain_unchanged',model_weights)
    passed=sum(x['status']=='passed' for x in tests)
    report={'schema':'qcache.validation.1.2','environment':environment(),'passed':passed,'failed':len(tests)-passed,
            'scope':'CPU mathematical and independent random toy model tests. No pretrained model or MPS execution.',
            'tests':tests}
    json_write(out,report)
    print(f'{passed}/{len(tests)} passed; {out}',flush=True)
    return report['failed']==0


def live_hf_smoke(out,device_name='cpu'):
    report={'schema':'qcache.hf_live_smoke.1.2','environment':environment(),'models':[],
            'scope':'Actual HF random tiny models, not pretrained language behavior'}
    try:
        hf=require_hf()
    except (ImportError,ValueError) as exc:
        report.update(status='not_run',reason=str(exc));json_write(out,report);print(str(exc));return False
    device=choose_device(device_name)
    try:
        for mt,config_cls,model_cls in [('llama',hf.LlamaConfig,hf.LlamaForCausalLM),
                                      ('mistral',hf.MistralConfig,hf.MistralForCausalLM),
                                      ('qwen2',hf.Qwen2Config,hf.Qwen2ForCausalLM)]:
            torch.manual_seed(902)
            opts=dict(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
                      num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256,
                      attention_dropout=0.,bos_token_id=1,eos_token_id=2,pad_token_id=0)
            if mt=='mistral': opts['sliding_window']=None
            if mt=='qwen2': opts.update(use_sliding_window=False,sliding_window=None)
            config=config_cls(**opts);config._attn_implementation='eager'
            with torch.device('cpu'): model=model_cls(config)
            model=model.to(device).eval();model.requires_grad_(False)
            ctrl=LiveController(max_tokens=96,trace_every=1,verify_every=3)
            ctrl.projectors={i:l.self_attn.o_proj for i,l in enumerate(model.model.layers)}
            runner=LiveRunner(model,ctrl,device);prompt=[1,5,9,8,12]
            native=free_source(runner,LivePolicy('native','baseline',0),prompt,max_new=8,prefill_mode='clean',seed=9,max_forks=0)
            with HFAdapter(model,ctrl):
                nulls=null_free_controls(runner,prompt,max_new=8,prefill_mode='clean',seed=9,eos_ids=set(),atol=1e-6,native=native)
                states=[]
                def cap(s,r):states.append(s);return {'memory_index':len(states)-1}
                root=free_source(runner,LivePolicy(),prompt,max_new=20,prefill_mode='clean',seed=9,
                                 fork_at=(5,),event_copies=(),max_forks=1,checkpoint_sink=cap)
                state=states[0];restored=[];continue_result=None
                for name in ('continue','cut','parallel','native','freeze','pulse_cut_2'):
                    result=free_branch(runner,state,name,max_new=10)
                    assert all(bool(torch.isfinite(x).all()) for x in result['_logits'])
                    if name=='continue':
                        restored=continuation_control(root,state,result,1e-6);continue_result=result
                    if name=='freeze':
                        torch.testing.assert_close(result['_logits'][0],continue_result['_logits'][0],atol=1e-6,rtol=0)
                        assert len(set(result['qcache']['queries_per_layer'].values()))==1
                    if name=='parallel':
                        assert any('projected_local_replay_l2' in t for t in ctrl.trace)
                # Serialization and actual DynamicCache restoration, not a fake cache.
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)/'c.json';save_checkpoint(p,state)
                    result=free_branch(runner,load_checkpoint(p),'continue',max_new=10)
                    continuation_control(root,state,result,1e-6)
                ctrl.engine='dense'
                dense=free_source(runner,LivePolicy(),prompt,max_new=12,prefill_mode='stream',seed=9,max_forks=0)
                ctrl.engine='online'
                online=free_source(runner,LivePolicy(),prompt,max_new=12,prefill_mode='stream',seed=9,max_forks=0)
                assert online['token_ids']==dense['token_ids']
                maxdiff=max(float((a-b).abs().max()) for a,b in zip(online['_logits'],dense['_logits']))
                require(maxdiff<=2e-5,'Online vs dense stream mismatch')
            report['models'].append({'model_type':mt,'status':'passed','controls':nulls,
                                     'restore':restored,'stream_online_dense_max_abs':maxdiff})
            print('HF LIVE PASS '+mt,flush=True)
            del model,ctrl,runner
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
    json_write(out,report)
    return report['status']=='passed'


def live_main():
    # Explicit opt-in retains the old fixed-token laboratory unchanged.
    if len(sys.argv)>1 and sys.argv[1]=='legacy-run':
        sys.argv[1]='run';return legacy_main()
    parser=argparse.ArgumentParser(description='QCache Lab V1.2: free trajectories, event capture and same-state forks.')
    parser.add_argument('--version',action='version',version='QCache Lab '+VERSION)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('device-check');p.add_argument('--out',type=Path,default=Path('qcache_v12_device_check.json'))
    p=sub.add_parser('self-test');p.add_argument('--out',type=Path,default=Path('qcache_v12_selftest.json'))
    p.add_argument('--live-only',action='store_true',help='Skip the retained legacy mathematical tests.')
    p.add_argument('--non-cpu-default',action='store_true',help='Use a meta default context to test explicit CPU placement; not an MPS test.')
    p=sub.add_parser('hf-smoke');p.add_argument('--out',type=Path,default=Path('qcache_v12_hf_smoke.json'))
    p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto'])
    p=sub.add_parser('inspect',help='Detect exact repeats in a previous results.json, without running a model.')
    p.add_argument('--results',required=True);p.add_argument('--out',default='qcache_repeat_inspection.json')
    for name,default in [('min-period',4),('max-period',64),('repeat-copies',3),('min-repeat-tokens',24)]:
        p.add_argument('--'+name,type=int,default=default)
    p=sub.add_parser('run',help='Free source generation plus state-fork continuations. No forced continuation.')
    p.add_argument('--model',required=True);p.add_argument('--revision',default='main')
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    p.add_argument('--dtype',default='auto',choices=['auto','float32','float16','bfloat16'])
    group=p.add_mutually_exclusive_group()
    group.add_argument('--prompt',default='Tell me about Japanese culture.')
    group.add_argument('--prompt-file');group.add_argument('--prompt-token-ids',help='JSON token-ID list or manifest with prompt_token_ids; bypass text template.')
    p.add_argument('--chat',action='store_true');p.add_argument('--chat-date',help='Explicit date_string passed to chat template; saved token IDs are authoritative.')
    p.add_argument('--lambdas',default='0.55');p.add_argument('--seeds',default='42')
    p.add_argument('--branches',default='continue,cut,native,parallel,freeze')
    p.add_argument('--pulse-steps',default='',help='Optional cut pulses, e.g. 1,4,16; Q/readouts are updated in shadow throughout.')
    p.add_argument('--fork-at',default='',help='Counts of generated tokens already selected, e.g. 52,66,94. Not source text offsets.')
    p.add_argument('--probe-period',type=int,default=0,help='Observe repetition of the last N prefix tokens even before a repeat is detected; never force them.')
    p.add_argument('--event-copies',default='3,6',help='Observed complete repeats that trigger snapshots; empty string disables automatic forks.')
    p.add_argument('--max-forks',type=int,default=3,help='Per source; skipped events are reported, not silently substituted.')
    p.add_argument('--max-new-tokens',type=int,default=192)
    p.add_argument('--branch-new-tokens',type=int,default=96)
    p.add_argument('--max-context','--context',type=context_argument,default='auto')
    p.add_argument('--max-tokens',dest='max_new_tokens',type=int,default=argparse.SUPPRESS)
    add_research_arguments(p)
    p.add_argument('--layers',default='all');p.add_argument('--prefill',default='clean',choices=['clean','stream'])
    p.add_argument('--engine',default='online',choices=['online','dense']);p.add_argument('--chunk',type=int,default=64)
    p.add_argument('--temperature',type=float,default=0.);p.add_argument('--top-p',type=float,default=1.)
    p.add_argument('--threads',type=int,default=0);p.add_argument('--trace-every',type=int,default=8)
    p.add_argument('--trace-projection',action='store_true',help='Record additional post-W_O norms at traced positions.')
    p.add_argument('--verify-every',type=int,default=0);p.add_argument('--control-atol',type=float,default=1e-6)
    p.add_argument('--save-logits',action='store_true',help='Save raw source and branch logits as safetensors; can be large.')
    for name,default in [('min-period',4),('max-period',64),('repeat-copies',3),('min-repeat-tokens',24)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--out',default='results_qcache_v12')
    args=parser.parse_args()
    try:
        if args.command=='device-check':return 0 if device_check(args.out) else 2
        if args.command=='self-test':
            with torch.device('meta' if args.non_cpu_default else torch.get_default_device()):
                return 0 if live_tests(args.out,include_legacy=not args.live_only) else 1
        if args.command=='hf-smoke':return 0 if live_hf_smoke(args.out,args.device) else 2
        if args.command=='inspect':inspect_old_results(args);return 0
        run_live(args);return 0
    except Exception as exc:
        print('ERROR: '+str(exc),file=sys.stderr);traceback.print_exc();return 2




# ============================================================================
# V1.3: budgeted free-trajectory campaigns. Legacy and V1.2 kernels above remain.
# ============================================================================
import gc
import os
import re
import statistics
from decimal import Decimal

VERSION = '1.3.0'
BACKEND = 'qcache_lab_v130'
CAMPAIGN_SCHEMA = 'qcache.campaign.1.3'


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def lambda_key(value: float) -> str:
    require(math.isfinite(value) and 0 <= value <= 1, 'lambda must be finite and in [0,1].')
    return format(value, '.12g')


def parse_campaign_lambdas(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(',') if x.strip()]
    require(bool(values), 'The coarse grid cannot be empty.')
    for value in values:
        lambda_key(value)
    keys = {lambda_key(x): x for x in values}
    require(len(keys) == len(set(values)), 'Coarse lambdas collide at 12-digit serialization precision.')
    return sorted(set([0.0] + values))


def _seeds(text: str) -> list[int]:
    values = list(dict.fromkeys(int(x.strip()) for x in text.split(',') if x.strip()))
    require(values and all(0 <= x < 2**63 for x in values), 'Provide nonnegative seeds below 2**63.')
    return values


def _branches(text: str, pulse_steps: str) -> list[str]:
    names = list(dict.fromkeys(['continue'] + [x.strip() for x in text.split(',') if x.strip()]))
    names += [f'pulse_cut_{int(x)}' for x in pulse_steps.split(',') if x.strip()]
    names = list(dict.fromkeys(names))
    for name in names:
        branch_policy(name, .5, 0)
    return names


def _safe_id(value: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value) is not None,
            'Suite IDs must be 1-80 ASCII letters/digits/underscore/dot/hyphen and start alphanumeric.')
    return value


def _load_json(path: Path) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key {key!r} in {path}')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)


def campaign_suite(args) -> dict[str, Any]:
    """Materialize the experiment input, not generated continuations. No HF load."""
    if args.suite:
        require(not args.model and args.prompt is None and not args.prompt_file and not args.prompt_token_ids,
                '--suite cannot be combined with --model or single-prompt input flags.')
        raw = _load_json(Path(args.suite))
        require(isinstance(raw, dict) and set(raw) <= {'models', 'prompts'},
                'Suite JSON accepts only models and prompts arrays.')
        require(isinstance(raw.get('models'), list) and raw['models'] and
                isinstance(raw.get('prompts'), list) and raw['prompts'], 'Suite models/prompts must be nonempty arrays.')
        models, prompts = [], []
        for item in raw['models']:
            require(isinstance(item, dict) and set(item) <= {'id', 'path', 'revision'} and
                    {'id', 'path'} <= set(item), 'Model item: {id, path, optional revision}.')
            require(isinstance(item['path'], str) and item['path'].strip(), 'Empty model path.')
            models.append(dict(id=_safe_id(item['id']), path=item['path'], revision=item.get('revision', args.revision)))
        for item in raw['prompts']:
            require(isinstance(item, dict) and set(item) <= {'id', 'text', 'chat'} and
                    {'id', 'text'} <= set(item), 'Prompt item: {id, text, optional chat}.')
            require(isinstance(item['text'], str) and item['text'].strip(), 'Empty prompt text.')
            require(isinstance(item.get('chat', args.chat), bool), 'Prompt chat must be boolean.')
            prompts.append(dict(id=_safe_id(item['id']), text=item['text'], chat=item.get('chat', args.chat)))
    else:
        require(bool(args.model), 'Supply --model, or --suite.')
        modes = sum([args.prompt is not None, bool(args.prompt_file), bool(args.prompt_token_ids)])
        require(modes == 1, 'Supply exactly one of --prompt, --prompt-file, --prompt-token-ids. No hidden default prompt.')
        text = Path(args.prompt_file).read_text(encoding='utf-8') if args.prompt_file else args.prompt
        item = dict(id='prompt', text=text, chat=args.chat)
        if args.prompt_token_ids:
            require(not args.chat, '--chat is not applicable to already tokenized input.')
            raw = _load_json(Path(args.prompt_token_ids))
            ids = raw.get('prompt_token_ids') if isinstance(raw, dict) else raw
            require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids),
                    'Expected nonempty prompt token IDs, not a result continuation.')
            item.update(token_ids=ids, token_input_file_sha256=_file_hash(Path(args.prompt_token_ids)))
        else:
            require(isinstance(text, str) and text.strip(), 'Prompt text is empty.')
        models = [dict(id='model', path=args.model, revision=args.revision)]
        prompts = [item]
    for collection in (models, prompts):
        require(len({x['id'] for x in collection}) == len(collection), 'Duplicate suite IDs.')
    require(all(isinstance(m['revision'], str) and m['revision'] for m in models), 'Invalid model revision.')
    return dict(models=models, prompts=prompts)


def campaign_options(args) -> dict[str, Any]:
    grid = parse_campaign_lambdas(args.coarse_lambdas)
    seeds = _seeds(args.seeds)
    branches = _branches(args.branches, args.pulse_steps)
    RepeatConfig(args.min_period, args.max_period, args.repeat_copies, args.min_repeat_tokens)
    require(len(grid) >= 2, 'A sweep requires at least two distinct lambdas (including zero).')
    require(args.max_lambdas >= len(grid), '--max-lambdas must cover the initial grid.')
    require(args.refine_rounds >= 0 and args.refine_intervals >= 1, 'Invalid refinement budgets.')
    require(args.focus_intervals >= 0 and args.max_forks >= 0, 'Invalid fork selection budgets.')
    require(math.isfinite(args.min_lambda_width) and args.min_lambda_width >= 1e-9,
            '--min-lambda-width must be at least 1e-9.')
    require(args.max_new_tokens >= 2 and args.branch_new_tokens >= 1 and args.max_context >= 2,
            'Invalid sequence budgets.')
    require(args.chunk >= 1 and args.threads >= 0 and args.trace_every >= 0 and args.verify_every >= 0,
            'Invalid resource/trace settings.')
    require(math.isfinite(args.temperature) and args.temperature >= 0 and 0 < args.top_p <= 1,
            'Invalid sampling settings.')
    require(math.isfinite(args.control_atol) and args.control_atol >= 0, 'Invalid tolerance.')
    require(args.control_tokens >= 2, 'Use at least two tokens for initial null controls.')
    for name in ('repeat_jump', 'length_jump', 'lexical_jump'):
        value = getattr(args, name)
        require(math.isfinite(value) and 0 < value <= 1, f'--{name.replace("_", "-")} must be in (0,1].')
    require(math.isfinite(args.entropy_jump) and args.entropy_jump > 0, 'Invalid entropy jump.')
    require(math.isfinite(args.onset_jump) and 0 < args.onset_jump <= 1, 'Invalid onset jump.')
    return {name: getattr(args, name) for name in (
        'device', 'dtype', 'local_files_only', 'chat_date', 'layers', 'prefill', 'engine', 'chunk',
        'temperature', 'top_p', 'threads', 'trace_every', 'trace_projection', 'verify_every', 'control_atol',
        'control_tokens', 'max_new_tokens', 'branch_new_tokens', 'max_context',
        'refine_rounds', 'refine_intervals', 'min_lambda_width', 'max_lambdas', 'focus_intervals', 'max_forks',
        'min_period', 'max_period', 'repeat_copies', 'min_repeat_tokens',
        'repeat_jump', 'length_jump', 'lexical_jump', 'entropy_jump', 'onset_jump', 'save_logits'
    )} | dict(coarse_lambdas=grid, seeds=seeds, branches=branches)


def campaign_budget(options, cases: int) -> dict[str, Any]:
    o = options
    lambdas = min(o['max_lambdas'], len(o['coarse_lambdas']) + o['refine_rounds'] * o['refine_intervals'])
    focus = min(lambdas, 2 * o['focus_intervals'])
    seeds = len(o['seeds'])
    source = cases * seeds * lambdas
    recaptures = cases * seeds * focus if o['max_forks'] else 0
    checkpoints = recaptures * o['max_forks']
    arms = checkpoints * len(o['branches'])
    control_runs = cases * seeds * 5  # actual native and 4 zero-output modes
    cap = min(o['control_tokens'], o['max_new_tokens'])
    return dict(cases=cases, max_tested_lambdas_per_case=lambdas, max_scout_trajectories=source,
                max_free_recaptures=recaptures, max_checkpoints=checkpoints,
                max_branch_trajectories=arms, max_initial_control_trajectories=control_runs,
                generated_token_upper_bound=(source+recaptures)*o['max_new_tokens'] +
                                            arms*o['branch_new_tokens'] + control_runs*cap,
                includes_model_forward_for_branch_pending_token=True,
                not_a_wall_time_estimate=True, note='Bounds, not targets. EOS/no candidates reduce work; prompt forwards are extra.')


def write_receipt(path: Path, signature: str, data: Any, artifacts=None) -> None:
    data_hash = _json_hash(data)
    record = dict(schema='qcache.receipt.1.3', signature=signature, data=data, data_sha256=data_hash,
                  artifacts=artifacts or {})
    # Receipt covers its artifact inventory as well as payload. This is not a signature.
    record['receipt_sha256'] = _json_hash(record)
    json_write(path, record)


def read_receipt(path: Path, signature: str, artifact_root: Path | None = None) -> Any:
    record = _load_json(path)
    supplied = record.pop('receipt_sha256', None)
    require(_json_hash(record) == supplied, f'Receipt checksum mismatch: {path}')
    require(record.get('schema') == 'qcache.receipt.1.3' and record.get('signature') == signature,
            f'Incompatible cached unit: {path}')
    require(record['data_sha256'] == _json_hash(record['data']), f'Payload checksum mismatch: {path}')
    for name, digest in record['artifacts'].items():
        require(artifact_root is not None, 'Artifact root required for receipt validation.')
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid receipt path.')
        target = artifact_root / relative
        require(target.is_file() and _file_hash(target) == digest, f'Missing/changed artifact: {target}')
    return record['data']


def logits_digest(logits: list[Tensor]) -> str:
    digest = hashlib.sha256()
    for row in logits:
        digest.update(_tensor_digest(row).encode('ascii'))
    return digest.hexdigest()


def trajectory_features(source, rc: RepeatConfig) -> dict[str, Any]:
    """Every signal is mechanical and trajectory-local. Never a quality score."""
    ids = source['token_ids']
    require(bool(ids), 'Empty scout result.')
    monitor = RepeatMonitor(rc)
    active_positions = 0
    first, signatures = None, set()
    for token in ids:
        monitor.step(token)
        if monitor.active:
            active_positions += 1
            signatures.add(tuple(monitor.active['canonical_unit']))
            if first is None:
                first = dict(detected_after_tokens=len(monitor.ids), **copy.deepcopy(monitor.active))
    rows = source.get('_rows', [])
    entropy = statistics.fmean(row['entropy_nats'] for row in rows) if rows else None
    margin = statistics.fmean(row['top1_top2_logit_margin'] for row in rows) if rows else None
    return dict(tokens=len(ids), stop_reason=source['stop_reason'],
                horizon_censored=source['stop_reason'] == 'max_new_tokens',
                repeat_observed=first is not None, first_repeat=first,
                periodic_suffix_fraction=active_positions / len(ids),
                distinct_observed_motifs=len(signatures),
                repeated_3gram_fraction=repetition_stats(ids)['repeated_3gram_fraction'],
                mean_own_history_entropy_nats=entropy, mean_own_history_top2_margin=margin,
                token_sha256=_json_hash(ids),
                note='Finite token periodicity, output length and own-history distributions; not semantic failure or recovery.')


def _ngram_distance(a: list[int], b: list[int], n: int = 2) -> float:
    aa = {tuple(a[i:i+n]) for i in range(max(0, len(a)-n+1))}
    bb = {tuple(b[i:i+n]) for i in range(max(0, len(b)-n+1))}
    union = aa | bb
    return 1 - len(aa & bb) / len(union) if union else float(a != b)


def neighbor_evidence(left, right, options) -> dict[str, Any]:
    a, b = left['features'], right['features']
    n = max(a['tokens'], b['tokens'])
    repeat_switch = a['repeat_observed'] != b['repeat_observed']
    stop_switch = a['stop_reason'] != b['stop_reason']
    coverage_delta = abs(a['periodic_suffix_fraction'] - b['periodic_suffix_fraction'])
    rep_delta = abs(a['repeated_3gram_fraction'] - b['repeated_3gram_fraction'])
    length_delta = abs(a['tokens'] - b['tokens']) / n
    distance = _ngram_distance(left['source']['token_ids'], right['source']['token_ids'])
    entropy_delta = abs(a['mean_own_history_entropy_nats'] - b['mean_own_history_entropy_nats']) \
        if a['mean_own_history_entropy_nats'] is not None and b['mean_own_history_entropy_nats'] is not None else 0.
    onset_delta, period_switch = 0., False
    if a['first_repeat'] and b['first_repeat']:
        onset_delta = abs(a['first_repeat']['detected_after_tokens'] - b['first_repeat']['detected_after_tokens']) / n
        period_switch = a['first_repeat']['period'] != b['first_repeat']['period']
    labels = []
    if repeat_switch: labels.append('repeat_observed_switch')
    if stop_switch: labels.append('eos_vs_horizon_switch')
    if coverage_delta >= options['repeat_jump'] or rep_delta >= options['repeat_jump']: labels.append('repeat_fraction_jump')
    if onset_delta >= options['onset_jump']: labels.append('repeat_detection_position_jump')
    if period_switch: labels.append('repeat_period_change')
    if length_delta >= options['length_jump']: labels.append('generated_length_jump')
    if entropy_delta >= options['entropy_jump']: labels.append('own_history_entropy_jump')
    if distance >= options['lexical_jump']: labels.append('token_bigram_set_change')
    weights = dict(repeat_observed_switch=6., eos_vs_horizon_switch=5., repeat_fraction_jump=4.,
                   repeat_detection_position_jump=3., repeat_period_change=2., generated_length_jump=3.,
                   own_history_entropy_jump=2., token_bigram_set_change=1.)
    # Deterministic acquisition heuristic only. Not probability, quality, or causal strength.
    priority = sum(weights[x] for x in labels)
    if labels:
        priority += .1 * (coverage_delta + rep_delta + length_delta + distance)
    prefix = free_prefix_comparison(left['source']['token_ids'], right['source']['token_ids'])
    return dict(labels=labels, priority=priority, repeat_fraction_delta=rep_delta,
                periodic_suffix_fraction_delta=coverage_delta, normalized_length_delta=length_delta,
                token_bigram_jaccard_distance=distance, own_history_entropy_delta_nats=entropy_delta,
                normalized_detection_position_delta=onset_delta,
                common_prefix_tokens=prefix['free_common_prefix_tokens'],
                token_ids_equal=left['source']['token_ids'] == right['source']['token_ids'])


def rank_intervals(scouts: dict[str, Any], tested: list[float], options) -> list[dict[str, Any]]:
    intervals = []
    for lo, hi in zip(sorted(tested), sorted(tested)[1:]):
        evidence = []
        for seed in options['seeds']:
            evidence.append(dict(seed=seed, **neighbor_evidence(
                scouts[f'{lambda_key(lo)}/{seed}'], scouts[f'{lambda_key(hi)}/{seed}'], options)))
        labels = sorted(set(x for row in evidence for x in row['labels']))
        intervals.append(dict(low=lo, high=hi, width=hi-lo, labels=labels,
                              priority=max(row['priority'] for row in evidence),
                              evidence=evidence, seed_support=sum(bool(row['labels']) for row in evidence),
                              seed_count=len(evidence),
                              interpretation='Candidate interval under this detector/horizon. No monotonicity, unique threshold, or significance assumed.'))
    return sorted(intervals, key=lambda x: (-x['priority'], -x['width'], x['low']))


def refinement_plan(intervals, tested, options) -> list[dict[str, Any]]:
    remaining = max(0, options['max_lambdas'] - len(tested))
    if remaining == 0:
        return []
    plan = []
    keys = {lambda_key(x) for x in tested}
    for interval in intervals:
        if not interval['labels'] or interval['width'] <= options['min_lambda_width']:
            continue
        midpoint = float((Decimal(str(interval['low'])) + Decimal(str(interval['high']))) / 2)
        require(interval['low'] < midpoint < interval['high'], 'Midpoint precision exhausted.')
        key = lambda_key(midpoint)
        if key in keys:
            continue
        plan.append(dict(lambda_value=midpoint, parent_interval=[interval['low'], interval['high']],
                         selection_labels=interval['labels'], priority=interval['priority']))
        keys.add(key)
        if len(plan) >= min(options['refine_intervals'], remaining):
            break
    return plan if remaining else []


def select_focus(intervals, options) -> dict[str, Any]:
    selected = [x for x in intervals if x['labels']][:options['focus_intervals']]
    values = sorted({x[side] for x in selected for side in ('low', 'high')})
    return dict(intervals=selected, lambda_values=values,
                status='candidate_intervals_selected' if selected else 'no_candidate_interval',
                interpretation='Both sampled endpoints selected; an interval is not a proven threshold. No fallback silently invents a transition.')


def plan_fork_points(scout, neighbors, rc, limit: int) -> dict[str, Any]:
    ids, features = scout['source']['token_ids'], scout['features']
    n = len(ids)
    proposals = []
    first = features['first_repeat']
    if first:
        period, start = first['period'], first['start']
        proposals = [(start+period, 'first_copy_end_retrospectively_located', period),
                     (first['detected_after_tokens'], 'first_online_repeat_detection', period),
                     (start+max(rc.copies+2, 6)*period, 'later_repeat_probe_not_assumed_to_persist', period)]
    else:
        prefixes = [free_prefix_comparison(ids, x['source']['token_ids'])['free_common_prefix_tokens'] for x in neighbors]
        different = [p for p in prefixes if p < n]
        if different:
            proposals.append((max(1, min(different)), 'neighbor_output_divergence_anchor', 0))
        if scout['source']['stop_reason'] == 'eos':
            proposals.append((n-1, 'pre_eos_boundary', 0))
        proposals += [(max(1,n//2), 'nonperiodic_midpoint_observation', 0), (n-1, 'late_horizon_observation', 0)]
    points, skipped, seen = [], [], set()
    for count, reason, period in proposals:
        if count in seen:
            continue
        seen.add(count)
        point = dict(generated_count=count, reason=reason, probe_period=period,
                     selection='scout-informed retrospective selection; source is regenerated freely, never teacher-forced')
        if not 1 <= count < n:
            skipped.append(dict(point, why='outside_nonterminal_source_prefix'))
        elif len(points) >= limit:
            skipped.append(dict(point, why='max_forks_budget'))
        else:
            points.append(point)
    return dict(points=points, skipped=skipped, repeat_based=first is not None,
                first_prediction_difference_not_reachable_if_common_prefix_zero=any(
                    free_prefix_comparison(ids, x['source']['token_ids'])['free_common_prefix_tokens']==0 for x in neighbors))


def _trajectory_kw(options, eos):
    return dict(max_new=options['max_new_tokens'], prefill_mode=options['prefill'],
                temperature=options['temperature'], top_p=options['top_p'], eos_ids=eos,
                repeat_config=RepeatConfig(options['min_period'], options['max_period'],
                                           options['repeat_copies'], options['min_repeat_tokens']))


def _scout_public(result, tokenizer, strength, seed, rc):
    public = public_trajectory(result, tokenizer)
    # The scouting phase has no capture callbacks or events requesting capture.
    return dict(lambda_value=strength, seed=seed, source=public,
                features=trajectory_features(result, rc), logits_sha256=logits_digest(result['_logits']))


def _artifact_inventory(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): _file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def _fresh_attempt(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    number = 1
    while (folder/f'attempt_{number:03d}').exists(): number += 1
    out = folder/f'attempt_{number:03d}'
    out.mkdir()
    return out


def _run_focus(runner, tokenizer, prompt, eos, options, scout, plan, out, configuration):
    """Re-generate source, prove correspondence, then restore genuine snapshots."""
    out.mkdir(parents=True, exist_ok=True)
    (out/'checkpoints').mkdir(exist_ok=True)
    key = f'lambda_{lambda_key(scout["lambda_value"])}_seed_{scout["seed"]}'
    report = dict(schema='qcache.free_tree.1.2', status='started', configuration=configuration,
                  controls=[], runs={}, automatic_fork_plan=plan)
    streams=[]
    old_trace, old_sink = runner.controller.trace_every, runner.controller.trace_sink
    def sink(name):
        stream=(out/name).open('w',encoding='utf-8');streams.append(stream)
        def emit(row): stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        return emit
    token_sink,event_sink,attention_sink = sink('token_trace.jsonl'),sink('events.jsonl'),sink('attention_trace.jsonl')
    runner.controller.trace_every=options['trace_every'];runner.controller.trace_sink=attention_sink
    points = {p['generated_count']:p for p in plan['points']}
    def capture(state, reasons):
        count = len(state['generated_ids'])
        info = save_checkpoint(out/'checkpoints'/f'g{count:05d}.json',state)
        return dict(info, automatic_selection=points[count])
    kw = _trajectory_kw(options,eos)
    try:
        print(f'  [capture] {key}: {sorted(points)}',flush=True)
        source=free_source(runner,LivePolicy(key,'replay',scout['lambda_value']),prompt,
                           seed=scout['seed'],**kw, fork_at=sorted(points),event_copies=(),
                           max_forks=len(points),checkpoint_sink=capture,
                           token_sink=token_sink,event_sink=event_sink,branch=key+'/source')
        tokens_match=source['token_ids']==scout['source']['token_ids']
        numeric_match=logits_digest(source['_logits'])==scout['logits_sha256']
        guard=dict(token_ids_identical=tokens_match, logits_digest_identical=numeric_match,
                   status='passed' if tokens_match and numeric_match else 'failed',
                   note='Independent free regeneration; no scout continuation was forced. A strict reproducibility check, not independent replication.')
        report['scout_reproduction']=guard
        if options['save_logits']:
            from safetensors.torch import save_file
            save_file({'logits':torch.stack(list(source['_logits']))},str(out/'source_logits.safetensors'))
        item=dict(source=public_trajectory(source,tokenizer),forks=[]);report['runs'][key]=item
        write_live_reports(out,report)
        require(tokens_match and numeric_match,
                'Scout regeneration changed tokens or logits. New source saved; forks were not treated as the original scout. Inspect hooks/device nondeterminism.')
        require({ck['generated_count'] for ck in source['checkpoints']}==set(points), 'Not all planned checkpoints were captured.')
        for ck in source['checkpoints']:
            state=load_checkpoint(Path(ck['path']));original_hash=state_digest(state)
            fork=dict(checkpoint=ck,prefix_text=tokenizer.decode(state['generated_ids'],skip_special_tokens=False),branches={})
            item['forks'].append(fork)
            continued=None
            for name in options['branches']:
                print(f'    [fork] g{ck["generated_count"]}: {name}',flush=True)
                tag=f'{key}/g{ck["generated_count"]}/{name}'
                result=free_branch(runner,state,name,max_new=options['branch_new_tokens'],
                                  temperature=options['temperature'],top_p=options['top_p'],eos_ids=eos,
                                  repeat_config=kw['repeat_config'],trace_label=tag,
                                  token_sink=lambda row:token_sink(dict(row,trajectory=tag)),
                                  event_sink=lambda row:event_sink(dict(row,trajectory=tag)),
                                  probe_period=points[ck['generated_count']]['probe_period'])
                require(state_digest(state)==original_hash,'Checkpoint mutated across branches.')
                if name=='continue':
                    continued=result
                    fork['restoration_control']=continuation_control(source,state,continued,options['control_atol'])
                else:
                    result['boundary_vs_continue']=boundary_comparison(continued['_logits'][0],result['_logits'][0],eos)
                    if name=='freeze':
                        delta=float((continued['_logits'][0]-result['_logits'][0]).abs().max())
                        require(delta<=options['control_atol'],'Freeze changed the first prediction before admission can act.')
                        result['freeze_first_prediction_control']=dict(status='passed',max_abs_logit_delta=delta)
                first=result['_rows'][0]
                result['first_prediction']=dict(first,top1_token=tokenizer.decode([first['top1_id']],skip_special_tokens=False))
                fork['branches'][name]=public_trajectory(result,tokenizer)
                if options['save_logits']:
                    from safetensors.torch import save_file
                    save_file({'logits':torch.stack(list(result['_logits']))},str(out/f'g{ck["generated_count"]}_{name}.safetensors'))
                write_live_reports(out,report)
                for stream in streams:stream.flush()
            del continued,state
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        runner.controller.trace_every=old_trace;runner.controller.trace_sink=old_sink
        for stream in streams:stream.close()
        write_live_reports(out,report)
    return dict(status=report['status'],scout_reproduction=report['scout_reproduction'],
                packet=_load_json(out/'share_packet.json'),report_path=(out/'reading_room.html').as_posix())


def write_case_reports(out: Path, report: dict[str, Any]) -> None:
    json_write(out/'campaign_results.json',report)
    scouts=list(report.get('scouts',{}).values())
    rows=[]
    for row in sorted(scouts,key=lambda x:(x['lambda_value'],x['seed'])):
        f=row['features'];repeat=f['first_repeat']
        rows.append(dict(lambda_value=row['lambda_value'],seed=row['seed'],tokens=f['tokens'],stop=f['stop_reason'],
                         repeat_observed=f['repeat_observed'],first_repeat_detection=None if repeat is None else repeat['detected_after_tokens'],
                         period=None if repeat is None else repeat['period'],periodic_suffix_fraction=f['periodic_suffix_fraction'],
                         repeated_3gram_fraction=f['repeated_3gram_fraction'],own_history_entropy=f['mean_own_history_entropy_nats']))
    write_csv_rows(out/'sweep_summary.csv',rows)
    intervals=report.get('final_intervals',[])
    write_csv_rows(out/'candidate_intervals.csv',[dict(low=x['low'],high=x['high'],priority=x['priority'],
                    labels=';'.join(x['labels']),seed_support=x['seed_support'],seed_count=x['seed_count']) for x in intervals])
    json_write(out/'selection_ledger.json',dict(rounds=report.get('rounds',[]),focus=report.get('focus'),
               final_intervals=intervals,stopping_reason=report.get('refinement_stopping_reason')))
    lines=['# QCache V1.3: sweep → candidate intervals → free forks','',f'Status: {report.get("status")}',
           '', 'Candidate intervals are heuristic observations, not universal thresholds or quality rankings.',
           'Different trajectories are not matched histories. No fixed future tokens. No automatic repeat suppression.',
           '', '| lambda | seed | tokens / stop | periodic suffix fraction | repeat-3gram |',
           '|---|---|---|---|---|']
    for r in rows:
        lines.append(f'| {r["lambda_value"]:g} | {r["seed"]} | {r["tokens"]} / {r["stop"]} | {r["periodic_suffix_fraction"]:.4f} | {r["repeated_3gram_fraction"]:.4f} |')
    lines+=['','## Candidate intervals','']
    for x in intervals:
        if x['labels']:lines.append(f'{x["low"]:g} … {x["high"]:g}: '+', '.join(x['labels']))
    if not any(x['labels'] for x in intervals):lines.append('No candidate interval observed under the chosen grid, horizon and detector.')
    for row in sorted(scouts,key=lambda x:(x['lambda_value'],x['seed'])):
        lines+=['',f'## λ={row["lambda_value"]:g}, seed={row["seed"]}', '', '```text',row['source']['text'].replace('```',"'''"),'```']
    for key,focus in report.get('focused_runs',{}).items():
        lines+=['',f'## Forks: {key}', '', 'Report: '+focus['relative_report_path'],
                'Scouting reproduction: '+focus['scout_reproduction']['status']]
    (out/'campaign.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    e=html.escape
    page=['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
          '<title>QCache V1.4 campaign</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem}table{border-collapse:collapse;width:100%}td,th{border:1px solid;padding:.5rem;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid;padding:1rem}details{margin:1rem 0}summary{cursor:pointer}small{display:block}</style>',
          '<h1>QCache V1.4</h1><p>Free sweep → candidate intervals → same-state forks</p>',
          '<p>Status: '+e(report.get('status',''))+'</p><p>候補区間は観測条件に依存する探索目印です。品質順位や唯一の閾値ではありません。反復は抑制せず記録します。</p>',
          '<h2>Sweep</h2><table><tr><th>λ</th><th>seed</th><th>長さ / 終了</th><th>周期検出</th><th>3-gram重複</th><th>枝</th></tr>']
    for row in rows:
        key=f'{lambda_key(row["lambda_value"])}/{row["seed"]}'
        focus=report.get('focused_runs',{}).get(key)
        link=''
        if focus:link='<a href="'+e(focus['relative_report_path'],quote=True)+'">forkを読む</a>'
        page.append(f'<tr><td>{row["lambda_value"]:g}</td><td>{row["seed"]}</td><td>{row["tokens"]} / {e(row["stop"])}</td><td>{row["first_repeat_detection"]} (period {row["period"]})</td><td>{row["repeated_3gram_fraction"]:.3f}</td><td>{link}</td></tr>')
    page+=['</table><h2>Candidate intervals</h2>']
    for x in intervals:
        if x['labels']:page.append(f'<p>{x["low"]:g} … {x["high"]:g}: '+e(', '.join(x['labels']))+'</p>')
    if not any(x['labels'] for x in intervals):page.append('<p>No candidate interval observed. Grid resolution and output horizon remain limits.</p>')
    for row in sorted(scouts,key=lambda x:(x['lambda_value'],x['seed'])):
        page.append(f'<details><summary>λ={row["lambda_value"]:g}, seed={row["seed"]}</summary><pre>'+e(row['source']['text'])+'</pre></details>')
    page.append('</html>');(out/'index.html').write_text(''.join(page),encoding='utf-8')
    packet=dict(schema='qcache.campaign_case_packet.1.3',status=report.get('status'),
                configuration=report.get('configuration'),controls=report.get('controls'),
                rounds=report.get('rounds'),focus=report.get('focus'),
                scouts=[dict(lambda_value=x['lambda_value'],seed=x['seed'],features=x['features'],
                             token_ids=x['source']['token_ids'],text=x['source']['text'],logits_sha256=x['logits_sha256']) for x in scouts],
                focused_runs=report.get('focused_runs'),refinement_stopping_reason=report.get('refinement_stopping_reason'))
    json_write(out/'campaign_packet.json',packet)


def campaign_case(runner, tokenizer, prompt, eos, options, out: Path, signature: str, configuration,
                  controls, resume=False):
    """Hardware-agnostic orchestrator. The runner performs real free forwards."""
    out.mkdir(parents=True,exist_ok=True);(out/'scouts').mkdir(exist_ok=True)
    report=dict(schema=CAMPAIGN_SCHEMA,status='started',configuration=configuration,controls=controls,
                scouts={},rounds=[],focused_runs={})
    tested=[]
    kw=_trajectory_kw(options,eos);rc=kw['repeat_config']
    old_trace,old_sink=runner.controller.trace_every,runner.controller.trace_sink
    runner.controller.trace_every=0;runner.controller.trace_sink=None
    def evaluate(value, stage):
        for seed in options['seeds']:
            key=f'{lambda_key(value)}/{seed}'
            path=out/'scouts'/f'lambda_{lambda_key(value)}_seed_{seed}.json'
            unit_sig=_json_hash(dict(case=signature,lambda_value=value,seed=seed))
            if path.exists():
                require(resume,'Scout receipt already exists; use --resume with unchanged settings.')
                print(f'  [reuse] {key}',flush=True)
                data=read_receipt(path,unit_sig,out/'scouts')
            else:
                print(f'  [{stage}] λ={value:g}, seed={seed}',flush=True)
                result=free_source(runner,LivePolicy('scout','replay',value),prompt,seed=seed,
                                   **kw,max_forks=0,event_copies=())
                data=_scout_public(result,tokenizer,value,seed,rc)
                artifacts={}
                if options['save_logits']:
                    from safetensors.torch import save_file
                    logit_path=path.with_suffix('.safetensors')
                    temporary=logit_path.with_suffix('.safetensors.tmp')
                    save_file({'logits':torch.stack(list(result['_logits']))},str(temporary))
                    temporary.replace(logit_path)
                    artifacts[logit_path.name]=_file_hash(logit_path)
                write_receipt(path,unit_sig,data,artifacts)
                del result
            report['scouts'][key]=data
            write_case_reports(out,report)
        if value not in tested:tested.append(value)
    try:
        for value in options['coarse_lambdas']:evaluate(value,'coarse')
        for number in range(options['refine_rounds']):
            intervals=rank_intervals(report['scouts'],tested,options)
            plan=refinement_plan(intervals,tested,options)
            report['rounds'].append(dict(round=number+1,tested_before=sorted(tested),ranked_intervals=intervals,plan=plan))
            write_case_reports(out,report)
            if not plan:
                report['refinement_stopping_reason']='lambda_budget' if len(tested)>=options['max_lambdas'] else \
                    'no_candidate_interval' if not any(x['labels'] for x in intervals) else 'minimum_width_or_no_new_midpoint'
                break
            print(f'  [refine round {number+1}] '+', '.join(lambda_key(x['lambda_value']) for x in plan),flush=True)
            for item in plan:evaluate(item['lambda_value'],'refine')
        else:report['refinement_stopping_reason']='round_budget'
        report['final_intervals']=rank_intervals(report['scouts'],tested,options)
        report['focus']=select_focus(report['final_intervals'],options)
        write_case_reports(out,report)
        if options['max_forks']:
            for value in report['focus']['lambda_values']:
                for seed in options['seeds']:
                    key=f'{lambda_key(value)}/{seed}';scout=report['scouts'][key]
                    neighbors=[report['scouts'][f'{lambda_key(x[side])}/{seed}']
                               for x in report['focus']['intervals'] if value in (x['low'],x['high'])
                               for side in ('low','high') if x[side]!=value]
                    fork_plan=plan_fork_points(scout,neighbors,rc,options['max_forks'])
                    if not fork_plan['points']:
                        report.setdefault('forks_unavailable',{})[key]=fork_plan
                        continue
                    job_dir=out/'focused'/f'lambda_{lambda_key(value)}_seed_{seed}'
                    receipt=job_dir/'completed.json'
                    job_sig=_json_hash(dict(case=signature,key=key,plan=fork_plan,scout=scout['features']['token_sha256']))
                    if receipt.exists():
                        require(resume,'Focus receipt already exists; --resume required.')
                        focus=read_receipt(receipt,job_sig,job_dir)
                        print(f'  [reuse forks] {key}',flush=True)
                    else:
                        attempt=_fresh_attempt(job_dir)
                        focus=_run_focus(runner,tokenizer,prompt,eos,options,scout,fork_plan,attempt,configuration)
                        focus['relative_report_path']=(attempt/'reading_room.html').relative_to(out).as_posix()
                        focus['fork_plan']=fork_plan
                        inventory={f'{attempt.name}/{k}':v for k,v in _artifact_inventory(attempt).items()}
                        write_receipt(receipt,job_sig,focus,inventory)
                    report['focused_runs'][key]=focus
                    write_case_reports(out,report)
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        (out/'campaign_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        runner.controller.trace_every=old_trace;runner.controller.trace_sink=old_sink
        write_case_reports(out,report)
    return report


def model_provenance(model_spec, config, tokenizer):
    """Resume binding; local weights use inventory, not a false cryptographic claim."""
    folder=Path(model_spec['path']).expanduser()
    inventory=[]
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.is_file() and (path.suffix in {'.safetensors','.json','.model'} or path.name.startswith('tokenizer')):
                stat=path.stat();item=dict(name=path.name,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
                if path.suffix!='.safetensors' and stat.st_size<=20*1024*1024:item['sha256']=_file_hash(path)
                inventory.append(item)
    vocab=tokenizer.get_vocab()
    return dict(model_spec=model_spec,config=config.to_dict(),resolved_revision=getattr(config,'_commit_hash',None),
                local_inventory=inventory,tokenizer_vocab_sha256=_json_hash(vocab),
                tokenizer_template_sha256=_json_hash(getattr(tokenizer,'chat_template',None)),
                limit='Local weight bytes are NOT fully hashed: names, sizes and mtimes bind resume. Config/tokenizer hashes and revision are also checked. Not adversarial authenticity.')


def _encode_campaign_prompt(tokenizer, prompt_spec, options, config):
    if 'token_ids' in prompt_spec:
        ids=list(prompt_spec['token_ids'])
    elif prompt_spec['chat']:
        require(bool(tokenizer.chat_template),'Selected tokenizer has no chat template.')
        extra={} if options['chat_date'] is None else {'date_string':options['chat_date']}
        ids=tokenizer.apply_chat_template([dict(role='user',content=prompt_spec['text'])],
                                          tokenize=True,add_generation_prompt=True,**extra)
    else:ids=tokenizer.encode(prompt_spec['text'],add_special_tokens=True)
    require(isinstance(ids,list) and ids and all(type(x) is int and 0<=x<config.vocab_size for x in ids),
            'Prompt IDs are invalid for this model vocabulary.')
    return ids


def _release_device(device):
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    elif device.type=='mps':torch.mps.empty_cache()


def write_campaign_index(out: Path, campaign):
    json_write(out/'campaign.json',campaign)
    page=['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
          '<title>QCache V1.3 experiments</title><style>body{font:17px system-ui;max-width:1100px;margin:3rem auto;padding:1rem}td,th{padding:.8rem;border:1px solid}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
          '<h1>QCache V1.3 · prompt × model</h1><p>Status: '+html.escape(campaign['status'])+'</p>',
          '<p>粗い自由生成 → 候補区間 → 局所的な追加探索 → 状態fork → 自由な続きを観察</p>',
          '<p>候補は探索用の目印です。唯一の閾値・性能順位・意味的回復の判定ではありません。</p>',
          '<table><tr><th>Model</th><th>Prompt</th><th>Status</th><th>Report</th></tr>']
    for item in campaign.get('cases',{}).values():
        page+=['<tr><td>'+html.escape(item['model_id'])+'</td><td>'+html.escape(item['prompt_id'])+'</td><td>'+html.escape(item['status'])+
               '</td><td><a href="'+html.escape(item['relative_report_path'],quote=True)+'">sweep / forks</a></td></tr>']
    page+=['</table><h2>Budget</h2><pre>'+html.escape(json.dumps(campaign['budget'],ensure_ascii=False,indent=2))+'</pre></html>']
    (out/'index.html').write_text(''.join(page),encoding='utf-8')
    packet=dict(schema='qcache.campaign_packet.1.3',status=campaign['status'],request=campaign['request'],
                budget=campaign['budget'],environment=campaign.get('environment'),cases={})
    for key,item in campaign.get('cases',{}).items():
        p=out/Path(item['relative_report_path']).parent/'campaign_packet.json'
        if p.exists():packet['cases'][key]=_load_json(p)
    json_write(out/'campaign_packet.json',packet)


def run_campaign(args):
    suite=campaign_suite(args);options=campaign_options(args)
    requested_options=copy.deepcopy(options)
    request=dict(suite=suite,options=options)
    budget=campaign_budget(options,len(suite['models'])*len(suite['prompts']))
    if args.dry_run:
        print(json.dumps(dict(request=request,budget=budget,model_loaded=False,files_written=False),ensure_ascii=False,indent=2))
        return
    out=Path(args.out).expanduser().resolve()
    require(not out.exists() or out.is_dir(),'Output must be a directory.')
    if not args.resume:require(not out.exists() or not any(out.iterdir()),'Nonempty output directory; choose a new path or explicit --resume.')
    if args.resume:require((out/'campaign.json').is_file(),'--resume requires an existing campaign.json.')
    out.mkdir(parents=True,exist_ok=True)
    lock=out/'.campaign.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise ValueError('Campaign lock exists. Do not run concurrently. If a previous process was killed, verify it is stopped before removing the lock.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    campaign=None
    try:
        env=environment()
        root_sig=_json_hash(dict(request=request,script=env['script_sha256'],python=sys.version,
                                 torch=env['torch'],transformers=env['transformers'],device_runtime=env['device_runtime'],
                                 ambient_threads=torch.get_num_threads() if not options['threads'] else options['threads']))
        if args.resume:
            old=_load_json(out/'campaign.json')
            require(old.get('signature')==root_sig,'Resume settings/code/runtime changed. Use a new directory; old results are not mixed.')
            campaign=old
        else:
            campaign=dict(schema=CAMPAIGN_SCHEMA,status='started',signature=root_sig,request=request,budget=budget,
                          environment=env,started_utc=datetime.now(timezone.utc).isoformat(),cases={})
        campaign['status']='running';write_campaign_index(out,campaign)
        hf=require_hf();device=choose_device(args.device);_cpu_metric_check()
        if options['threads']:torch.set_num_threads(options['threads'])
        print('[campaign] generated-token upper bound:',budget['generated_token_upper_bound'],flush=True)
        if options['temperature']==0 and len(options['seeds'])>1:
            print('[note] Multiple greedy seeds are repeatability checks, not independent diverse samples.',flush=True)
        for model_spec in suite['models']:
            model=None;ctrl=None;runner=None;adapter=None
            try:
                load=dict(local_files_only=options['local_files_only'],trust_remote_code=False,revision=model_spec['revision'])
                config=hf.AutoConfig.from_pretrained(model_spec['path'],**load);validate_config(config)
                tokenizer=hf.AutoTokenizer.from_pretrained(model_spec['path'],**load)
                provenance=model_provenance(model_spec,config,tokenizer)
                layers=parse_layers(options['layers'],config.num_hidden_layers)
                dtype_name=options['dtype'] if options['dtype']!='auto' else ('float32' if device.type=='cpu' else 'float16')
                print(f'[load once] {model_spec["id"]}: {model_spec["path"]} / {device} / {dtype_name}',flush=True)
                torch.manual_seed(options['seeds'][0])
                model=hf.AutoModelForCausalLM.from_pretrained(model_spec['path'],config=config,dtype=getattr(torch,dtype_name),
                         attn_implementation='eager',use_safetensors=True,**load).to(device).eval()
                model.requires_grad_(False)
                mismatch=[(n,str(t.device)) for n,t in itertools.chain(model.named_parameters(),model.named_buffers()) if
                          t.device.type!=device.type or (device.type=='cuda' and t.device.index!=
                            (device.index if device.index is not None else torch.cuda.current_device()))]
                require(not mismatch,f'Model placement mismatch: {mismatch[:5]}')
                versions={n:p._version for n,p in model.named_parameters()}
                ctrl=LiveController(layers=layers,max_tokens=(options['max_context'] if isinstance(options['max_context'],int) else 512),engine=options['engine'],chunk=options['chunk'],trace_every=0,verify_every=options['verify_every'])
                runner=LiveRunner(model,ctrl,device)
                raw_eos=getattr(model.generation_config,'eos_token_id',None)
                if raw_eos is None:raw_eos=tokenizer.eos_token_id
                eos=set(raw_eos if isinstance(raw_eos,list) else [] if raw_eos is None else [raw_eos])
                for prompt_spec in suite['prompts']:
                    options=copy.deepcopy(requested_options)
                    key=model_spec['id']+'/'+prompt_spec['id'];case_out=out/'cases'/model_spec['id']/prompt_spec['id']
                    case_out.mkdir(parents=True,exist_ok=True)
                    input_sig=_json_hash(dict(root=root_sig,model=provenance,prompt=prompt_spec,device=str(device),dtype=dtype_name))
                    input_path=case_out/'input.json'
                    if input_path.exists():
                        require(args.resume,'Input receipt exists.')
                        saved=read_receipt(input_path,input_sig)
                        prompt=saved['prompt_token_ids']
                    else:
                        prompt=_encode_campaign_prompt(tokenizer,prompt_spec,options,config)
                        saved=dict(prompt_token_ids=prompt,model_provenance=provenance,prompt=prompt_spec,
                                   model_placement=dict(parameters=sorted({str(p.device) for p in model.parameters()}),
                                                        buffers=sorted({str(b.device) for b in model.buffers()})))
                        write_receipt(input_path,input_sig,saved)
                    reserve=options['branch_new_tokens'] if options['focus_intervals'] and options['max_forks'] else 0
                    total=len(prompt)+options['max_new_tokens']+reserve
                    lengths=resolve_context(options['max_context'],config,len(prompt),options['max_new_tokens'],reserve)
                    options['context_request']=options['max_context']
                    options['max_context']=lengths['effective_context']
                    options['length_resolution']=lengths
                    ctrl.max_tokens=lengths['effective_context']
                    print('[lengths] '+json.dumps(lengths),flush=True)
                    case_sig=_json_hash(dict(input_signature=input_sig,prompt_token_ids=prompt))
                    configuration=dict(model=model_spec,prompt=prompt_spec,prompt_token_ids=prompt,options=options,
                                       script_version=VERSION,case_signature=case_sig,device=str(device),dtype=dtype_name,
                                       model_provenance=provenance,eos_token_ids=sorted(eos),
                                       selection='Adaptive exploratory acquisition; no statistical threshold confidence claimed.',
                                       no_fixed_future_tokens=True)
                    campaign['cases'][key]=dict(model_id=model_spec['id'],prompt_id=prompt_spec['id'],status='running',
                                                relative_report_path=(case_out/'index.html').relative_to(out).as_posix())
                    write_campaign_index(out,campaign)
                    # Native runs happen without the adapter for every prompt/seed.
                    if adapter is not None:adapter.close();adapter=None
                    native={}
                    control_kw=dict(max_new=min(options['control_tokens'],options['max_new_tokens']),
                                    prefill_mode=options['prefill'],temperature=options['temperature'],top_p=options['top_p'],eos_ids=eos,max_forks=0,event_copies=())
                    print(f'[controls] {key}',flush=True)
                    for seed in options['seeds']:
                        native[seed]=free_source(runner,LivePolicy('native','baseline',0),prompt,seed=seed,**control_kw)
                    adapter=HFAdapter(model,ctrl)
                    if options['trace_projection']:
                        ctrl.projectors=attention_projectors(model,layers)
                    controls=[]
                    for seed in options['seeds']:
                        checks=null_free_controls(runner,prompt,max_new=control_kw['max_new'],prefill_mode=options['prefill'],
                                  seed=seed,eos_ids=eos,atol=options['control_atol'],native=native.pop(seed),
                                  temperature=options['temperature'],top_p=options['top_p'])
                        controls.append(dict(seed=seed,checks=checks))
                    configuration['capabilities']=capability_report(model)
                    configuration['research']=asdict(ctrl.research)
                    report=campaign_case(runner,tokenizer,prompt,eos,options,case_out,case_sig,configuration,controls,args.resume)
                    campaign['cases'][key].update(status=report['status'],focus=report.get('focus'))
                    write_campaign_index(out,campaign)
                require(all(versions[n]==p._version for n,p in model.named_parameters()),'Parameter version counter changed.')
            finally:
                if adapter is not None:adapter.close()
                if runner is not None:runner.cache=None
                if ctrl is not None:
                    ctrl.banks.clear();ctrl.projectors.clear()
                runner=ctrl=adapter=model=None
                _release_device(device)
        campaign.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        if campaign is not None:
            campaign.update(status='failed',error=repr(exc))
            for case in campaign['cases'].values():
                if case['status']=='running':case['status']='failed'
            (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        try:
            if campaign is not None:write_campaign_index(out,campaign)
        finally:lock.unlink(missing_ok=True)
    print(f'Completed: {out/"index.html"}; share: {out/"campaign_packet.json"}',flush=True)


def campaign_parser():
    p=argparse.ArgumentParser(description='QCache V1.3: prompt x model -> free sweep -> candidate intervals -> refine -> same-state forks -> observation.')
    p.add_argument('--model');p.add_argument('--suite',help='JSON {models:[{id,path}],prompts:[{id,text,chat}]} cross product.')
    p.add_argument('--revision',default='main');p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--prompt');p.add_argument('--prompt-file');p.add_argument('--prompt-token-ids')
    p.add_argument('--chat',action='store_true');p.add_argument('--chat-date')
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--dtype',default='auto',choices=['auto','float32','float16','bfloat16'])
    p.add_argument('--coarse-lambdas',default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
    p.add_argument('--seeds',default='42');p.add_argument('--branches',default='continue,cut,native,parallel,freeze')
    p.add_argument('--pulse-steps',default='')
    for name,default in [('refine-rounds',2),('refine-intervals',3),('max-lambdas',17),('focus-intervals',1),
                         ('max-forks',2),('max-new-tokens',192),('branch-new-tokens',96),('max-context',512),
                         ('chunk',64),('threads',0),('trace-every',8),('verify-every',0),('control-tokens',24),
                         ('min-period',1),('max-period',64),('repeat-copies',3),('min-repeat-tokens',24)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('min-lambda-width',.0125),('repeat-jump',.15),('length-jump',.2),('lexical-jump',.55),
                         ('entropy-jump',.5),('onset-jump',.125),('temperature',0.),('top-p',1.),('control-atol',1e-6)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--layers',default='all');p.add_argument('--prefill',default='clean',choices=['clean','stream'])
    p.add_argument('--engine',default='online',choices=['online','dense'])
    p.add_argument('--trace-projection',action='store_true');p.add_argument('--save-logits',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--out',default='results_qcache_v13')
    return p


def v13_main():
    if len(sys.argv)==1 or sys.argv[1] in {'--help','-h'}:
        print('QCache Lab V1.3\n\nCommands:\n  explore       Prompt/model -> free sweep -> refine candidates -> forks -> observation\n  run           V1.2 manual free sources and forks\n  self-test     Retained numerical/state/branch tests\n  campaign-test Planner, resume, and campaign integration tests\n  hf-smoke      Actual HF tiny-model adapter/state tests\n  device-check  CPU metric placement under ambient device hooks\n  inspect       Inspect saved output for exact token repeats\n  legacy-run    Explicit opt-in fixed-token v0.1.2 comparison\n\nUse explore --help for the new end-to-end entry; explore --dry-run to inspect budgets.')
        return 0
    if len(sys.argv)>1 and sys.argv[1]=='explore':
        args=campaign_parser().parse_args(sys.argv[2:])
        try:run_campaign(args);return 0
        except Exception as exc:print('ERROR: '+str(exc),file=sys.stderr);traceback.print_exc();return 2
    if len(sys.argv)>1 and sys.argv[1]=='campaign-test':
        p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('qcache_v13_campaign_tests.json'))
        p.add_argument('--non-cpu-default',action='store_true');args=p.parse_args(sys.argv[2:])
        with torch.device('meta' if args.non_cpu_default else torch.get_default_device()):
            return 0 if campaign_tests(args.out) else 1
    return live_main()


def campaign_tests(out: Path) -> bool:
    """Planner/failure tests plus real free-forward toy integration; NOT HF language tests."""
    from unittest.mock import patch
    from types import SimpleNamespace
    torch.set_num_threads(min(2,torch.get_num_threads()))
    tests=[]
    def test(name,fn):
        start=time.perf_counter()
        try:
            with torch.inference_mode():detail=fn()
            tests.append(dict(name=name,status='passed',seconds=time.perf_counter()-start,detail=detail))
            print('PASS campaign/'+name,flush=True)
        except Exception as exc:
            tests.append(dict(name=name,status='failed',error=repr(exc),traceback=traceback.format_exc()))
            print('FAIL campaign/'+name+': '+repr(exc),flush=True)
    def rejected(fn):
        try:fn()
        except (ValueError,FileNotFoundError):return dict(rejected=True)
        raise AssertionError('Expected validation failure.')
    def opts(**changes):
        args=campaign_parser().parse_args(['--model','toy','--prompt','example'])
        for k,v in changes.items():setattr(args,k,v)
        return campaign_options(args)
    class Tok:
        chat_template='testing only'
        eos_token_id=None
        def decode(self,ids,skip_special_tokens=False):return ' '.join('T'+str(x) for x in ids)
        def encode(self,text,add_special_tokens=True):return [1,3,5,7]
        def apply_chat_template(self,*a,**kw):return [1,8,6,4,2]
        def get_vocab(self):return {f'T{i}':i for i in range(97)}
    def fake_scout(ids,strength=0.,seed=42,stop='max_new_tokens',rc=None):
        rc=rc or RepeatConfig(1,16,3,12)
        source=dict(token_ids=ids,stop_reason=stop,_rows=[dict(entropy_nats=1.,top1_top2_logit_margin=1.) for _ in ids])
        return dict(lambda_value=strength,seed=seed,source=dict(token_ids=ids,stop_reason=stop,text='test only'),
                    features=trajectory_features(source,rc),logits_sha256='fixture')
    def planner_islands():
        o=opts(coarse_lambdas='0,0.25,0.5,0.75,1',max_lambdas=9,refine_intervals=4,lexical_jump=1.)
        grid=o['coarse_lambdas'];rows={}
        for i,value in enumerate(grid):
            ids=list(range(36)) if i%2==0 else [1,2,3,4]*9
            rows[f'{lambda_key(value)}/42']=fake_scout(ids,value)
        intervals=rank_intervals(rows,grid,o)
        assert len(intervals)==4 and all('repeat_observed_switch' in x['labels'] for x in intervals)
        plan=refinement_plan(intervals,grid,o)
        assert len(plan)==4 and len({x['lambda_value'] for x in plan})==4
        assert set(x['lambda_value'] for x in plan)=={.125,.375,.625,.875}
        return dict(nonmonotonic_switches_preserved=4,midpoints=[x['lambda_value'] for x in plan])
    test('multiple_nonmonotonic_islands_are_refined',planner_islands)
    def no_candidates():
        o=opts();grid=o['coarse_lambdas'];rows={f'{lambda_key(v)}/42':fake_scout(list(range(36)),v) for v in grid}
        intervals=rank_intervals(rows,grid,o)
        assert not any(x['labels'] for x in intervals)
        assert refinement_plan(intervals,grid,o)==[]
        assert select_focus(intervals,o)['lambda_values']==[]
        return dict(no_transition_invented=True)
    test('identical_outputs_produce_no_candidates_or_forced_forks',no_candidates)
    def budget_test():
        o=opts(max_lambdas=12,refine_intervals=3)
        grid=o['coarse_lambdas'];rows={f'{lambda_key(v)}/42':fake_scout(([1,2,3,4]*9 if v>=.5 else list(range(36))),v) for v in grid}
        plan=refinement_plan(rank_intervals(rows,grid,o),grid,o)
        assert len(plan)==1
        grid.append(plan[0]['lambda_value']);assert refinement_plan([],grid,o)==[]
        bound=campaign_budget(opts(),1)
        assert bound['max_scout_trajectories']==17 and bound['max_branch_trajectories']==20
        return bound
    test('budgets_cap_refinements_forks_and_generated_tokens',budget_test)
    def resolution():
        o=opts(min_lambda_width=.025)
        assert refinement_plan([dict(low=.5,high=.51,width=.01,labels=['repeat_observed_switch'],priority=6)], [.5,.51],o)==[]
        a=refinement_plan([dict(low=.5,high=.55,width=.05,labels=['repeat_observed_switch'],priority=6)], [.5,.55],o)
        assert a[0]['lambda_value']==.525
        return dict(decimal_midpoint=.525)
    test('midpoints_and_minimum_width_are_explicit',resolution)
    def seed_test():
        o=opts(coarse_lambdas='0,1',seeds='42,43',max_lambdas=3)
        rows={}
        for value,seed in itertools.product(o['coarse_lambdas'],o['seeds']):
            ids=[1,2,3,4]*9 if value==1 and seed==43 else list(range(36))
            rows[f'{lambda_key(value)}/{seed}']=fake_scout(ids,value,seed)
        x=rank_intervals(rows,o['coarse_lambdas'],o)[0]
        assert x['seed_support']==1 and x['seed_count']==2
        return dict(support=1,total=2,no_confidence_interval=True)
    test('seed_evidence_not_collapsed_to_false_population_claim',seed_test)
    def forks_repeat():
        ids=list(range(100,138))+[1,2,3,4,5,6,7,8,9,10,11,12,13,14]*11
        rc=RepeatConfig(1,64,3,24);s=fake_scout(ids,.55,rc=rc)
        plan=plan_fork_points(s,[],rc,3)
        assert [x['generated_count'] for x in plan['points']]==[52,80,122]
        assert all(x['probe_period']==14 for x in plan['points'])
        return plan
    test('loop_entry_detection_and_late_probe_counts',forks_repeat)
    def no_repeats_fork():
        a=fake_scout(list(range(10)),.3,stop='eos');b=fake_scout(list(range(6))+[60,61,62,63],.4)
        p=plan_fork_points(a,[b],RepeatConfig(),2)
        assert [x['generated_count'] for x in p['points']]==[6,9]
        assert not p['repeat_based']
        return p
    test('nonperiodic_lexical_and_pre_eos_anchors',no_repeats_fork)
    def earlyeos():
        a=fake_scout([9],.2,stop='eos');p=plan_fork_points(a,[],RepeatConfig(),2)
        assert p['points']==[] and p['skipped']
        return dict(no_post_eos_fork=True)
    test('one_token_eos_has_no_fabricated_snapshot',earlyeos)
    def minperiod():
        a=periodic_suffix([2]*24,RepeatConfig(1,64,3,24))
        b=periodic_suffix([2]*24,RepeatConfig(4,64,3,24))
        assert a['period']==1 and b is None
        return dict(new_explore_default_includes_short_periods=True,old_detector_unchanged=True)
    test('short_period_observer_is_explicit_not_suppression',minperiod)
    test('reject_nan_lambda',lambda:rejected(lambda:parse_campaign_lambdas('0,nan')))
    test('reject_small_grid_budget',lambda:rejected(lambda:opts(max_lambdas=2)))
    test('reject_negative_seed',lambda:rejected(lambda:_seeds('-1')))
    test('reject_empty_seed_set',lambda:rejected(lambda:_seeds('')))
    test('reject_unknown_branch',lambda:rejected(lambda:_branches('unknown','')))
    test('reject_path_traversal_suite_id',lambda:rejected(lambda:_safe_id('../x')))
    test('reject_missing_prompt',lambda:rejected(lambda:campaign_suite(campaign_parser().parse_args(['--model','toy']))))
    def receipt_test():
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'r.json';a=root/'a.txt';a.write_text('data')
            write_receipt(p,'sig',dict(ok=True),{'a.txt':_file_hash(a)})
            assert read_receipt(p,'sig',root)==dict(ok=True)
            rejected(lambda:read_receipt(p,'other',root))
            a.write_text('bad');rejected(lambda:read_receipt(p,'sig',root))
            a.write_text('data');doc=_load_json(p);doc['data']['ok']=False;json_write(p,doc)
            rejected(lambda:read_receipt(p,'sig',root))
        return dict(roundtrip=True,signature_artifact_and_payload_changes_rejected=True)
    test('receipts_check_data_and_artifact_inventory',receipt_test)
    def json_duplicates():
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'dup.json';p.write_text('{"models":[],"models":[]}')
            return rejected(lambda:_load_json(p))
    test('duplicate_json_keys_rejected',json_duplicates)
    def suite_roundtrip():
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'suite.json';json_write(p,dict(models=[dict(id='a',path='one'),dict(id='b',path='two')],
                                                     prompts=[dict(id='x',text='abc',chat=True),dict(id='y',text='def')]))
            args=campaign_parser().parse_args(['--suite',str(p)])
            s=campaign_suite(args);assert len(s['models'])*len(s['prompts'])==4
            assert s['prompts'][0]['chat'] and not s['prompts'][1]['chat']
            return dict(cases=4)
    test('suite_cross_product_inputs',suite_roundtrip)
    def lexical_signal():
        o=opts();a=fake_scout(list(range(36)));b=fake_scout(list(range(50,86)),.1)
        e=neighbor_evidence(a,b,o)
        assert e['labels']==['token_bigram_set_change']
        return e
    test('nonrepeating_text_change_can_trigger_exploration',lexical_signal)
    def eos_signal():
        o=opts();a=fake_scout(list(range(36)));b=fake_scout(list(range(8)),.1,stop='eos')
        e=neighbor_evidence(a,b,o)
        assert 'eos_vs_horizon_switch' in e['labels'] and 'generated_length_jump' in e['labels']
        assert a['features']['horizon_censored'] and not b['features']['horizon_censored']
        return e
    test('eos_and_horizon_censoring_are_distinct',eos_signal)

    # Actual toy tensor model forwards, not precomputed source rows.
    def integration(temperature=0.,resume_only=False,artifact_tamper=False):
        torch.manual_seed(128);model=_ToyLM().eval();model.requires_grad_(False)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        ctrl=LiveController(max_tokens=96,trace_every=0);runner=LiveToyRunner(model,ctrl)
        o=opts(coarse_lambdas='0,0.5,1',max_lambdas=4,refine_rounds=1,refine_intervals=1,
               focus_intervals=1,max_forks=1,max_new_tokens=24,branch_new_tokens=8,max_context=96,
               lexical_jump=.001,repeat_jump=.01,entropy_jump=1e-8,control_tokens=4,trace_every=2,temperature=temperature)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);result=campaign_case(runner,Tok(),[1,3,5],set(),o,folder,'toy_signature',{'scope':'random toy'},[])
            assert result['status']=='completed' and len(result['scouts'])<=4
            assert result['focused_runs'],'Toy should yield a mechanically selected candidate.'
            for focus in result['focused_runs'].values():
                assert focus['scout_reproduction']['status']=='passed'
                for item in focus['packet']['runs'].values():
                    for fork in item['forks']:
                        assert fork['restoration_control']['status']=='passed'
                        assert set(fork['branches'])==set(o['branches'])
            if artifact_tamper:
                receipt=next(folder.glob('focused/*/completed.json'))
                record=_load_json(receipt);rel=next(iter(record['artifacts']));path=receipt.parent/rel
                path.write_bytes(path.read_bytes()+b'altered')
                rejected(lambda:campaign_case(runner,Tok(),[1,3,5],set(),o,folder,'toy_signature',{},[],True))
            else:
                # Reused units must need zero new source/fork generation calls.
                def forbidden(*a,**kw):raise AssertionError('Cached completed unit was executed again.')
                with patch.dict(globals(),{'free_source':forbidden,'free_branch':forbidden}):
                    resumed=campaign_case(runner,Tok(),[1,3,5],set(),o,folder,'toy_signature',{'scope':'random toy'},[],True)
                assert resumed['scouts']==result['scouts'] and resumed['focused_runs']==result['focused_runs']
            assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
            return dict(scouts=len(result['scouts']),focus_runs=len(result['focused_runs']),
                        future_tokens_forced=False,temperature=temperature,actual_HF=False,model_weights_unchanged=True)
    test('toy_end_to_end_coarse_refine_capture_fork_reports_resume',lambda:integration())
    test('toy_sampled_sources_resume_and_rng_forks',lambda:integration(.8))
    test('resume_rejects_modified_fork_artifact',lambda:integration(artifact_tamper=True))

    def interrupted_resume():
        torch.manual_seed(128);model=_ToyLM().eval();ctrl=LiveController(max_tokens=48,trace_every=0);runner=LiveToyRunner(model,ctrl)
        o=opts(coarse_lambdas='0,0.5,1',max_lambdas=3,refine_rounds=0,max_forks=0,max_new_tokens=8,branch_new_tokens=4,max_context=48)
        native_fn=free_source;calls=0
        def interrupt(*a,**kw):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('simulated power interruption')
            return native_fn(*a,**kw)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            try:
                with patch.dict(globals(),{'free_source':interrupt}):
                    campaign_case(runner,Tok(),[1,3],set(),o,folder,'resume-sig',{},[])
            except RuntimeError:pass
            else:raise AssertionError('interruption not exercised')
            assert len(list((folder/'scouts').glob('*.json')))==1
            calls2=0
            def count(*a,**kw):
                nonlocal calls2
                calls2+=1;return native_fn(*a,**kw)
            with patch.dict(globals(),{'free_source':count}):
                result=campaign_case(runner,Tok(),[1,3],set(),o,folder,'resume-sig',{},[],True)
            assert calls2==2 and result['status']=='completed'
        return dict(completed_unit_reused=True,remaining_trajectories=2)
    test('interruption_reuses_only_completed_scout_receipts',interrupted_resume)
    def unsafe_html():
        with tempfile.TemporaryDirectory() as tmp:
            row=fake_scout(list(range(30)));row['source']['text']='</pre><script>alert(1)</script>'
            r=dict(status='completed',scouts={'0/42':row},rounds=[],final_intervals=[],focused_runs={})
            folder=Path(tmp);write_case_reports(folder,r)
            page=(folder/'index.html').read_text()
            assert '<script>alert' not in page and '&lt;script&gt;' in page
        return dict(model_output_not_interpreted_as_html=True)
    test('html_escapes_model_output',unsafe_html)
    def full_suite_loader():
        # Native/adapter are stubs here. The model and campaign computations are real toy tensors.
        class Config:
            model_type='llama';vocab_size=97;num_hidden_layers=3;max_position_embeddings=96
            _commit_hash=None
            def to_dict(self):return dict(model_type=self.model_type,vocab_size=97,num_hidden_layers=3,max_position_embeddings=96)
        loads=[];encodes=[]
        class Tokenizer(Tok):
            def encode(self,text,add_special_tokens=True):encodes.append(text);return [1,3,5]
        def load_model(path,**kwargs):
            loads.append(path)
            with torch.inference_mode(False):
                m=_ToyLM().eval()
            m.config=kwargs['config'];m.generation_config=SimpleNamespace(eos_token_id=None);return m
        hf=SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *a,**k:Config()),
                           AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()),
                           AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model))
        class Adapter:
            def __init__(self,*a):pass
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);suite=folder/'suite.json'
            json_write(suite,dict(models=[dict(id='a',path='toy_A'),dict(id='b',path='toy_B')],
                                 prompts=[dict(id='p',text='one'),dict(id='q',text='two')]))
            args=campaign_parser().parse_args(['--suite',str(suite),'--device','cpu','--coarse-lambdas','0,1',
                 '--refine-rounds','0','--max-lambdas','2','--max-forks','0','--max-new-tokens','6','--control-tokens','4',
                 '--branch-new-tokens','4','--max-context','32','--out',str(folder/'run')])
            with patch.dict(globals(),{'require_hf':lambda:hf,'HFAdapter':Adapter,
                    'LiveRunner':lambda model,ctrl,device:LiveToyRunner(model,ctrl)}):
                run_campaign(args)
                assert loads==['toy_A','toy_B'] and encodes==['one','two','one','two']
                report=_load_json(folder/'run'/'campaign.json')
                assert len(report['cases'])==4 and report['status']=='completed'
                args.resume=True;run_campaign(args)
                assert len(encodes)==4,'Resume must reuse prompt IDs, not reapply a time-varying template.'
                args.temperature=.1
                rejected(lambda:run_campaign(args))
            assert not (folder/'run'/'.campaign.lock').exists()
        return dict(cases=4,model_loads_per_session=2,retokenization_on_resume=False,actual_HF=False)
    test('full_multi_model_multi_prompt_campaign_loads_once_and_resume_binds_options',full_suite_loader)
    def lock_refusal():
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);json_write(folder/'campaign.json',{});(folder/'.campaign.lock').write_text('unknown')
            args=campaign_parser().parse_args(['--model','toy','--prompt','x','--out',str(folder),'--resume'])
            rejected(lambda:run_campaign(args))
            assert (folder/'.campaign.lock').exists()
        return dict(active_lock_not_removed=True)
    test('concurrent_or_uncleared_lock_refused',lock_refusal)
    def zero_pure_signal():
        o=opts();a=fake_scout([1,2,3,4]*12,0);b=fake_scout([1,2,3,4]*12,1)
        e=neighbor_evidence(a,b,o)
        assert e['labels']==[]
        return dict(baseline_repetition_not_mislabelled_as_intervention_onset=True)
    test('already_repeating_baseline_is_not_new_onset',zero_pure_signal)
    def reproduction_guard_test():
        torch.manual_seed(128)
        model=_ToyLM().eval();ctrl=LiveController(max_tokens=64,trace_every=0);runner=LiveToyRunner(model,ctrl)
        o=opts(max_new_tokens=12,branch_new_tokens=4,max_context=64)
        rc=RepeatConfig(o['min_period'],o['max_period'],o['repeat_copies'],o['min_repeat_tokens'])
        source=free_source(runner,LivePolicy('scout','replay',.5),[1,3],max_new=12,prefill_mode='clean',seed=42,max_forks=0,event_copies=())
        scout=_scout_public(source,Tok(),.5,42,rc);scout['logits_sha256']='wrong digest'
        plan=dict(points=[dict(generated_count=5,probe_period=0,reason='test anchor')])
        with tempfile.TemporaryDirectory() as tmp:
            def forbidden(*a,**kw):raise AssertionError('A fork was executed despite mismatching scout.')
            with patch.dict(globals(),{'free_branch':forbidden}):
                rejected(lambda:_run_focus(runner,Tok(),[1,3],set(),o,scout,plan,Path(tmp),{}))
            saved=_load_json(Path(tmp)/'results.json')
            assert saved['status']=='failed' and saved['scout_reproduction']['status']=='failed'
        return dict(mismatched_scout_blocks_all_branches=True)
    test('mismatched_free_regeneration_never_silently_forks',reproduction_guard_test)
    def saved_logits_receipt_test():
        torch.manual_seed(128)
        model=_ToyLM().eval();ctrl=LiveController(max_tokens=32,trace_every=0);runner=LiveToyRunner(model,ctrl)
        o=opts(coarse_lambdas='0,1',max_lambdas=2,refine_rounds=0,focus_intervals=0,max_forks=0,
               max_new_tokens=5,branch_new_tokens=2,max_context=32,save_logits=True)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            campaign_case(runner,Tok(),[1,3],set(),o,folder,'sig',{},[])
            assert len(list((folder/'scouts').glob('*.safetensors')))==2
            campaign_case(runner,Tok(),[1,3],set(),o,folder,'sig',{},[],True)
        return dict(explicit_logit_saving_and_integrity_resume=True)
    test('optional_scout_logits_are_saved_and_checked_on_resume',saved_logits_receipt_test)
    passed=sum(t['status']=='passed' for t in tests)
    json_write(out,dict(schema='qcache.campaign_tests.1.3',environment=environment(),passed=passed,failed=len(tests)-passed,
                        scope='Planner, receipts, failures and real random-weight toy forwards; HF loaders/adapter stubbed only in explicitly named suite test.',tests=tests))
    print(f'{passed}/{len(tests)} campaign tests passed: {out}',flush=True)
    return passed==len(tests)


# ============================================================================
# V1.4: capability-based adapters, explicit resource controls, independent Q
# gain/admission knobs, and freely generated answer evaluation.
# ============================================================================
import inspect
import struct
import unicodedata
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

VERSION = '1.4.0'
BACKEND = 'qcache_lab_v140'


@dataclass(frozen=True)
class ResearchSettings:
    """Recorded experimental choices, not an adaptive quality optimizer.

    output = (1 - current_attenuation * lambda) * ordinary + lambda * history.
    A None prompt_share preserves the original uniform pool over admitted Q.
    max_queries=0 means no separate Q membership cap. KV is never evicted here.
    """
    current_attenuation: float = 1.0
    prompt_share: float | None = None
    admit_every: int = 1
    max_queries: int = 0
    logits_storage: str = 'memory'

    def __post_init__(self):
        require(math.isfinite(self.current_attenuation) and 0 <= self.current_attenuation <= 1,
                'current-attenuation must lie in [0,1].')
        require(self.prompt_share is None or (math.isfinite(self.prompt_share) and 0 <= self.prompt_share <= 1),
                'prompt-share must lie in [0,1], or be omitted for a uniform Q pool.')
        require(type(self.admit_every) is int and self.admit_every >= 1, 'admit-every must be a positive integer.')
        require(type(self.max_queries) is int and self.max_queries >= 0, 'max-queries must be zero or positive.')
        require(self.logits_storage in {'memory','disk'}, 'logits-storage must be memory or disk.')


RESEARCH_DEFAULTS = ResearchSettings()


class LogitTape(Sequence):
    """Process-local exact CPU float32 logit tape. Not a model checkpoint.

    Disk storage bounds resident logit history; explicit slices/stack exports can
    still be large. The tempfile is deleted when the owning result is released.
    """
    def __init__(self):
        self.file = tempfile.TemporaryFile(prefix='qcache_logits_', suffix='.bin')
        self.width = None
        self.count = 0

    def append(self, row):
        import numpy as np
        a=row.detach().to(device='cpu',dtype=torch.float32).contiguous().numpy()
        require(a.ndim == 1, 'Logit tape expects one vocabulary vector.')
        if self.width is None: self.width = int(a.shape[0])
        require(a.shape[0] == self.width, 'Vocabulary size changed within logit tape.')
        self.file.seek(self.count * self.width * 4)
        self.file.write(a.tobytes()); self.count += 1

    def __len__(self): return self.count

    def __getitem__(self, index):
        import numpy as np
        if isinstance(index,slice): return [self[i] for i in range(*index.indices(self.count))]
        if index < 0: index += self.count
        if not 0 <= index < self.count: raise IndexError(index)
        self.file.seek(index*self.width*4)
        b=self.file.read(self.width*4)
        require(len(b)==self.width*4,'Truncated temporary logit tape.')
        return torch.from_numpy(np.frombuffer(b,dtype=np.float32).copy())

    def __iter__(self):
        for i in range(self.count): yield self[i]

    def close(self): self.file.close()

    def __del__(self):
        try: self.file.close()
        except Exception: pass


def make_logit_tape(controller=None):
    settings=getattr(controller,'research',RESEARCH_DEFAULTS)
    return LogitTape() if settings.logits_storage == 'disk' else []


def context_argument(text):
    if str(text).lower() in {'auto','model'}: return str(text).lower()
    try: value=int(text)
    except (TypeError,ValueError): raise argparse.ArgumentTypeError('context must be auto, model, or an integer >=2')
    if value < 2: raise argparse.ArgumentTypeError('context must be >=2')
    return value


def native_context_limit(config):
    """Read a declared positional bound, never guess from a model name."""
    for field in ('max_position_embeddings','n_positions','max_sequence_length','seq_length'):
        value=getattr(config,field,None)
        if type(value) is int and 1 < value < 2**40:
            return value,field
    return None,None


def resolve_context(request,config,prompt_tokens,source_tokens,branch_tokens=0):
    require(all(type(n) is int and n >= 0 for n in (prompt_tokens,source_tokens,branch_tokens)), 'Invalid lengths.')
    require(prompt_tokens>0 and source_tokens>0,'Prompt and source budgets must be nonempty.')
    needed=prompt_tokens+source_tokens+branch_tokens
    native,field=native_context_limit(config)
    request=context_argument(request)
    if request=='auto': effective=needed
    elif request=='model':
        require(native is not None,'No declared model context limit; pass an explicit --context.')
        effective=native
    else: effective=request
    require(needed <= effective,
            f'prompt({prompt_tokens}) + source({source_tokens}) + reserved fork({branch_tokens}) = {needed} > context {effective}. '
            'Increase --context or reduce the explicit generation budgets. Nothing was truncated.')
    require(native is None or effective <= native,
            f'Requested context {effective} exceeds the model-declared {field}={native}. '
            'This build does not silently change positional encoding or extrapolate context.')
    return dict(request=request,effective_context=effective,model_declared_context=native,
                model_context_field=field,prompt_tokens=prompt_tokens,source_max_new_tokens=source_tokens,
                reserved_branch_new_tokens=branch_tokens,conservative_required_positions=needed,
                allocation='grow on demand; context is a bound, not a preallocation',
                truncation=False,position_extrapolation=False)


# ---------------------------------------------------------------------------
# Capability contract. No model_type whitelist; actual original eager kernels,
# masks, shapes, scalar score transform, and complete cache are checked instead.
# ---------------------------------------------------------------------------

def require_hf():
    try: hf=importlib.import_module('transformers')
    except ImportError as exc:
        raise ImportError('Install transformers (reference environment: 4.57.6), safetensors, and tokenizer dependencies.') from exc
    missing=[n for n in ('AttentionInterface','AttentionMaskInterface','AutoModelForCausalLM') if not hasattr(hf,n)]
    require(not missing,'Installed Transformers lacks required APIs: '+', '.join(missing)+'. Use a compatible release; 4.57.6 is the reference.')
    return hf


_legacy_validate_config = validate_config

def validate_config(config):
    """Reject missing mathematical/state capabilities, not brand names.

    Local/sliding attention is admitted only with FULL retained KV. Its ordinary
    mask remains native; replay explicitly sees all retained observed KV. Native
    recurrent/mixed state and changing-frequency RoPE need a different snapshot
    adapter and cannot be made correct by deleting a whitelist.
    """
    require(not getattr(config,'is_encoder_decoder',False),'Encoder-decoder/cross-attention requires a separate state adapter.')
    require(not getattr(config,'add_cross_attention',False),'Cross-attention is not the decoder self-attention experiment.')
    require(not (getattr(config,'vision_config',None) or getattr(config,'audio_config',None)),
            'Multimodal wrapper requires an explicit text-backbone adapter; load its supported text causal model.')
    n=getattr(config,'num_hidden_layers',None)
    require(type(n) is int and n>0,'Missing num_hidden_layers: a model-state adapter is required.')
    require(type(getattr(config,'vocab_size',None)) is int,'Missing text vocabulary size.')
    layer_types=getattr(config,'layer_types',None) or []
    require(not any(any(x in str(t).lower() for x in ('linear','mamba','recurrent','conv')) for t in layer_types),
            'Hybrid recurrent/linear layers have additional state. This full-QKV adapter cannot snapshot them.')
    rope=getattr(config,'rope_scaling',None) or {}
    if isinstance(rope,dict):
        typ=rope.get('rope_type',rope.get('type','default'))
        require(typ not in {'dynamic','longrope'},
                'Sequence-dependent RoPE state needs an extended snapshot/reprojection protocol; not silently frozen.')
    require(not getattr(config,'quantization_config',None),
            'Quantized weight loaders/device placement require a separate load adapter; this entry uses ordinary HF tensors.')


def discover_attention_modules(model):
    """Return each decoder attention module and its own eager implementation."""
    validate_config(model.config)
    found=[]
    for path,module in model.named_modules():
        if not hasattr(module,'layer_idx') or not isinstance(getattr(module,'layer_idx'),int): continue
        # The eager kernel is architecture-specific. Do not replace Q projections,
        # normalization, rotary embeddings, or fused-QKV splitting ourselves.
        src=importlib.import_module(type(module).__module__)
        eager=getattr(src,'eager_attention_forward',None)
        signature=inspect.signature(module.forward)
        if not callable(eager) or 'hidden_states' not in signature.parameters: continue
        if not (getattr(module,'is_causal',False) or 'attention' in type(module).__name__.lower()): continue
        found.append((int(module.layer_idx),path,module,eager))
    found.sort(key=lambda x:x[0])
    expected=list(range(model.config.num_hidden_layers))
    require([x[0] for x in found]==expected,
            f'Need one discoverable eager self-attention module per decoder layer; found {[x[0] for x in found]}, expected {expected}. '
            'No allowlist bypass is used. A dedicated adapter is needed for this module graph.')
    require(callable(getattr(model,'set_attn_implementation',None)), 'Model does not expose set_attn_implementation.')
    return found


def attention_projectors(model,layers=None):
    result={}
    for i,path,module,_ in discover_attention_modules(model):
        if layers is not None and i not in layers: continue
        for name in ('o_proj','out_proj','dense','c_proj'):
            p=getattr(module,name,None)
            if isinstance(p,nn.Linear): result[i]=p;break
    return result


def capability_report(model):
    records=[]
    for i,path,module,_ in getattr(model,'_qcache_v14_modules',[]):
        records.append(dict(layer=i,path=path,attention_class=type(module).__name__,
                            runtime=getattr(module,'_qcache_contract',{'status':'not_exercised'}),
                            calls=getattr(module,'_qcache_contract_calls',0)))
    return dict(model_type=getattr(model.config,'model_type',None),model_class=type(model).__name__,
                transformers=package_version('transformers'),adapter='HF eager capability contract',
                layer_count=model.config.num_hidden_layers,layers=records,
                native_mask='forwarded unchanged, including native sliding mask',
                query_space='exact tensors presented to the native attention kernel, after model-specific Q/K transforms',
                replay_visibility='all observed KV retained in full DynamicCache; a new global read path on local-attention models',
                not_universal=True)


def _score_scale(module,q,kwargs):
    value=kwargs.get('scaling',getattr(module,'scaling',None))
    if value is None and hasattr(module,'norm_factor'):value=1/float(module.norm_factor)
    if value is None and hasattr(module,'scale_attn_weights'):
        value=q.shape[-1]**-.5 if module.scale_attn_weights else 1.
        if getattr(module,'scale_attn_by_inverse_layer_idx',False):value/=module.layer_idx+1
    require(value is not None,'Native attention scale was not exposed; an explicit score adapter is required.')
    value=float(value)
    require(math.isfinite(value) and value>0,'Invalid native attention scale.')
    return value


def _native_output_layout(module,query,key,value,mask,kwargs,original):
    # Probe a single query to disambiguate H from T even when the real prefill
    # happens to have exactly H tokens. No model state is changed by eager.
    one_mask=None if mask is None else mask[..., :1, :]
    single,_=original(module,query[..., :1, :],key,value,one_mask,**kwargs)
    h=query.shape[1];d=value.shape[-1];b=query.shape[0]
    if single.shape==(b,1,h,d): return 'BTHD'
    if single.shape==(b,h,1,d): return 'BHTD'
    raise ValueError(f'Unknown native attention output shape {tuple(single.shape)}; need BTHD or BHTD.')


def _validate_native_attention(module,q,k,v,mask,base,scale,softcap,layout):
    require(q.ndim==k.ndim==v.ndim==4 and q.shape[0]==1,'Need unpadded rank-4 Q/K/V with one batch.')
    require(q.shape[-1]==k.shape[-1] and k.shape[-2]==v.shape[-2] and k.shape[1]==v.shape[1],
            'Native Q/K/V layout does not match grouped dot-product attention.')
    require(q.shape[1]%k.shape[1]==0,'Query heads must be an integer multiple of KV heads.')
    # Small real-input probe on every call catches later score/bias changes too.
    # Only one query row is recomputed, so long prefill does not double the entire score matrix.
    qp=q[..., -1:, :]
    scores=gqa_scores(qp,k,scale)
    if softcap is not None: scores=torch.tanh(scores/softcap)*softcap
    if mask is not None:
        require(isinstance(mask,Tensor) and mask.ndim==4 and mask.dtype != torch.bool,
                'Need a native additive 4D mask at the eager boundary.')
        scores=scores+mask[..., -1:, :k.shape[-2]]
    probs=torch.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
    expected=gqa_values(probs,v).transpose(1,2)
    actual=base[:,-1:,:,:] if layout=='BTHD' else base[:,:, -1:, :].transpose(1,2)
    require(actual.shape==expected.shape,'Unexpected native output head/value dimensions.')
    atol,rtol=(2e-3,3e-3) if q.dtype in {torch.float16,torch.bfloat16} else (2e-5,2e-5)
    delta=float((actual.float()-expected.float()).abs().max())
    require(torch.allclose(actual,expected,atol=atol,rtol=rtol),
            f'Native score contract mismatch at layer {module.layer_idx}: max_abs={delta:g}. '
            'Potential position bias, sink, transformed score, head mapping, or nonstandard attention. No replay was applied.')
    return delta


def hf_attention(module,query,key,value,attention_mask,**kwargs):
    require(not module.training and not torch.is_grad_enabled(),'QCache is inference-only.')
    original=getattr(module,'_qcache_lab_original_eager',None)
    require(original is not None,'Attention adapter not attached.')
    require(query.shape[-2]<=1 or attention_mask is not None,'Missing native prefill causal mask.')
    require(float(kwargs.get('dropout',0))==0,'Inference attention dropout must be zero.')
    for keyname in ('alibi','position_bias','head_mask','sinks'):
        require(kwargs.get(keyname) is None,f'{keyname} requires a dedicated score adapter.')
    scale=_score_scale(module,query,kwargs)
    softcap=kwargs.get('softcap',None)
    if softcap is not None:
        softcap=float(softcap)
        require(math.isfinite(softcap) and softcap>0,'Invalid score softcap.')
    layout=getattr(module,'_qcache_layout',None)
    if layout is None:
        layout=_native_output_layout(module,query,key,value,attention_mask,kwargs,original)
        module._qcache_layout=layout
    base,weights=original(module,query,key,value,attention_mask,**kwargs)
    delta=_validate_native_attention(module,query,key,value,attention_mask,base,scale,softcap,layout)
    module._qcache_contract_calls=getattr(module,'_qcache_contract_calls',0)+1
    old=getattr(module,'_qcache_contract',{})
    module._qcache_contract=dict(status='runtime_checked',layout=layout,query_heads=query.shape[1],
        kv_heads=key.shape[1],qk_dim=query.shape[-1],value_dim=value.shape[-1],scale=scale,
        softcap=softcap,native_sliding_window=kwargs.get('sliding_window',getattr(module,'sliding_window',None)),
        max_probe_abs=max(delta,old.get('max_probe_abs',0.0)))
    controller=module._qcache_lab_controller
    if hasattr(controller,'kernel_specs'):
        spec=dict(softcap=softcap)
        previous=controller.kernel_specs.get(int(module.layer_idx))
        require(previous is None or previous==spec,'Native score rule changed within the model run.')
        controller.kernel_specs[int(module.layer_idx)]=spec
    else:
        require(softcap is None,'Legacy controller does not support softcapped replay.')
    normal=base if layout=='BTHD' else base.transpose(1,2)
    out=controller.apply(int(module.layer_idx),query,key,value,normal,scale)
    if out is normal: return base,weights
    return (out if layout=='BTHD' else out.transpose(1,2)),None


class HFAdapter:
    """Reversible registry adapter for discoverable eager self-attention models."""
    def __init__(self,model,controller):
        hf=require_hf();require(not model.training,'Call model.eval() before attaching.')
        self.model,self.controller=model,controller
        self.previous=model.config._attn_implementation or 'eager'
        require(isinstance(self.previous,str),'Nested multimodal attention configurations require a separate adapter.')
        self.modules=[];self.found=discover_attention_modules(model)
        from transformers.masking_utils import eager_mask
        hf.AttentionInterface.register(BACKEND,hf_attention)
        hf.AttentionMaskInterface.register(BACKEND,eager_mask)
        try:
            for i,path,module,eager in self.found:
                require(not hasattr(module,'_qcache_lab_controller'),'Adapter already attached.')
                module._qcache_lab_original_eager=eager;module._qcache_lab_controller=controller
                module._qcache_contract_calls=0;module._qcache_layout=None
                self.modules.append(module)
            model._qcache_v14_modules=self.found
            model.set_attn_implementation(BACKEND)
        except BaseException:
            self.close();raise
    def close(self):
        self.model.set_attn_implementation(self.previous)
        for module in self.modules:
            for name in ('_qcache_lab_original_eager','_qcache_lab_controller','_qcache_layout'):
                if hasattr(module,name):delattr(module,name)
        self.modules=[]
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


_BaseLiveRunner=LiveRunner
class LiveRunner(_BaseLiveRunner):
    @torch.inference_mode()
    def forward(self,token_ids):
        require(bool(token_ids),'Cannot forward an empty prefix.')
        total=self.count+len(token_ids)
        require(total<=self.controller.max_tokens,'Context capacity reached. No silent cache eviction.')
        kw=dict(input_ids=torch.tensor([token_ids],device=self.device,dtype=torch.long),
                past_key_values=self.cache,use_cache=True,
                attention_mask=torch.ones((1,total),device=self.device,dtype=torch.long),return_dict=True)
        sig=inspect.signature(self.model.forward).parameters
        if 'cache_position' in sig:
            kw['cache_position']=torch.arange(self.count,total,device=self.device,dtype=torch.long)
        if 'position_ids' in sig:
            kw['position_ids']=torch.arange(self.count,total,device=self.device,dtype=torch.long).unsqueeze(0)
        if 'logits_to_keep' in sig:kw['logits_to_keep']=1
        elif 'num_logits_to_keep' in sig:kw['num_logits_to_keep']=1
        active=[m for _,_,m,_ in getattr(self.model,'_qcache_v14_modules',[]) if hasattr(m,'_qcache_lab_controller')]
        before=[getattr(m,'_qcache_contract_calls',0) for m in active]
        output=self.model(**kw)
        require(all(getattr(m,'_qcache_contract_calls',0)==n+1 for m,n in zip(active,before)),
                'Some attention layers did not invoke the registered kernel exactly once. Adapter coverage failed.')
        self.cache,self.count=output.past_key_values,total
        require(self.cache is not None,'Model did not return an autoregressive cache.')
        require(all(b.seen==total for b in self.controller.banks.values()),'Q bank and full KV position mismatch.')
        logits=output.logits[0,-1].detach().to(device='cpu',dtype=torch.float32)
        require(bool(torch.isfinite(logits).all()),'Nonfinite model logits.')
        return logits

    def kv_snapshot(self):
        from transformers.cache_utils import DynamicCache
        require(type(self.cache) is DynamicCache and not getattr(self.cache,'offloading',False),
                'State forks need an ordinary non-offloaded DynamicCache.')
        if hasattr(self.cache,'layers'):
            require(all(type(x).__name__=='DynamicLayer' and getattr(x,'is_initialized',True) for x in self.cache.layers),
                    'Only full DynamicLayer is serializable here; a hybrid/evicting cache needs an adapter.')
            return [(_clone_cpu(x.keys),_clone_cpu(x.values)) for x in self.cache.layers]
        require(hasattr(self.cache,'to_legacy_cache'),'Unrecognized DynamicCache serialization API.')
        return [(_clone_cpu(k),_clone_cpu(v)) for k,v in self.cache.to_legacy_cache()]

    def kv_restore(self,state):
        from transformers.cache_utils import DynamicCache
        require(len(state)==self.model.config.num_hidden_layers,'KV layer count mismatch.')
        items=[(k.to(self.device).clone(),v.to(self.device).clone()) for k,v in state]
        params=inspect.signature(DynamicCache).parameters
        if 'ddp_cache_data' in params:self.cache=DynamicCache(ddp_cache_data=items)
        elif hasattr(DynamicCache,'from_legacy_cache'):self.cache=DynamicCache.from_legacy_cache(tuple(items))
        else:
            self.cache=DynamicCache()
            for i,(k,v) in enumerate(items):self.cache.update(k,v,i)


# ---------------------------------------------------------------------------
# Replay/readout research controls. Original settings retain the old arithmetic.
# A softcap applies independently to each Q/K score and therefore also admits
# the exact append-only online normalization recurrence.
# ---------------------------------------------------------------------------
class ResearchBank(LiveBank):
    def __init__(self,*,softcap=None,**kwargs):
        super().__init__(**kwargs);self.softcap=softcap

    def scores(self,q,k,scale):
        s=gqa_scores(q,k,scale)
        return s if self.softcap is None else torch.tanh(s/self.softcap)*self.softcap

    def read(self,q,k,v,scale):
        if self.softcap is None:return dense_read(q,k,v,scale,self.acc_dtype,self.chunk)
        kf,vf=k.to(self.acc_dtype),v.to(self.acc_dtype);rs=[];zs=[]
        for start in range(0,q.shape[-2],self.chunk):
            s=self.scores(q[...,start:start+self.chunk,:].to(self.acc_dtype),kf,scale)
            zs.append(s.logsumexp(-1,keepdim=True));rs.append(gqa_values(s.softmax(-1),vf))
        return torch.cat(rs,-2),torch.cat(zs,-2)

    def bootstrap(self,q,k,v,scale,limit=0):
        if self.softcap is None and (not limit or q.shape[-2]<=limit):
            return super().bootstrap(q,k,v,scale)
        require(q.shape[-2]==k.shape[-2]==v.shape[-2] and self.n==0,'Invalid bootstrap.')
        # Explicit cap retains the earliest Q, not an undocumented eviction rule.
        count=min(q.shape[-2],limit) if limit else q.shape[-2]
        qq=q[...,:count,:]
        self._reserve(count,qq,v.shape[-1]);self.q[...,:count,:].copy_(qq)
        self.scale=float(scale);self.n=count;self.seen=k.shape[-2];self.positions=list(range(count))
        r,z=self.read(qq,k,v,scale);self.r[...,:count,:].copy_(r);self.z[...,:count,:].copy_(z)

    @torch.no_grad()
    def step(self,q,k,v,scale,diagnostics=False,admit=True):
        if self.softcap is None:return super().step(q,k,v,scale,diagnostics,admit)
        require(self.n>0 and q.shape[-2]==1 and k.shape[-2]==v.shape[-2]==self.seen+1,'Q/KV append mismatch.')
        require(self.seen+1<=self.max_tokens and float(scale)==self.scale,'Position budget or score scale changed.')
        require(q.device==self.q.device and q.dtype==self.q.dtype and q.shape[:2]==self.q.shape[:2], 'Q layout/device changed.')
        if admit:self._reserve(self.n+1,q,v.shape[-1])
        oldr,oldz=self.r[...,:self.n,:],self.z[...,:self.n,:]
        s=self.scores(self.q[...,:self.n,:].to(self.acc_dtype),k[...,-1:,:].to(self.acc_dtype),scale)
        m=torch.maximum(oldz,s);a,b=torch.exp(oldz-m),torch.exp(s-m);alpha,beta=a/(a+b),b/(a+b)
        if self.engine=='online':
            newv=v[...,-1:,:].to(self.acc_dtype).repeat_interleave(q.shape[1]//v.shape[1],1)
            newr,newz=alpha*oldr+beta*newv,m+torch.log(a+b)
        else:newr,newz=self.read(self.q[...,:self.n,:],k,v,scale)
        detail={} if not diagnostics else dict(gate_mean=float(beta.mean()),gate_max=float(beta.max()),
                                             readout_drift_rms=float((newr-oldr).square().mean().sqrt()))
        oldr.copy_(newr);oldz.copy_(newz);mean=newr.mean(-2,keepdim=True)
        if admit:
            r,z=self.read(q,k,v,scale)
            self.q[...,self.n:self.n+1,:].copy_(q);self.r[...,self.n:self.n+1,:].copy_(r);self.z[...,self.n:self.n+1,:].copy_(z)
            self.positions.append(self.seen);self.n+=1
        self.seen+=1
        return mean,detail

    def verify(self,k,v,atol=2e-4,rtol=2e-4):
        r,z=self.read(self.q[...,:self.n,:],k,v,self.scale)
        torch.testing.assert_close(self.r[...,:self.n,:],r,atol=atol,rtol=rtol)
        torch.testing.assert_close(self.z[...,:self.n,:],z,atol=atol,rtol=rtol)
        return dict(r_max_abs=float((r-self.r[...,:self.n,:]).abs().max()),logz_max_abs=float((z-self.z[...,:self.n,:]).abs().max()))

    def snapshot(self):return super().snapshot()|{'softcap':self.softcap}
    @classmethod
    def restore(cls,data,device):
        bank=super().restore(data,device);bank.softcap=data.get('softcap');return bank


_BaseLiveController=LiveController
class LiveController(_BaseLiveController):
    def __init__(self,*,research=None,**kwargs):
        self.research=research or RESEARCH_DEFAULTS
        self.prompt_length=None;self.kernel_specs={}
        super().__init__(**kwargs)

    def begin_prompt(self,n):self.prompt_length=int(n)

    @torch.no_grad()
    def apply(self,layer,q,k,v,base,scale):
        if self.layers is not None and layer not in self.layers:return base
        require(q.shape[0]==1 and base.shape==(1,q.shape[-2],q.shape[1],v.shape[-1]),'QCache expects BTHD output and one sequence.')
        self.calls+=1;bank=self.banks.get(layer)
        if bank is None:
            bank=ResearchBank(max_tokens=self.max_tokens,replay=True,engine=self.engine,chunk=self.chunk,
                              acc_dtype=self.acc_dtype,softcap=self.kernel_specs.get(layer,{}).get('softcap'))
            bank.bootstrap(q,k,v,scale,limit=self.research.max_queries)
            self.banks[layer]=bank
            if self.prompt_length is None:self.prompt_length=k.shape[-2]
            return base
        policy=self.condition;past=bank.n;position=k.shape[-2]-1
        diag=bool(self.trace_every and position%self.trace_every==0)
        prompt_pos=position<self.prompt_length
        admit=policy.admit and (prompt_pos or (position-self.prompt_length)%self.research.admit_every==0)
        if self.research.max_queries and past>=self.research.max_queries:admit=False
        mean,detail=bank.step(q,k,v,scale,diagnostics=diag,admit=admit)
        prompt_members=sum(p<self.prompt_length for p in bank.positions[:past])
        generated_members=past-prompt_members
        if self.research.prompt_share is not None and prompt_members and generated_members:
            w=self.research.prompt_share
            mean=w*bank.r[...,:prompt_members,:].mean(-2,keepdim=True)+(1-w)*bank.r[...,prompt_members:past,:].mean(-2,keepdim=True)
        self.eligible_positions.setdefault(layer,[]).append(position)
        if self.verify_every and bank.seen%self.verify_every==0:bank.verify(k,v)
        strength=policy.strength
        ordinary_gain=1-self.research.current_attenuation*strength
        hypothetical=None
        if policy.mode in {'baseline','store'} or strength==0:out=base
        elif policy.mode=='attenuation':out=(ordinary_gain*base.to(self.acc_dtype)).to(base.dtype)
        else:
            hypothetical=(ordinary_gain*base.to(self.acc_dtype)+strength*mean.transpose(1,2)).to(base.dtype)
            out=hypothetical if policy.mode=='replay' else parallel_output(base,hypothetical,self.acc_dtype)
        if diag:
            a,h,y=base.to(self.acc_dtype),mean.transpose(1,2),out.to(self.acc_dtype);eps=torch.finfo(self.acc_dtype).eps
            uniform_prompt_weight=prompt_members/past
            effective_prompt_weight=(uniform_prompt_weight if self.research.prompt_share is None else
                                     self.research.prompt_share if generated_members and prompt_members else float(bool(prompt_members)))
            row=dict(branch=self.branch,phase=self.phase,mode=policy.mode,lambda_value=strength,
                layer=layer,position=position,admitted_queries_before=past,admitted_queries_after=bank.n,
                observed_kv=bank.seen,admission_enabled=admit,applied=out is not base,
                base_l2=float(a.norm()),output_l2=float(y.norm()),history_l2=float(h.norm()),
                relative_update=float((y-a).norm()/a.norm().clamp_min(eps)),
                history_current_cosine=float(((a*h).sum(-1)/(a.norm(dim=-1)*h.norm(dim=-1)).clamp_min(eps)).mean()),
                prompt_queries=prompt_members,generated_queries=generated_members,
                prompt_readout_weight=effective_prompt_weight,
                current_coefficient=ordinary_gain,history_coefficient=strength if policy.mode in {'replay','parallel'} else 0,
                coefficient_scope='local hypothetical replay; parallel actual output is norm-scaled ordinary direction',
                q_membership_cap=self.research.max_queries,score_softcap=bank.softcap,
                bank_used_bytes=bank.bytes(),**detail)
            allr=bank.r[...,:past,:].to(self.acc_dtype)
            avg=allr.mean(-2,keepdim=True)
            row['readout_variance_fraction']=float((allr-avg).square().mean()/allr.square().mean().clamp_min(eps))
            row['retained_last_q_position']=bank.positions[past-1]
            row['dispersion_scope']='vector spread among retained readouts, not semantic novelty'
            if policy.mode=='parallel' and hypothetical is not None:
                row['local_head_norm_match_max_abs']=float((y.norm(dim=-1)-hypothetical.to(self.acc_dtype).norm(dim=-1)).abs().max())
                row['norm_target_scope']='this arm, this layer/position/head, before W_O'
            projector=self.projectors.get(layer)
            if projector is not None:
                w,bias=projector.weight,projector.bias
                def projected(z):return torch.nn.functional.linear(z.reshape(z.shape[0],z.shape[1],-1).to(w.dtype),w,bias)
                row['projected_output_l2']=float(projected(out).float().norm())
                if hypothetical is not None:row['projected_local_replay_l2']=float(projected(hypothetical).float().norm())
            if self.trace_sink is None:self.trace.append(row)
            else:self.trace_sink(row)
        return out

    def snapshot(self):
        return super().snapshot()|dict(research=asdict(self.research),prompt_length=self.prompt_length,kernel_specs=copy.deepcopy(self.kernel_specs))

    def restore(self,data,device,branch):
        settings=ResearchSettings(**data.get('research',{}))
        require(settings==self.research,'Research configuration changed on restore. Fork laws change outputs, not saved parameter settings.')
        self.prompt_length=data.get('prompt_length')
        self.kernel_specs={int(i):v for i,v in data.get('kernel_specs',{}).items()}
        require(data['layers']==(None if self.layers is None else sorted(self.layers)),'Layer set changed on restore.')
        require(data['max_tokens']==self.max_tokens and data['engine']==self.engine and data['chunk']==self.chunk
                and data['acc_dtype']==str(self.acc_dtype).split('.')[-1],'Controller configuration changed on restore.')
        self.condition=LivePolicy(**data['condition']);self.phase=data['phase'];self.calls=data['calls'];self.branch=branch
        self.eligible_positions={int(i):list(v) for i,v in data['eligible_positions'].items()}
        self.banks={int(i):ResearchBank.restore(b,device) for i,b in data['banks'].items()};self.trace=[]


_original_free_source=free_source

def free_source(runner,policy,prompt,**kwargs):
    if hasattr(runner.controller,'begin_prompt'):runner.controller.begin_prompt(len(prompt))
    return _original_free_source(runner,policy,prompt,**kwargs)


_old_campaign_options=campaign_options

def campaign_options(args):
    request=args.max_context
    if not isinstance(request,int):args.max_context=2**40
    try:o=_old_campaign_options(args)
    finally:args.max_context=request
    o['max_context']=request
    o['research']=asdict(RESEARCH_DEFAULTS)
    return o


_old_campaign_parser=campaign_parser

def add_research_arguments(p,storage_default='disk'):
    p.add_argument('--current-attenuation',type=float,default=1.,help='a in (1-a*lambda)*o + lambda*h; 1=old blend, 0=additive history.')
    p.add_argument('--prompt-share',type=float,default=None,help='Optional fixed prompt-Q share in the readout pool. Omit for original count-weighted mean.')
    p.add_argument('--admit-every',type=int,default=1,help='Admit every Nth generated Q; all prompt Q unless an explicit cap applies.')
    p.add_argument('--max-queries',type=int,default=0,help='0=no independent cap. Otherwise retain earliest Q and stop further admission, never evict KV.')
    p.add_argument('--logits-storage',choices=['memory','disk'],default=storage_default,help='Internal exact logit history. Disk uses temporary CPU float32 tape; save-logits exports are separate.')


def configure_research(args):
    global RESEARCH_DEFAULTS
    RESEARCH_DEFAULTS=ResearchSettings(**{k:getattr(args,k,v) for k,v in asdict(ResearchSettings()).items()})
    return RESEARCH_DEFAULTS


def campaign_parser():
    p=_old_campaign_parser()
    p.description='QCache V1.4: capability-probed free sweeps and same-state forks. Improvement is evaluated separately by assess.'
    p.add_argument('--max-tokens',dest='max_new_tokens',type=int,default=argparse.SUPPRESS,help='Alias of max-new-tokens; counts generated tokens, not prompt.')
    action=p._option_string_actions['--max-context'];action.type=context_argument;action.default='auto'
    p.add_argument('--context',dest='max_context',type=context_argument,default=argparse.SUPPRESS,help='auto, model, or explicit total position budget.')
    p.set_defaults(out='results_qcache_v14',max_context='auto')
    add_research_arguments(p)
    return p

# ---------------------------------------------------------------------------
# Task evidence: generation stays free. Answer keys are used only by graders
# after generation, never passed to a controller or a token selector.
# ---------------------------------------------------------------------------

def _normal_text(text,casefold=False):
    s=unicodedata.normalize('NFKC',str(text)).strip()
    return s.casefold() if casefold else s


def task_answer(text,spec):
    mode=spec.get('extract','whole')
    if mode=='whole':return text.strip()
    require(mode=='final_line','extract must be whole or final_line.')
    marker=spec.get('marker','FINAL:')
    require(isinstance(marker,str) and bool(marker),'Invalid final-line marker.')
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][len(marker):].strip() if lines and lines[-1].startswith(marker) else None


def grade_answer(text,task):
    """Small auditable built-in graders. No eval/exec/model-generated code."""
    spec=task['grader'];kind=spec['kind'];answer=task_answer(text,spec)
    if answer is None:return dict(score=0.,passed=False,reason='missing_final_marker',extracted=None)
    if kind=='exact':
        candidates=spec.get('answers',[spec.get('answer')])
        require(candidates and all(isinstance(x,str) for x in candidates),'exact grader needs string answer(s).')
        passed=any(_normal_text(answer,spec.get('casefold',False))==_normal_text(x,spec.get('casefold',False)) for x in candidates)
    elif kind=='number':
        try:
            a=Decimal(answer);b=Decimal(str(spec['answer']));tol=Decimal(str(spec.get('atol',0)))
            require(tol.is_finite() and tol>=0,'Invalid numeric grading tolerance.')
            passed=a.is_finite() and b.is_finite() and abs(a-b)<=tol
        except (InvalidOperation,ValueError):passed=False
    elif kind=='json':
        def no_duplicates(pairs):
            obj={}
            for k,v in pairs:
                if k in obj:raise ValueError('duplicate JSON key')
                obj[k]=v
            return obj
        def reject_constant(s):raise ValueError('nonfinite JSON constant')
        try:
            a=json.loads(answer,object_pairs_hook=no_duplicates,parse_constant=reject_constant)
            # Compare canonical JSON instead of Python equality (True != 1 here).
            passed=json.dumps(a,sort_keys=True,ensure_ascii=False,allow_nan=False)==json.dumps(spec['answer'],sort_keys=True,ensure_ascii=False,allow_nan=False)
        except (ValueError,TypeError):passed=False
    else:raise ValueError('Unknown grader kind: '+str(kind))
    return dict(score=float(passed),passed=bool(passed),reason='matched' if passed else 'not_matched',extracted=answer)


def read_tasks(path):
    rows=[]
    for line_number,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        try:t=json.loads(line)
        except ValueError as exc:raise ValueError(f'Task JSONL line {line_number}: {exc}') from exc
        require(isinstance(t,dict) and set(t)>={'id','task','split','prompt','grader'},'Task requires id/task/split/prompt/grader.')
        require(all(isinstance(t[k],str) and t[k].strip() for k in ('id','task','prompt')),'Invalid task identity/prompt.')
        require(t['split'] in {'dev','test'},'Task split must be dev or test.')
        require(isinstance(t['grader'],dict) and t['grader'].get('kind') in {'exact','number','json'},'Unsupported grader.')
        g=t['grader'];kind=g['kind']
        require(g.get('extract','whole') in {'whole','final_line'},'Unknown answer extractor.')
        if kind in {'number','json'}:require('answer' in g,'Grader needs answer.')
        if kind=='number':
            target=Decimal(str(g['answer']));tol=Decimal(str(g.get('atol',0)))
            require(target.is_finite() and tol.is_finite() and tol>=0,'Invalid numeric target/tolerance.')
        if kind=='json':json.dumps(g['answer'],allow_nan=False)
        if kind=='exact':require(type(g.get('casefold',False)) is bool,'casefold must be boolean.')
        grade_answer('',t)  # Validate schema without running a model.
        rows.append(t)
    require(rows and len({t['id'] for t in rows})==len(rows),'Tasks must be nonempty and IDs unique.')
    dev={_json_hash(t['prompt']) for t in rows if t['split']=='dev'}
    test={_json_hash(t['prompt']) for t in rows if t['split']=='test'}
    require(not dev&test,'Identical prompts appear in dev and test; fix the split before selection.')
    return rows


def read_presets(path):
    data=_load_json(Path(path))
    raw=data.get('presets') if isinstance(data,dict) else data
    require(isinstance(raw,list) and raw,'Presets must be a nonempty JSON array or {presets:[...]}.')
    allowed={'id','lambda','current_attenuation','prompt_share','admit_every','max_queries'}
    result=[dict(id='native',lambda_value=0.,research=asdict(ResearchSettings()))]
    seen={'native'}
    for item in raw:
        require(isinstance(item,dict) and set(item)<=allowed and {'id','lambda'}<=set(item),'Unknown or missing preset fields.')
        name=_safe_id(item['id']);require(name not in seen,'Duplicate/reserved preset ID.');seen.add(name)
        strength=float(item['lambda']);require(math.isfinite(strength) and 0<=strength<=1,'Invalid preset lambda.')
        opts={k:v for k,v in item.items() if k in ResearchSettings.__dataclass_fields__}
        result.append(dict(id=name,lambda_value=strength,research=asdict(ResearchSettings(**opts))))
    return result


def summarize_quality(rows):
    groups={}
    for row in rows:groups.setdefault(row['task'],[]).append(row)
    per_task={name:dict(n=len(rs),score=sum(r['grade']['score'] for r in rs)/len(rs),
                        repeat_rate=sum(r['repeat_observed'] for r in rs)/len(rs),
                        horizon_rate=sum(r['stop_reason']=='max_new_tokens' for r in rs)/len(rs))
              for name,rs in sorted(groups.items())}
    macro=sum(t['score'] for t in per_task.values())/len(per_task) if per_task else None
    return dict(n=len(rows),task_count=len(per_task),macro_score=macro,per_task=per_task,
                repeat_rate=sum(r['repeat_observed'] for r in rows)/len(rows) if rows else None,
                observation_seconds=sum(r['elapsed_seconds_with_observation'] for r in rows),
                scope='finite supplied tasks; not a general-intelligence estimate')


def select_development_preset(summaries,presets,max_repeat_excess=.05,max_task_regression=0.):
    native=summaries['native'];selection='native';best=native['macro_score'];ledger=[]
    for preset in presets:
        name=preset['id']
        if name=='native':continue
        s=summaries[name]
        require(set(s['per_task'])==set(native['per_task']),'Unequal task coverage cannot be compared.')
        deltas={task:s['per_task'][task]['score']-native['per_task'][task]['score'] for task in s['per_task']}
        excess=s['repeat_rate']-native['repeat_rate']
        feasible=excess<=max_repeat_excess and min(deltas.values())>=-max_task_regression
        gain=s['macro_score']-native['macro_score']
        ledger.append(dict(id=name,macro_delta=gain,per_task_deltas=deltas,repeat_rate_excess=excess,feasible=feasible))
        # Strict improvement only; ties retain native or the earlier preregistered preset.
        if feasible and s['macro_score']>best:selection=name;best=s['macro_score']
    chosen=next(p for p in presets if p['id']==selection)
    return dict(selected_preset=chosen,positive_development_gain=selection!='native',ledger=ledger,
                rule=dict(objective='macro exact/number/JSON task success',max_repeat_excess=max_repeat_excess,
                          max_per_task_regression=max_task_regression,tie='native, then declared preset order'),
                warning='Development selection is exploratory; evaluate this frozen preset once on untouched test tasks.')


def paired_quality_difference(native_rows,candidate_rows,repeats=1000):
    require([r['id'] for r in native_rows]==[r['id'] for r in candidate_rows],'Task-paired comparison requires identical ordered IDs.')
    groups={};wins=losses=ties=0
    for a,b in zip(native_rows,candidate_rows):
        d=b['grade']['score']-a['grade']['score'];groups.setdefault(a['task'],[]).append(d)
        wins+=d>0;losses+=d<0;ties+=d==0
    require(bool(groups),'Empty paired evaluation.')
    delta=sum(sum(ds)/len(ds) for ds in groups.values())/len(groups)
    rng=random.Random(17041);draws=[]
    for _ in range(repeats):
        draws.append(sum(sum(rng.choice(ds) for _ in ds)/len(ds) for ds in groups.values())/len(groups))
    draws.sort()
    return dict(macro_delta=delta,wins=wins,losses=losses,ties=ties,
                paired_bootstrap_95_interval=[draws[int(.025*len(draws))],draws[min(len(draws)-1,int(.975*len(draws)))]] if draws else None,
                bootstrap_unit='items resampled within each observed task family; families are fixed',
                limitation='Exploratory interval on these tasks; small n, repeated candidates and shared templates limit inference.')


def assessment_options(args):
    return dict(max_new_tokens=args.max_new_tokens,context=args.max_context,prefill=args.prefill,chat=args.chat,
                chat_date=args.chat_date,temperature=args.temperature,top_p=args.top_p,seed=args.seed,
                layers=args.layers,device=args.device,dtype=args.dtype,logits_storage=args.logits_storage,
                min_period=args.min_period,max_period=args.max_period,repeat_copies=args.repeat_copies,min_repeat_tokens=args.min_repeat_tokens)


def validate_test_lock(lock,tasks,binding):
    require(lock.get('schema')=='qcache.preset_lock.1.4','Unsupported preset lock.')
    require(lock['binding']==binding,'Model/code/generation settings differ from development lock. Use the same evaluation protocol.')
    require(not set(lock['development_prompt_hashes'])&{_json_hash(t['prompt']) for t in tasks},
            'Test prompt overlaps development data from the preset lock.')
    require(all(t['split']=='test' for t in tasks),'Test lock can only be used with test tasks.')
    return lock['selection']['selected_preset']


def write_assessment_reports(out,report):
    json_write(out/'assessment_packet.json',report)
    lines=['# QCache V1.4 task evidence','',f'Status: {report.get("status")} / split: {report.get("split")}',
           '', 'Freely generated answers. Keys are used only after generation. No best-of-fork answers.',
           'Scores describe only the supplied task set. Repetition, censoring and resource observations are separate.', '',
           '| Preset | Macro success | Repeat rate | Items | Observed seconds |', '|---|---:|---:|---:|---:|']
    for name,s in report.get('summaries',{}).items():
        lines.append(f'| {name} | {s["macro_score"]:.4f} | {s["repeat_rate"]:.4f} | {s["n"]} | {s["observation_seconds"]:.3f} |')
    for row in report.get('rows',[]):
        lines += ['',f'## {row["preset"]} / {row["id"]}',f'Task: {row["task"]}; grade: {row["grade"]["score"]}; stop: {row["stop_reason"]}',
                  '```text',row['text'].replace('```',"'''"),'```']
    (out/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def assess_loaded(model,tokenizer,runner_factory,adapter_factory,tasks,presets,args,out,base_binding):
    """Runner-injected loop; integration tested with a real random toy decoder."""
    selected=[t for t in tasks if t['split']==args.split]
    require(selected,'No tasks for the selected split.')
    binding=dict(base_binding,protocol=assessment_options(args))
    selection_from_lock=None
    if args.split=='test':
        require(args.lock and not args.presets,'Test needs --lock and forbids --presets (no test-time parameter sweep).')
        selected_preset=validate_test_lock(_load_json(Path(args.lock)),selected,binding)
        presets=[dict(id='native',lambda_value=0.,research=asdict(ResearchSettings()))]
        if selected_preset['id']!='native':presets.append(selected_preset)
        selection_from_lock=_file_hash(Path(args.lock))
    rc=RepeatConfig(args.min_period,args.max_period,args.repeat_copies,args.min_repeat_tokens)
    report=dict(schema='qcache.assessment.1.4',status='running',split=args.split,binding=binding,
                presets=presets,rows=[],summaries={},selection_lock_sha256=selection_from_lock,
                future_tokens_forced=False,grader_keys_sent_to_model=False,selection_on_test=False,
                caution='Passing these tasks is not a demonstration of broad intelligence. No answer is selected from forks.',
                effective_research={})
    ids_by_task={}
    for t in selected:
        text=t['prompt']
        if args.chat:
            require(bool(getattr(tokenizer,'chat_template',None)),'Tokenizer lacks a chat template.')
            extra={} if args.chat_date is None else {'date_string':args.chat_date}
            ids=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,**extra)
        else:ids=tokenizer.encode(text,add_special_tokens=True)
        require(isinstance(ids,list) and ids and all(type(i)is int and 0<=i<model.config.vocab_size for i in ids),'Bad task prompt IDs.')
        ids_by_task[t['id']]=ids
    report['inputs']=[dict(id=t['id'],task=t['task'],prompt=t['prompt'],prompt_token_ids=ids_by_task[t['id']]) for t in selected]
    eosraw=getattr(getattr(model,'generation_config',None),'eos_token_id',getattr(tokenizer,'eos_token_id',None))
    eos=set(eosraw if isinstance(eosraw,list) else [] if eosraw is None else [eosraw])
    native_first={};native_control={}
    global RESEARCH_DEFAULTS
    previous_defaults=RESEARCH_DEFAULTS
    try:
        for preset in presets:
            # Disk/memory is observational storage, not an algorithmic preset field.
            RESEARCH_DEFAULTS=ResearchSettings(**(preset['research']|{'logits_storage':args.logits_storage}))
            report['effective_research'][preset['id']]=asdict(RESEARCH_DEFAULTS)
            ctrl=LiveController(layers=parse_layers(args.layers,model.config.num_hidden_layers),max_tokens=512,
                                trace_every=0,research=RESEARCH_DEFAULTS)
            runner=runner_factory(model,ctrl)
            adapter=None
            try:
                if preset['id']!='native':adapter=adapter_factory(model,ctrl)
                for index,t in enumerate(selected):
                    prompt=ids_by_task[t['id']]
                    lengths=resolve_context(args.max_context,model.config,len(prompt),args.max_new_tokens)
                    ctrl.max_tokens=lengths['effective_context']
                    kw=dict(max_new=args.max_new_tokens,prefill_mode=args.prefill,seed=args.seed,
                            temperature=args.temperature,top_p=args.top_p,eos_ids=eos,repeat_config=rc,
                            max_forks=0,event_copies=())
                    print(f'[assess/{args.split}] {preset["id"]}: {index+1}/{len(selected)} {t["id"]}',flush=True)
                    if preset['id']=='native':policy=LivePolicy('native','baseline',0.)
                    else:policy=LivePolicy(preset['id'],'replay',preset['lambda_value'])
                    if preset['id']!='native':
                        reference=native_control[t['id']]
                        control=free_source(runner,LivePolicy('zero','replay',0.),prompt,
                                            **(kw|{'max_new':len(reference['tokens'])}))
                        require(control['token_ids']==reference['tokens'] and logits_digest(control['_logits'])==reference['logits_sha256'],
                                'Task-specific zero-control differs from native. No quality score was accepted.')
                        report.setdefault('zero_controls',[]).append(dict(preset=preset['id'],id=t['id'],status='passed',tokens=len(reference['tokens'])))
                        del control
                    result=free_source(runner,policy,prompt,**kw)
                    if preset['id']=='native':
                        native_first[t['id']]=result['_logits'][0].clone()
                        n=min(24,len(result['token_ids']))
                        native_control[t['id']]=dict(tokens=result['token_ids'][:n],logits_sha256=logits_digest(result['_logits'][:n]))
                    elif args.prefill=='clean':
                        require(torch.equal(native_first[t['id']],result['_logits'][0]),'Clean first-token native control changed.')
                    text=tokenizer.decode(result['token_ids'],skip_special_tokens=True)
                    row=dict(id=t['id'],task=t['task'],preset=preset['id'],split=args.split,text=text,
                             token_ids=result['token_ids'],grade=grade_answer(text,t),stop_reason=result['stop_reason'],
                             output_horizon_censored=result['stop_reason']=='max_new_tokens',
                             repeat_observed=any(e.get('kind') in {'enter','reenter'} for e in result['events']),
                             repetition=result['repetition'],events=result['events'],qcache=result['qcache'],length_resolution=lengths,
                             elapsed_seconds_with_observation=result['elapsed_seconds_with_observation'])
                    # Detector event fields are checked directly, not inferred from text style.
                    row['repeat_observed']=trajectory_features(result,rc)['repeat_observed']
                    report['rows'].append(row)
                    with (out/'answers.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    write_assessment_reports(out,report)
                    del result
                rs=[r for r in report['rows'] if r['preset']==preset['id']]
                report['summaries'][preset['id']]=summarize_quality(rs)
                report.setdefault('capabilities',{})[preset['id']]=capability_report(model)
            finally:
                if adapter is not None:adapter.close()
                runner.cache=None;ctrl.banks.clear()
        native_rows=[r for r in report['rows'] if r['preset']=='native']
        report['paired_differences']={p['id']:paired_quality_difference(native_rows,[r for r in report['rows'] if r['preset']==p['id']])
                                      for p in presets if p['id']!='native'}
        if args.split=='dev':
            selection=select_development_preset(report['summaries'],presets,args.max_repeat_excess,args.max_task_regression)
            report['development_selection']=selection
            lock=dict(schema='qcache.preset_lock.1.4',binding=binding,selection=selection,
                      development_prompt_hashes=[_json_hash(t['prompt']) for t in selected],
                      development_ids=[t['id'] for t in selected],task_file_sha256=_file_hash(Path(args.tasks)),
                      tested_presets_sha256=_json_hash(presets),created_utc=datetime.now(timezone.utc).isoformat())
            json_write(out/'preset_lock.json',lock)
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));(out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        RESEARCH_DEFAULTS=previous_defaults;write_assessment_reports(out,report)
    return report


def assess_parser():
    p=argparse.ArgumentParser(description='Free-answer task evaluation, development selection and locked held-out evaluation. No forced future tokens.')
    p.add_argument('--model',required=True);p.add_argument('--revision',default='main');p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--tasks',required=True,help='JSONL with id/task/split/prompt/grader. Grader answers never enter the model prompt.')
    p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--presets');p.add_argument('--lock')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--dtype',choices=['auto','float32','float16','bfloat16'],default='auto')
    p.add_argument('--max-new-tokens','--max-tokens',type=int,default=512)
    p.add_argument('--max-context','--context',type=context_argument,default='auto')
    p.add_argument('--chat',action='store_true');p.add_argument('--chat-date');p.add_argument('--layers',default='all')
    p.add_argument('--prefill',choices=['clean','stream'],default='clean')
    p.add_argument('--temperature',type=float,default=0.);p.add_argument('--top-p',type=float,default=1.);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-repeat-excess',type=float,default=.05);p.add_argument('--max-task-regression',type=float,default=0.)
    p.add_argument('--logits-storage',choices=['disk','memory'],default='disk');p.add_argument('--threads',type=int,default=0)
    for name,default in [('min-period',1),('max-period',64),('repeat-copies',3),('min-repeat-tokens',24)]:p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--out',default='qcache_assessment_v14')
    return p


def run_assess(args):
    require(args.max_new_tokens>=2 and args.threads>=0,'Invalid token/thread budget.')
    require(math.isfinite(args.temperature) and args.temperature>=0 and 0<args.top_p<=1,'Invalid sampling settings.')
    require(0<=args.max_repeat_excess<=1 and 0<=args.max_task_regression<=1,'Invalid selection constraints.')
    tasks=read_tasks(args.tasks)
    if args.chat and args.chat_date is None:
        if args.split=='test' and args.lock:
            args.chat_date=_load_json(Path(args.lock))['binding']['protocol'].get('chat_date')
        else:args.chat_date=datetime.now(timezone.utc).strftime('%d %b %Y')
    if args.split=='dev':
        require(args.presets and not args.lock,'Development uses --presets, not a test lock.')
        presets=read_presets(args.presets)
    else:
        require(args.lock and not args.presets,'Held-out testing requires --lock and forbids --presets.')
        presets=[dict(id='native')]
        chosen=_load_json(Path(args.lock))['selection']['selected_preset']
        if chosen['id']!='native':presets.append(chosen)
    selected=[t for t in tasks if t['split']==args.split]
    require(selected,'No tasks for this split.')
    if args.dry_run:
        print(json.dumps(dict(model=args.model,split=args.split,task_count=len(selected),presets=[p['id'] for p in presets],
                              generated_token_upper_bound=len(selected)*(len(presets)*args.max_new_tokens+(len(presets)-1)*min(24,args.max_new_tokens)),
                              quality_claim=False,model_loaded=False,files_written=False),indent=2));return
    out=Path(args.out);require(not out.exists() or not any(out.iterdir()),'Choose a new empty assessment output directory.')
    out.mkdir(parents=True,exist_ok=True)
    hf=require_hf();device=choose_device(args.device)
    if args.threads:torch.set_num_threads(args.threads)
    dtype=args.dtype if args.dtype!='auto' else 'float32' if device.type=='cpu' else 'float16'
    load=dict(local_files_only=args.local_files_only,trust_remote_code=False,revision=args.revision)
    config=hf.AutoConfig.from_pretrained(args.model,**load);validate_config(config)
    tokenizer=hf.AutoTokenizer.from_pretrained(args.model,**load)
    model=hf.AutoModelForCausalLM.from_pretrained(args.model,config=config,dtype=getattr(torch,dtype),
             attn_implementation='eager',use_safetensors=True,**load).to(device).eval();model.requires_grad_(False)
    _cpu_metric_check()
    template_probe=None
    if args.chat:
        extra={} if args.chat_date is None else {'date_string':args.chat_date}
        template_probe=tokenizer.apply_chat_template([dict(role='user',content='QCache deterministic template probe.')],
                            tokenize=True,add_generation_prompt=True,**extra)
    binding=dict(chat_template_probe_token_ids=template_probe,
                 model=model_provenance(dict(path=args.model,revision=args.revision,id='model'),config,tokenizer),
                 script_sha256=_file_hash(Path(__file__)),transformers=package_version('transformers'),torch=torch.__version__,
                 effective_device=str(device),effective_dtype=dtype)
    report=assess_loaded(model,tokenizer,lambda m,c:LiveRunner(m,c,device),HFAdapter,tasks,presets,args,out,binding)
    print('Completed:',out/'assessment.md','; share:',out/'assessment_packet.json',flush=True)
    if report.get('development_selection'):print('Development selected:',report['development_selection']['selected_preset']['id'],'; lock:',out/'preset_lock.json',flush=True)


class QCacheSession:
    """Importable post-hoc session, sharing frozen model weights.

    Example:
      with QCacheSession(model, strength=.15, context=4096) as session:
          result = session.generate(prompt_ids, max_new=256)
    The returned generation is free. This is not a monkeypatch to model.generate.
    Each call resets KV/Q; separate calls are not persistent chat sessions.
    """
    def __init__(self,model,*,strength=.15,context=4096,layers=None,settings=None):
        self.model=model;self.strength=float(strength)
        require(0<=self.strength<=1,'Invalid strength.')
        self.controller=LiveController(layers=layers,max_tokens=int(context),research=settings or ResearchSettings())
        self.runner=LiveRunner(model,self.controller,next(model.parameters()).device)
        self.adapter=HFAdapter(model,self.controller)
    def generate(self,prompt_ids,*,max_new=256,seed=42,temperature=0.,top_p=1.,eos_ids=None,prefill_mode='clean'):
        resolve_context(self.controller.max_tokens,self.model.config,len(prompt_ids),max_new)
        if eos_ids is None:
            raw=getattr(getattr(self.model,'generation_config',None),'eos_token_id',None)
            eos_ids=set(raw if isinstance(raw,list) else [] if raw is None else [raw])
        return free_source(self.runner,LivePolicy('module','replay',self.strength),list(prompt_ids),max_new=max_new,
                           seed=seed,temperature=temperature,top_p=top_p,eos_ids=eos_ids,prefill_mode=prefill_mode,max_forks=0,event_copies=())
    def close(self):self.adapter.close();self.runner.cache=None;self.controller.banks.clear()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

_old_run_live=run_live

def run_live(args):
    configure_research(args)
    args.context_request=args.max_context
    if not isinstance(args.max_context,int):args.max_context=2**40
    return _old_run_live(args)


def probe_loaded(model,device,prompt,context=256):
    """Short real-model contract, no-op and state-fork probe, not quality evaluation."""
    ctrl=LiveController(max_tokens=context,trace_every=1,verify_every=3)
    runner=LiveRunner(model,ctrl,device)
    kw=dict(max_new=8,prefill_mode='clean',seed=31,max_forks=0,event_copies=())
    native=free_source(runner,LivePolicy('native','baseline',0.),prompt,**kw)
    with HFAdapter(model,ctrl):
        controls=null_free_controls(runner,prompt,max_new=8,prefill_mode='clean',seed=31,eos_ids=set(),atol=1e-6,native=native)
        states=[]
        def cap(s,r):states.append(s);return dict(memory_index=len(states)-1)
        root=free_source(runner,LivePolicy('probe','replay',.15),prompt,max_new=16,prefill_mode='clean',seed=31,
                         fork_at=(4,),max_forks=1,event_copies=(),checkpoint_sink=cap)
        continued=free_branch(runner,states[0],'continue',max_new=8)
        restored=continuation_control(root,states[0],continued,1e-6)
        frozen=free_branch(runner,states[0],'freeze',max_new=8)
        require(torch.equal(continued['_logits'][0],frozen['_logits'][0]),'Freeze first-prediction check failed.')
        ctrl.engine='dense'
        dense=free_source(runner,LivePolicy('dense','replay',.15),prompt,max_new=8,prefill_mode='stream',seed=31,max_forks=0)
        ctrl.engine='online'
        online=free_source(runner,LivePolicy('online','replay',.15),prompt,max_new=8,prefill_mode='stream',seed=31,max_forks=0)
        maxdiff=max(float((a-b).abs().max()) for a,b in zip(online['_logits'],dense['_logits']))
        require(maxdiff<=2e-4,'Online/dense replay mismatch in capability probe.')
        caps=capability_report(model)
    return dict(status='passed',capabilities=caps,null_controls=controls,restore=restored,
                freeze_first_prediction_identical=True,stream_online_dense_max_abs=maxdiff,
                scope='Runtime attention/cache/no-op/fork checks on this loaded model. Not task quality or universal compatibility.')


def run_probe(args):
    out=Path(args.out);report=dict(schema='qcache.probe.1.4',environment=environment(),model=args.model,status='started')
    try:
        hf=require_hf();device=choose_device(args.device)
        load=dict(local_files_only=args.local_files_only,trust_remote_code=False,revision=args.revision)
        config=hf.AutoConfig.from_pretrained(args.model,**load);validate_config(config)
        dtype=args.dtype if args.dtype!='auto' else ('float32' if device.type=='cpu' else 'float16')
        tokenizer=hf.AutoTokenizer.from_pretrained(args.model,**load)
        ids=tokenizer.encode('A small attention and cache check.',add_special_tokens=True)
        lengths=resolve_context(args.max_context,config,len(ids),16,8)
        model=hf.AutoModelForCausalLM.from_pretrained(args.model,config=config,attn_implementation='eager',dtype=getattr(torch,dtype),
                    use_safetensors=True,**load).to(device).eval();model.requires_grad_(False)
        report.update(probe_loaded(model,device,ids,lengths['effective_context']),length_resolution=lengths)
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
    json_write(out,report);print('Probe:',report['status'],out,flush=True)
    return report['status']=='passed'


def extended_hf_smoke(out,device_name='cpu'):
    report=dict(schema='qcache.hf_smoke.1.4',environment=environment(),models=[],
                scope='Actual HF random tiny classes; no checkpoint download and no language-quality claim.')
    try:hf=require_hf()
    except (ImportError,ValueError) as exc:
        report.update(status='not_run',reason=str(exc));json_write(out,report);print('HF NOT RUN:',exc);return False
    device=choose_device(device_name)
    cases=[('llama','LlamaConfig','LlamaForCausalLM',{}),
           ('mistral','MistralConfig','MistralForCausalLM',{'sliding_window':None}),
           ('qwen2','Qwen2Config','Qwen2ForCausalLM',{'use_sliding_window':False,'sliding_window':None}),
           ('qwen3','Qwen3Config','Qwen3ForCausalLM',{'use_sliding_window':False,'sliding_window':None,'head_dim':8}),
           ('phi3','Phi3Config','Phi3ForCausalLM',{'sliding_window':None,'rope_scaling':None,'original_max_position_embeddings':128}),
           ('olmo2','Olmo2Config','Olmo2ForCausalLM',{}),
           ('mixtral','MixtralConfig','MixtralForCausalLM',{'sliding_window':None,'num_local_experts':2,'num_experts_per_tok':1}),
           ('gemma2','Gemma2Config','Gemma2ForCausalLM',{'head_dim':8,'query_pre_attn_scalar':8,'sliding_window':8,
                                                       'attn_logit_softcapping':12.0})]
    for name,cn,mn,extra in cases:
        if not hasattr(hf,cn) or not hasattr(hf,mn):
            report['models'].append(dict(model_type=name,status='not_available_in_installed_transformers'));continue
        try:
            torch.manual_seed(409)
            opts=dict(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
                      num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0.,bos_token_id=1,eos_token_id=2,pad_token_id=0)
            config=getattr(hf,cn)(**(opts|extra));config._attn_implementation='eager'
            with torch.device('cpu'):model=getattr(hf,mn)(config)
            model.to(device).eval();model.requires_grad_(False)
            item=probe_loaded(model,device,[1,4,7,9,13],64)
            report['models'].append(dict(model_type=name,**item));del model
            print('HF PASS',name,flush=True)
        except Exception as exc:
            report['models'].append(dict(model_type=name,status='failed',error=repr(exc),traceback=traceback.format_exc()))
            print('HF FAIL',name,repr(exc),flush=True)
    exercised=[m for m in report['models'] if m['status']!='not_available_in_installed_transformers']
    report['status']='passed' if exercised and all(m['status']=='passed' for m in exercised) else 'failed'
    json_write(out,report);return report['status']=='passed'


def v14_tests(out,non_cpu_default=False):
    from types import SimpleNamespace
    from unittest.mock import patch
    tests=[];torch.set_num_threads(min(2,torch.get_num_threads()))
    def test(name,fn):
        t=time.perf_counter()
        try:
            with torch.inference_mode():d=fn()
            tests.append(dict(name=name,status='passed',seconds=time.perf_counter()-t,detail=d));print('PASS v14/'+name,flush=True)
        except Exception as exc:
            tests.append(dict(name=name,status='failed',error=repr(exc),traceback=traceback.format_exc()));print('FAIL v14/'+name,repr(exc),flush=True)
    def rejected(fn):
        try:fn()
        except (ValueError,TypeError,AssertionError,argparse.ArgumentTypeError):return True
        raise AssertionError('Expected rejection')
    def lengths():
        c=SimpleNamespace(max_position_embeddings=131072)
        a=resolve_context('auto',c,1000,4096,1024);assert a['effective_context']==6120
        assert resolve_context('model',c,1000,4096,1024)['effective_context']==131072
        rejected(lambda:resolve_context(512,c,500,96,96))
        rejected(lambda:resolve_context(262144,c,10,20))
        assert resolve_context(10000,SimpleNamespace(),10,20)['model_declared_context'] is None
        return a
    def tape():
        g=torch.Generator(device='cpu').manual_seed(101);rows=[torch.randn(31,generator=g,device='cpu') for _ in range(9)]
        t=LogitTape()
        for r in rows:t.append(r)
        assert all(torch.equal(a,b) for a,b in zip(rows,t))
        assert torch.equal(t[-1],rows[-1]) and logits_digest(rows)==logits_digest(t)
        assert all(torch.equal(a,b) for a,b in zip(rows[2:5],t[2:5]));t.close()
        return dict(exact_float32_roundtrip=True,rows=9)
    def config_gate():
        base=dict(model_type='never_seen_brand',num_hidden_layers=2,vocab_size=97,max_position_embeddings=256)
        validate_config(SimpleNamespace(**base))
        validate_config(SimpleNamespace(**base,sliding_window=32,layer_types=['sliding_attention','full_attention']))
        rejected(lambda:validate_config(SimpleNamespace(**base,layer_types=['linear_attention'])))
        rejected(lambda:validate_config(SimpleNamespace(**base,rope_scaling={'rope_type':'dynamic'})))
        return dict(unseen_model_name_accepted=True,hybrid_state_rejected=True)
    def tensors(n=16,dtype=torch.float64):
        g=torch.Generator(device='cpu').manual_seed(166)
        return [torch.randn(1,h,n,d,generator=g,device='cpu',dtype=dtype) for h,d in [(6,8),(2,8),(2,5)]]
    def recurrence(softcap):
        q,k,v=tensors();b=ResearchBank(max_tokens=16,acc_dtype=torch.float64,softcap=softcap)
        b.bootstrap(q[...,:4,:],k[...,:4,:],v[...,:4,:],.4)
        maximum=0
        for i in range(4,16):
            mean,_=b.step(q[...,i:i+1,:],k[...,:i+1,:],v[...,:i+1,:],.4,admit=(i%2==0))
            e=b.verify(k[...,:i+1,:],v[...,:i+1,:],atol=1e-12,rtol=1e-12);maximum=max(maximum,e['r_max_abs'])
        restored=ResearchBank.restore(b.snapshot(),'cpu')
        assert restored.softcap==softcap and state_digest(b.snapshot())==state_digest(restored.snapshot())
        return dict(max_abs=maximum,softcap=softcap)
    def research_algebra():
        q,k,v=tensors(dtype=torch.float32);settings=ResearchSettings(current_attenuation=0,prompt_share=.7,admit_every=2,max_queries=7)
        c=LiveController(max_tokens=32,research=settings,trace_every=1);c.begin_prompt(4);c.reset(LivePolicy(strength=.3))
        g=torch.Generator(device='cpu').manual_seed(62)
        base=torch.randn(1,4,6,5,generator=g,device='cpu')
        assert c.apply(0,q[...,:4,:],k[...,:4,:],v[...,:4,:],base,.4) is base
        for i in range(4,12):
            base=torch.randn(1,1,6,5,generator=g,device='cpu');b=c.banks[0];past=b.n
            out=c.apply(0,q[...,i:i+1,:],k[...,:i+1,:],v[...,:i+1,:],base,.4)
            h=b.r[...,:past,:]
            h=h.mean(-2,keepdim=True) if past==4 else .7*h[...,:4,:].mean(-2,keepdim=True)+.3*h[...,4:,:].mean(-2,keepdim=True)
            torch.testing.assert_close(out,base+.3*h.transpose(1,2),atol=1e-7,rtol=1e-7)
        assert c.banks[0].n==7 and c.banks[0].seen==12 and c.banks[0].positions==[0,1,2,3,4,6,8]
        restored=LiveController(max_tokens=32,research=settings);restored.restore(c.snapshot(),'cpu','r')
        assert state_digest(restored.snapshot())==state_digest(c.snapshot())
        assert c.trace[-1]['prompt_readout_weight']==.7
        return dict(cap=7,kv_seen=12,additive_formula_verified=True)
    def capped_prefill():
        q,k,v=tensors();c=LiveController(max_tokens=32,research=ResearchSettings(max_queries=2),acc_dtype=torch.float64)
        c.begin_prompt(6);c.reset(LivePolicy())
        base=torch.zeros(1,6,6,5,device='cpu',dtype=torch.float64)
        c.apply(0,q[...,:6,:],k[...,:6,:],v[...,:6,:],base,.4)
        b=c.banks[0];assert b.n==2 and b.seen==6
        return b.verify(k[...,:6,:],v[...,:6,:],atol=1e-12,rtol=1e-12)
    def default_regression():
        torch.manual_seed(122);model=_ToyLM().eval()
        maximum=0.;count=0
        for mode in ('clean','stream'):
            for strength in (0.,.1,.55,1.):
                for dtype in (torch.float32,torch.float16):
                    model=model.to(dtype)
                    old=_BaseLiveController(max_tokens=64,trace_every=0)
                    new=LiveController(max_tokens=64,trace_every=0,research=ResearchSettings())
                    kw=dict(max_new=12,prefill_mode=mode,seed=2,max_forks=0,event_copies=())
                    a=free_source(LiveToyRunner(model,old),LivePolicy(strength=strength),[1,3,7],**kw)
                    b=free_source(LiveToyRunner(model,new),LivePolicy(strength=strength),[1,3,7],**kw)
                    assert a['token_ids']==b['token_ids'] and logits_digest(a['_logits'])==logits_digest(b['_logits']);count+=1
        return dict(free_conditions_bitwise_equal=count)
    def forks_settings():
        torch.manual_seed(38);model=_ToyLM().eval();checks=0
        for settings in (ResearchSettings(current_attenuation=0),ResearchSettings(prompt_share=1),ResearchSettings(prompt_share=.5,admit_every=3,max_queries=7)):
            c=LiveController(max_tokens=64,research=settings,trace_every=1);runner=LiveToyRunner(model,c);states=[]
            def cap(s,r):states.append(s);return dict(index=0)
            root=free_source(runner,LivePolicy(strength=.3),[1,7,3],max_new=16,prefill_mode='clean',seed=41,
                             temperature=.8,top_p=.95,fork_at=(5,),event_copies=(),max_forks=1,checkpoint_sink=cap)
            continued=free_branch(runner,states[0],'continue',max_new=8,temperature=.8,top_p=.95)
            continuation_control(root,states[0],continued,0.)
            frozen=free_branch(runner,states[0],'freeze',max_new=8,temperature=.8,top_p=.95)
            assert torch.equal(frozen['_logits'][0],continued['_logits'][0]);checks+=1
        return dict(settings_tested=checks,sampled_continuation_reproduced=True)
    def kernel_contract():
        q,k,v=tensors(dtype=torch.float32)
        def original(m,q,k,v,mask,scaling,softcap=None,**kw):
            s=gqa_scores(q,k,scaling)
            if softcap:s=torch.tanh(s/softcap)*softcap
            if mask is not None:s=s+mask
            p=s.softmax(-1);return gqa_values(p,v).transpose(1,2).contiguous(),p
        for cap in (None,2.):
            c=LiveController(max_tokens=32,trace_every=0)
            module=SimpleNamespace(training=False,layer_idx=0,_qcache_lab_original_eager=original,_qcache_lab_controller=c)
            mask=torch.zeros(1,1,4,4,device='cpu').masked_fill(torch.ones(4,4,dtype=torch.bool,device='cpu').triu(1),-torch.inf)
            native,_=original(module,q[...,:4,:],k[...,:4,:],v[...,:4,:],mask,scaling=.4,softcap=cap)
            out,_=hf_attention(module,q[...,:4,:],k[...,:4,:],v[...,:4,:],mask,scaling=.4,softcap=cap)
            assert torch.equal(native,out) and c.banks[0].softcap==cap
            bad=native+2.
            rejected(lambda:_validate_native_attention(module,q[...,:4,:],k[...,:4,:],v[...,:4,:],mask,bad,.4,cap,'BTHD'))
        return dict(plain_and_softcap_probed=True,bad_kernel_rejected=True,real_hf=False)
    def graders():
        for kind,answer,good,bad in [('exact','ABC','ABC','ABC later'),('number','42','42.0','41'),('json',{'x':1},'{"x":1}','{"x":true}')]:
            t=dict(grader=dict(kind=kind,answer=answer))
            assert grade_answer(good,t)['passed'] and not grade_answer(bad,t)['passed']
        assert not grade_answer('{"x":1,"x":1}',dict(grader=dict(kind='json',answer={'x':1})))['passed']
        t=dict(grader=dict(kind='number',answer=7,extract='final_line'))
        assert grade_answer('reasoning\nFINAL: 7',t)['passed'] and not grade_answer('FINAL: 7\nother text',t)['passed']
        return dict(no_generated_code_execution=True,strict_parsers=True)
    def selection():
        def summary(score,rep):return dict(macro_score=score,repeat_rate=rep,per_task={'a':{'score':score}})
        presets=[dict(id='native'),dict(id='sharper_only'),dict(id='good')]
        sel=select_development_preset({'native':summary(.5,0),'sharper_only':summary(.5,0),'good':summary(.7,0)},presets)
        assert sel['selected_preset']['id']=='good'
        sel=select_development_preset({'native':summary(.5,0),'sharper_only':summary(.9,.5),'good':summary(.5,0)},presets)
        assert sel['selected_preset']['id']=='native'
        lock=dict(schema='qcache.preset_lock.1.4',binding={'x':1},selection=sel,development_prompt_hashes=[_json_hash('dev')])
        rejected(lambda:validate_test_lock(lock,[{'split':'test','prompt':'dev'}],{'x':1}))
        rejected(lambda:validate_test_lock(lock,[{'split':'test','prompt':'new'}],{'x':2}))
        return dict(no_gain_retains_native=True,repeat_constraint_applied=True,test_overlap_rejected=True)
    def assessment_e2e():
        torch.manual_seed(601);model=_ToyLM().eval()
        model.config=SimpleNamespace(num_hidden_layers=3,vocab_size=97,max_position_embeddings=128,model_type='toy')
        model.generation_config=SimpleNamespace(eos_token_id=None)
        class Tok:
            chat_template=None;eos_token_id=None
            def encode(self,text,add_special_tokens=True):return [1,4,8]
            def decode(self,ids,skip_special_tokens=False):return ' '.join(str(i) for i in ids)
        class no_adapter:
            def __init__(self,m,c):pass
            def close(self):pass
        tasks=[dict(id=f'{split}_{i}',split=split,task='toy_fixture',prompt=f'{split} synthetic {i}',grader={'kind':'exact','answer':'not a toy answer'})
               for split in ('dev','test') for i in range(2)]
        presets=[dict(id='native',lambda_value=0.,research=asdict(ResearchSettings())),
                 dict(id='candidate',lambda_value=.1,research=asdict(ResearchSettings(prompt_share=.5)))]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);tp=path/'tasks.jsonl';tp.write_text('\n'.join(json.dumps(t) for t in tasks))
            args=assess_parser().parse_args(['--model','toy','--tasks',str(tp),'--split','dev','--presets','fixture','--max-tokens','8'])
            args.logits_storage='disk'
            d=path/'dev';d.mkdir()
            report=assess_loaded(model,Tok(),LiveToyRunner,no_adapter,tasks,presets,args,d,{'fixture':'random weights'})
            assert report['status']=='completed' and len(report['rows'])==4
            assert report['development_selection']['selected_preset']['id']=='native'
            args.split='test';args.presets=None;args.lock=str(d/'preset_lock.json')
            td=path/'test';td.mkdir()
            r=assess_loaded(model,Tok(),LiveToyRunner,no_adapter,tasks,None,args,td,{'fixture':'random weights'})
            assert r['status']=='completed' and len(r['rows'])==2 and not r['selection_on_test']
            forced=_load_json(Path(args.lock));forced['selection']['selected_preset']=presets[1]
            fixture=path/'fixture_lock.json';json_write(fixture,forced);args.lock=str(fixture)
            t2=path/'test_candidate';t2.mkdir()
            r2=assess_loaded(model,Tok(),LiveToyRunner,no_adapter,tasks,None,args,t2,{'fixture':'random weights'})
            assert r2['status']=='completed' and len(r2['rows'])==4 and 'candidate' in r2['paired_differences']
            return dict(dev_answers=4,test_answers=2,locked_candidate_path_answers=4,
                        candidate_selection='manually forced TEST FIXTURE, no measured toy improvement',
                        forward_engine='real random-weight toy decoder',real_hf=False)
    def bhtd_kernel():
        q,k,v=tensors(dtype=torch.float32)
        def original(m,q,k,v,mask,scaling,**kw):
            s=gqa_scores(q,k,scaling)
            if mask is not None:s+=mask
            p=s.softmax(-1)
            return gqa_values(p,v).contiguous(),p
        c=LiveController(max_tokens=32,trace_every=0)
        module=SimpleNamespace(training=False,layer_idx=0,_qcache_lab_original_eager=original,_qcache_lab_controller=c)
        # T=H=6 would be ambiguous without the one-query probe.
        mask=torch.zeros(1,1,6,6,device='cpu').masked_fill(torch.ones(6,6,dtype=torch.bool,device='cpu').triu(1),-torch.inf)
        native,_=original(module,q[...,:6,:],k[...,:6,:],v[...,:6,:],mask,scaling=.4)
        out,_=hf_attention(module,q[...,:6,:],k[...,:6,:],v[...,:6,:],mask,scaling=.4)
        assert module._qcache_layout=='BHTD' and torch.equal(out,native)
        return dict(ambiguous_equal_heads_and_query_length_handled=True)
    def discovery_paths():
        from types import ModuleType
        import sys
        mod=ModuleType('qcache_test_attention_module')
        mod.eager_attention_forward=lambda *a,**k:None
        class NovelAttention(nn.Module):
            def __init__(self,i):super().__init__();self.layer_idx=i;self.is_causal=True
            def forward(self,hidden_states):return hidden_states
        NovelAttention.__module__=mod.__name__
        class Shell(nn.Module):
            def __init__(self):
                super().__init__();self.strange_path=nn.ModuleList([NovelAttention(0),NovelAttention(1)])
                self.config=SimpleNamespace(num_hidden_layers=2,vocab_size=97,model_type='unlisted_model')
            def set_attn_implementation(self,n):pass
        with patch.dict(sys.modules,{mod.__name__:mod}):
            found=discover_attention_modules(Shell());assert [x[1] for x in found]==['strange_path.0','strange_path.1']
        return dict(no_model_dot_model_dot_layers_dependency=True,real_hf=False)
    def disk_source_fork():
        torch.manual_seed(179);model=_ToyLM().eval();states=[]
        def run(storage,capture=False):
            c=LiveController(max_tokens=64,research=ResearchSettings(logits_storage=storage));r=LiveToyRunner(model,c)
            def save(s,reasons):states.append(s);return dict(memory=0)
            result=free_source(r,LivePolicy(strength=.15),[1,3,5],max_new=16,prefill_mode='clean',seed=47,
                               fork_at=(5,),checkpoint_sink=save if capture else None,event_copies=(),max_forks=1)
            return result,r
        a,_=run('memory');b,r=run('disk',True)
        assert isinstance(b['_logits'],LogitTape) and logits_digest(a['_logits'])==logits_digest(b['_logits'])
        branch=free_branch(r,states[0],'continue',max_new=8)
        continuation_control(b,states[0],branch,0.)
        return dict(memory_and_disk_identical=True,disk_continue_verified=True)
    def coupled_cut_native():
        torch.manual_seed(780);m=_ToyLM().eval();c=LiveController(max_tokens=64,research=ResearchSettings(current_attenuation=0));r=LiveToyRunner(m,c);states=[]
        def cap(s,x):states.append(s);return dict(index=0)
        free_source(r,LivePolicy(strength=.2),[1,2,3],max_new=12,prefill_mode='clean',seed=2,
                    fork_at=(4,),event_copies=(),checkpoint_sink=cap,max_forks=1)
        a=free_branch(r,states[0],'cut',max_new=6);b=free_branch(r,states[0],'native',max_new=6)
        assert logits_digest(a['_logits'])==logits_digest(b['_logits'])
        return dict(additive_cut_equals_native=True)
    def task_validation():
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'t.jsonl'
            rows=[dict(id=split,task='fixture',split=split,prompt='same question',grader=dict(kind='exact',answer='a')) for split in ('dev','test')]
            p.write_text('\n'.join(json.dumps(t) for t in rows));rejected(lambda:read_tasks(p))
            p.write_text(json.dumps(dict(id='x',task='fixture',split='dev',prompt='x',grader=dict(kind='json'))));rejected(lambda:read_tasks(p))
        return dict(split_overlap_and_missing_key_rejected=True)

    with torch.device('meta' if non_cpu_default else 'cpu'):
        for name,fn in [('context_resolution',lengths),('disk_logits_exact_roundtrip',tape),('no_model_name_allowlist',config_gate),
                        ('plain_online_dense',lambda:recurrence(None)),('softcap_online_dense',lambda:recurrence(2.)),
                        ('independent_gain_pool_admission',research_algebra),('explicit_query_cap_prefill',capped_prefill),
                        ('default_free_regression',default_regression),('new_settings_fork_restore',forks_settings),
                        ('native_score_contract_stub',kernel_contract),('strict_answer_graders',graders),
                        ('dev_selection_and_test_lock',selection),('free_assessment_end_to_end_toy',assessment_e2e),('native_BHTD_layout',bhtd_kernel),
                        ('arbitrary_module_paths',discovery_paths),('disk_source_fork',disk_source_fork),
                        ('additive_cut_native_control',coupled_cut_native),('task_input_validation',task_validation)]:test(name,fn)
    report=dict(schema='qcache.v14_tests',environment=environment(),passed=sum(t['status']=='passed' for t in tests),
                failed=sum(t['status']=='failed' for t in tests),tests=tests,
                scope='CPU math/state/selection plus random toy free-generation integration; not HF pretrained behavior.',non_cpu_default_meta=non_cpu_default)
    json_write(out,report);return report['failed']==0


def v14_main():
    if len(sys.argv)==1 or sys.argv[1] in {'--help','-h'}:
        print('QCache Lab V1.4\n\n  explore       free sweep -> refine -> same-state forks\n  run           manual free trajectories and forks\n  probe         actual model capability/no-op/cache/fork check\n  assess        free-answer task evaluation, dev selection and locked test\n  research-test new capacity/adapter/math/evaluation tests\n  self-test     retained mathematical and state tests\n  campaign-test retained campaign/resume integration tests\n  hf-smoke      random tiny HF classes, including expanded adapters\n  device-check  CPU metric placement check\n  legacy-run    explicit old fixed-token lab\n\nUse <command> --help. No model-name whitelist; mathematical/state requirements still apply.')
        return 0
    command=sys.argv[1]
    try:
        if command=='explore':
            args=campaign_parser().parse_args(sys.argv[2:]);configure_research(args);run_campaign(args);return 0
        if command=='assess':run_assess(assess_parser().parse_args(sys.argv[2:]));return 0
        if command=='probe':
            p=argparse.ArgumentParser(description='Probe actual model attention, native no-op, cache restore and forks.')
            p.add_argument('--model',required=True);p.add_argument('--revision',default='main');p.add_argument('--local-files-only',action='store_true')
            p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda']);p.add_argument('--dtype',default='auto',choices=['auto','float32','float16','bfloat16'])
            p.add_argument('--max-context','--context',type=context_argument,default='auto');p.add_argument('--out',default='qcache_v14_probe.json')
            return 0 if run_probe(p.parse_args(sys.argv[2:])) else 2
        if command=='research-test':
            p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('qcache_v14_research_tests.json'));p.add_argument('--non-cpu-default',action='store_true')
            a=p.parse_args(sys.argv[2:]);return 0 if v14_tests(a.out,a.non_cpu_default) else 1
        if command=='hf-smoke':
            p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('qcache_v14_hf_smoke.json'));p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto'])
            a=p.parse_args(sys.argv[2:]);return 0 if extended_hf_smoke(a.out,a.device) else 2
        return v13_main()
    except Exception as exc:
        print('ERROR: '+str(exc),file=sys.stderr);traceback.print_exc();return 2


if __name__ == '__main__':
    raise SystemExit(v14_main())
