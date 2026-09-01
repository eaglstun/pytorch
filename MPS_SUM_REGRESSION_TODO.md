# TODO: MPS `sum` regression on small reduced extents (torch 2.13)

> **Upstream status (2026-08-31): superseded.** The merge from `upstream/main`
> includes the newer five-part Faster MPS reductions series (#191097–#191101),
> which replaces the local `3102913a757` implementation described below. The
> merged kernels build successfully and the reduction-focused `test/test_mps.py`
> selections pass: 552 passed, 4 skipped, and 6 expected xfails. Keep the rest of
> this document as historical regression analysis, not as an implementation plan.
>
> The full MPS suite also ran 10,458 tests. Its initial 88 JIT-conformance errors
> were caused by stale headers in the editable checkout's generated
> `torch/include` mirror; after refreshing that mirror, all 88 pass. An allocator
> state assertion also passes in isolation. One unrelated error remains
> reproducible in
> `TestConsistencyMPS.test_output_grad_match_combinations_mps_float16`: relative
> error `0.0013418` exceeds the current `0.001` tolerance for sample input 11.

## ✅ FIXED AND VERIFIED — `3102913a757`, "Fix sum/nansum/mean perf regression at both reduction extremes"

**Status (final, 2026-07-13):** both failure modes are fixed and independently re-verified
from a source build (`torch 2.14.0a0+git3102913`) by the PULSE side. Nothing outstanding.

| reduction                                        | 2.12.1 (good) | 2.13.0 (broken) | **2.14a `3102913`** |
| ------------------------------------------------ | ------------- | --------------- | ------------------- |
| `sum(dim=(3,5))` — expand bwd                    | 0.77 ms       | 107.73          | **0.81 ✅**         |
| `sum(dim=5)` (inner)                             | 0.95          | 27.75           | **1.03 ✅**         |
| `sum(dim=3)` (middle)                            | 0.60          | 61.12           | **1.09 ✅**         |
| `sum(dim=(0,2,3))` — **bcast bwd (the §8 case)** | 0.29          | 3.09            | **0.27 ✅**         |
| `clone()` (control — rules out build confound)   | 0.59          | 0.59            | 0.61                |

**End-to-end, the real model recovers.** PULSE's StyleGAN fwd+bwd hot loop, 6 interleaved
reps on an idle machine: **2.12.1 = 22.09 it/s best / 22.03 median; 2.14a-fixed = 22.33 /
22.21.** Parity, marginally ahead. The 25% regression is gone.

The earlier `8b72abcb031` fixed only the small-extent end and left the tiny-output end
broken (that was §8 below, now resolved by `3102913a757`). **§8 is kept for the record —
it is history, not an open bug.** The lesson in it still stands: a reduction is pathological
at _both_ ends, so sweep the **output** size as well as the reduced extent.

**Still to do:** the upstream issue is drafted (`agent_space/mps_sum_issue_body.md`) but
**not filed**, and no PR is open. Everything is on branch `mps-sum-small-reduced-extent`.

**Written:** 2026-07-13, from an investigation done in a different repo (`~/Documents/dev/pulse`).
**Audience:** a fresh agent. Everything you need is in this file; you do not need that other repo.

---

## 1. The bug in one paragraph

torch **2.13.0** made `sum` / `nansum` / `mean` on MPS dramatically slower **when the
reduced dimension is small** (extent 2, 4, 8...). Full reductions are fine. Large reduced
extents are fine — actually _faster_ than before. The new Metal kernel assigns **one
32-lane SIMD group per output element**, so cost scales with the size of the _output_
rather than with the work done. Reducing a dimension of extent 2 therefore burns 30 idle
lanes and a full 32-lane shuffle to add two floats, across millions of threadgroups.

This is not a niche shape. **The backward pass of `expand` / broadcast is exactly this
reduction** (grad flows back through an expanded dim by summing over it), so it hits:

- any model that upsamples via `view→expand→contiguous` (the classic pre-`F.interpolate`
  StyleGAN/GAN idiom — reduced extent = the upsample factor, i.e. **2**)
- any broadcast-add / FiLM / AdaIN-style conditioning backward (`x * scale + bias` with
  `scale` shaped `[C,1,1]`)

Measured cost on a real model (StyleGAN 1024×1024 fwd+bwd, batch 1, M4 Max): **22.4 it/s
on torch 2.10–2.12 → 16.7 it/s on 2.13, a 25% regression.** The forward pass is completely
unaffected; it is entirely in the backward.

## 2. Evidence (measured, M4 Max / 64GB / macOS)

All numbers from **released PyPI wheels**, not source builds.

**Isolating forward vs backward** (StyleGAN synthesis, batch 1):

|                     | torch 2.12.1 | torch 2.13.0        |
| ------------------- | ------------ | ------------------- |
| forward (`no_grad`) | 27.2 ms      | 24.1 ms (_faster_)  |
| forward (w/ graph)  | 27.5 ms      | 27.5 ms (identical) |
| **backward**        | **25.0 ms**  | **41.2 ms (1.65×)** |

**Per-op bisect** — every conv is untouched; only the broadcast/expand reductions move:

| backward op (33.5M elems)                               | 2.12.1  | 2.13.0       |                      |
| ------------------------------------------------------- | ------- | ------------ | -------------------- |
| conv2d 3×3, conv_transpose2d, instance_norm, leaky_relu | —       | —            | **1.00× (all fine)** |
| `expand(...).contiguous()`                              | 2.41 ms | **87.07 ms** | **36×**              |
| broadcast-add                                           | 1.18 ms | 1.68 ms      | 1.42×                |

**The decisive experiment** — hold total elements fixed at 33.5M, vary **only** the size of
the reduced dim. 2.12 is flat; 2.13 scales as `1/extent`:

| reduced extent | 2.12 `sum(dim=-1)` | 2.13 `sum(dim=-1)` | 2.12 `sum(dim=1)` | 2.13 `sum(dim=1)` |
| -------------- | ------------------ | ------------------ | ----------------- | ----------------- |
| 2              | 0.93 ms            | **25.63 ms**       | 0.43 ms           | **30.50 ms**      |
| 4              | 0.44               | 12.90              | 0.35              | 19.53             |
| 8              | 0.29               | 6.42               | 0.30              | 7.31              |
| 32             | 0.26               | 1.64               | 0.28              | 1.87              |
| 128            | 0.25               | 0.49               | 0.27              | 1.78              |
| 1024           | 0.25               | **0.24 (fine!)**   | 0.29              | 1.21              |

Note the new kernel is **better than the old one at extent 1024**. It is not bad code — it
is optimised for the wrong regime. Also note `sum()` (full reduce) is unaffected (0.33 →
0.29 ms), and `clone()` is unaffected (0.58 → 0.58 ms), so this is not bandwidth or
allocation.

## 3. Root cause

Culprit: **`10a41bb2626` — "[MPS] Replace sum/nansum/mean ops with native Metal kernel"
(#180709)**, by Nikita Shulga. It was landed, reverted, then re-landed. It moved `sum` off
MPSGraph's `reductionSumWithTensor:` onto hand-written Metal shaders dispatched through the
common `sum_stub` TensorIterator path. (Follow-up `#182688 "[MPS] Sum sliced reduction fix"`
shows this kernel had already caused trouble.)

**Two separable defects. Verified still present on `main` (`84cc6889912`, checked
2026-07-13).**

### (a) The inner kernel gives a whole SIMD group per output element

`aten/src/ATen/native/mps/kernels/ReduceOps.metal`, `sum_reduction_inner` (~line 565 on
main). With `SUM_NCHAINS = 8` (`kernels/ReduceOps.h:5`):

```metal
// Each SIMD group handles a different row
uint row = tgid * num_simd_groups + simdgroup_id;
...
const uint stride = 32 * NCHAINS;             // = 256
const uint aligned_N = (N / stride) * stride; // N=2  ->  0
for (; base < aligned_N; base += stride) { ... }   // <-- NEVER EXECUTES for N < 256
// Tail: remaining elements after last full block, one per lane
for (uint i = aligned_N + simd_lane_id; i < N; i += 32) { acc[0] += ...; }  // 30/32 lanes idle
sum = c10::metal::simd_sum(sum);              // full 32-lane shuffle to add 2 floats
```

So for `N < 256` the vectorised loop is skipped **entirely**; for `N < 32` most lanes idle;
and the grid is `M = numel/N` simdgroups (16.7M of them at N=2). Cost ∝ output numel, which
is exactly the measured `1/extent` curve.

### (b) No fast path for middle / multi-dim reductions

`aten/src/ATen/native/mps/operations/ReduceOps.mm`, `reduction_dispatch_mps` (line **874**
on main). It only specialises:

- line **910**: `num_reduced == 1 && reduced_dim == 0` (outer kernel)
- line **937**: `num_reduced == 1 && reduced_dim == input.dim()-1` (inner kernel)
- line **969**: `output.numel() == 1` (full reduce, split path)

**Everything else** — any middle-dim reduction, any multi-dim reduction like
`sum(dim=(3,5))` (which is what `ExpandBackward` / `sum_to_size` produces) — falls to the
generic single-pass kernel at line **1053**, which stays **~4× slower than 2.12 even at
extent 1024**, where the inner kernel has fully recovered.

## 4. Repro scripts

Self-contained; no models, no other repos. Run under 2.12.1 and 2.13.0 (or a source build)
and diff.

### 4a. Minimal repro (the one to put in the issue)

```python
import time, torch
dev = torch.device("mps")

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.mps.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.mps.synchronize()
    return (time.time() - t0) / iters * 1000

# ExpandBackward: grad flows back through an expanded dim by summing over it.
x = torch.randn(1, 32, 512, 512, device=dev, requires_grad=True)
def upsample_bwd():
    (x.view(1, 32, 512, 1, 512, 1)
      .expand(-1, -1, -1, 2, -1, 2)
      .contiguous()
      .view(1, 32, 1024, 1024)
      .sum().backward())
print(f"expand-upsample backward: {bench(upsample_bwd):.2f} ms")
# torch 2.12.1:  2.41 ms
# torch 2.13.0: 87.07 ms   <-- 36x
```

### 4b. The characterising sweep (proves it's the reduced extent, not the tensor size)

```python
import time, torch
dev = torch.device("mps")
TOTAL = 32 * 1024 * 1024  # element count held CONSTANT across every row

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.mps.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.mps.synchronize()
    return (time.time() - t0) / iters * 1000

print(f"torch {torch.__version__}  (total elements fixed at {TOTAL:,})")
print(f"{'reduced extent':>14} {'sum(dim=-1)':>13} {'sum(dim=1)':>12}")
for red in (2, 4, 8, 32, 128, 1024):
    x = torch.randn(TOTAL // red, red, device=dev)                     # reduce last dim
    y = torch.randn(64, red, TOTAL // (64 * red), device=dev)          # reduce a MIDDLE dim
    print(f"{red:>14} {bench(lambda: x.sum(dim=-1)):>10.2f} ms {bench(lambda: y.sum(dim=1)):>9.2f} ms")
```

Expect: 2.12 flat (~0.25–0.93 ms everywhere); 2.13 blowing up as the extent shrinks.

## 5. The two deliverables

### Task A — file the upstream issue (cheap, do first, no build required)

Open on `github.com/pytorch/pytorch`. Include §1 framing, the §2 tables, the §3 root cause
with the shader snippet, and the §4a + §4b repros. Reference `#180709` and cc/mention
Nikita Shulga (`@malfet`) — it's his commit and he's active. Title suggestion:

> [MPS] `sum` regression on small reduced extents since #180709 (36× on `expand` backward)

**Value:** this may simply get fixed upstream for free. It also affects far more than the
model I found it on.

### Task B — fix it and submit upstream (expensive: a from-source macOS build is ~1hr+)

**Fix direction.** In `reduction_dispatch_mps`, add a threshold on the reduced extent. When
`N` is small, the simdgroup-per-row geometry is wrong — use **thread-per-output-row** (each
thread serially sums its own `N` elements, no simd shuffle at all), or route small-`N` back
to MPSGraph. Consider also giving middle-/multi-dim reductions a real fast path (defect b),
though (a) is the big one and can ship alone.

Sensible threshold to start from: the vectorised loop needs `N >= 32*NCHAINS = 256` to run
at all, and the simd shuffle only pays for itself somewhere above `N ~ 32`. The data says
2.13 is at parity by extent ~1024 and clearly broken by extent ~32. Tune it, don't guess it.

**Acceptance criteria:**

1. §4b: extent 2 drops from ~25 ms back to ~1 ms (2.12 parity).
2. **No regression at large extents** — extent 1024 must stay at ~0.24 ms, where the new
   Metal kernel genuinely beats MPSGraph. Do not "fix" this by reverting to MPSGraph
   wholesale; the new kernel is _better_ in its intended regime.
3. §4a: `expand` backward returns to ~2.4 ms.
4. Correctness: `sum`/`nansum`/`mean` still match CPU across dtypes and shapes. Use this
   repo's own harness, not bare pytest: `python test/test_mps.py -k "sum or mean or nansum"`.
5. Ideally: a real model recovers. StyleGAN-ish fwd+bwd should go 16.7 → ~22 it/s.

## 6. Repo state / hygiene — READ `CLAUDE.md` FIRST, IT OVERRIDES ME

**This repo's `CLAUDE.md` (== `AGENTS.md`) is strict and load-bearing. It is upstream
PyTorch's workflow, not a normal fork workflow. Where it disagrees with this file, it wins.**
The bits that will bite you here:

- **Git: ghstack, NOT `git push`.** A detached HEAD is intentional; commit onto it. **Never
  `git push`**, never touch `gh/USERNAME/N` branches, never drop the `Pull-Request:` /
  `ghstack-source-id:` trailers. Submit a single commit with `ghstack --no-stack`. (There is
  no `~/.ghstackrc` yet — it needs setting up.) Don't commit unless asked; commit messages
  need a Test Plan with literal commands, a root-cause explanation, and AI-assistance
  disclosure.
- **Build: only ever `pip install -e . -v --no-build-isolation`.** No other command, ever.
- **There is no venv, and torch is not built** (`import torch` from the repo root fails);
  `spin` isn't installed either. Per `CLAUDE.md`: if `pip`/`python`/`spin` are missing and
  there's no `.venv`, **stop and ask Eric.** Do not go install things on your own. Standing
  up the build is itself a big task (~1hr+; `.git` alone is 1.2 GB).
- **Lint:** `spin lint` / `spin fixlint`; `lintrunner -a` before any commit.
- Scratch work goes in `agent_space/` (git-ignored). **Never touch `.ci/docker/`** — it's
  content-hashed and any edit forces a full Docker image rebuild.

**Use the in-repo skills** (`.claude/skills/`) — there are 16, and two are directly on point:
**`metal-kernel`** (MPS operator support — exactly this task) and **`fix-issue`**.
`/pr-review` is the required path for reviewing a PR here.

Fork facts: `origin = git@github.com:eaglstun/pytorch.git`, on `main` at `84cc6889912`.
Upstream `pytorch/pytorch` is **not** configured as a local remote; add it fetch-only if you
need tags: `git remote add upstream https://github.com/pytorch/pytorch.git && git fetch --tags upstream`

## 7. What is verified vs. what is not — read this before you start

**Verified:** the timings in §2 (released wheels, interleaved A/B, reproducible to 3
significant figures — not thermal noise); the source analysis in §3 (read directly from the
`v2.13.0` tag and re-checked against `main`); that the offending kernel geometry and the
`SUM_NCHAINS = 8` constant are **still present on `main` today**.

**NOT verified:** that the bug reproduces in a **from-source build of `main`** (only
released wheels were tested), and that the proposed fix works — **nothing has been built or
patched.** Your first move after the issue is filed should be: build `main` unmodified, run
§4b, and confirm the curve looks like the 2.13 column. If it doesn't, something between
2.13.0 and `main` already fixed it and Task B is moot — check that before spending an hour
compiling a second time.

**Benchmarking caveat for this machine:** Eric runs a flash-attention MPS test suite out of
`/tmp/fa-mps-venv` (from `~/Documents/dev/flash-attention`) that contends for the GPU and
silently wrecks timings — a contended run read 12–16 it/s with 12% spread where a clean one
read 19.5 with 0.07. Check `pgrep -f "fa[-]mps-venv"` before timing anything (the bracket
stops `pgrep` matching its own command line). Also watch for `mediaanalysisd` and VS Code's
`cpptools` indexer, both of which can sit at 100–180% CPU. Correctness checks are immune;
only timings are affected. **Always include a `clone()` timing as a control** — if `clone()`
matches across builds, the build itself is not the variable.

---

# 8. ⚠️ RESIDUAL: the fix does NOT cover the tiny-output case (found 2026-07-13, after the fix)

Measured against the **built, fixed** `torch 2.14.0a0+git8b72abc` (branch
`mps-sum-small-reduced-extent`). Three interleaved reps, reproducible to 2 decimal places,
with `clone()` as a control to rule out any source-build-vs-wheel confound.

## What the fix DID fix (all restored to 2.12 parity — this part works)

| reduction                     | 2.12.1  | 2.13.0 | **2.14a fixed** |
| ----------------------------- | ------- | ------ | --------------- |
| `sum(dim=(3,5))` (expand bwd) | 0.77 ms | 107.73 | **0.76 ✅**     |
| `sum_to_size(...)`            | 0.89    | 107.69 | **1.05 ✅**     |
| `sum(dim=5)` (inner)          | 0.95    | 27.75  | **1.06 ✅**     |
| `sum(dim=3)` (middle)         | 0.60    | 61.12  | **1.06 ✅**     |

## What is STILL BROKEN

|                                              | 2.12.1      | 2.13.0 | **2.14a fixed**                               |
| -------------------------------------------- | ----------- | ------ | --------------------------------------------- |
| **`sum(dim=(0,2,3))` on `[1,32,1024,1024]`** | **0.32 ms** | 3.09   | **2.40 — still 7.5× slow ❌**                 |
| broadcast `x + w.view(1,-1,1,1)*n` fwd+bwd   | 3.76        | —      | 5.77                                          |
| `clone()` (control)                          | 0.58        | 0.59   | 0.57 — **identical, so not a build artifact** |

## Why the existing fix doesn't reach it — it's the OPPOSITE failure mode

- §3's bug: **tiny reduced extent, huge output** (N=2, M=16.7M) → 16.7M simdgroups, each
  doing almost no work. Fixed by thread-per-output.
- This bug: **huge reduced extent, tiny output** (reduce 1×1024×1024 = 1M elements down to
  32 channels). One simdgroup per output element gives **32 simdgroups for a 33M-element
  tensor** — the GPU is almost entirely idle. It needs the _opposite_ treatment: split the
  reduction across many threadgroups and do a second pass (the `output.numel() == 1` split
  path at ReduceOps.mm:969 already does exactly this — it just isn't reached when
  `output.numel() == 32`).

**This shape is arguably more common than the one already fixed.** `sum(dim=(0,2,3))` is the
parameter-gradient reduction for **every per-channel broadcast**: conv bias, FiLM, AdaIN,
StyleGAN `StyleMod`/`NoiseLayer`, any `[C,1,1]`-shaped affine. It is on the backward path of
a large fraction of CNNs.

## Repro

```python
import time, torch
dev = torch.device("mps")

def bench(fn, iters=30, warmup=8):
    for _ in range(warmup): fn()
    torch.mps.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.mps.synchronize()
    return (time.time() - t0) / iters * 1000

x = torch.randn(1, 32, 1024, 1024, device=dev)   # 33.5M elems, 32 outputs
print(f"sum(dim=(0,2,3)): {bench(lambda: x.sum(dim=(0,2,3), keepdim=True)):.2f} ms")
print(f"clone() control : {bench(lambda: x.clone()):.2f} ms")
# 2.12.1: 0.32 ms / 0.58 ms      2.14a-fixed: 2.40 ms / 0.57 ms
```

## Extra acceptance criteria (add these — §5's were insufficient)

6. `sum(dim=(0,2,3))` on `[1,32,1024,1024]` returns to ~0.3 ms (2.12 parity).
7. Sweep the **output** size as well as the reduced extent. §4b only sweeps the extent while
   holding the output large, which is exactly why this slipped through. A reduction is
   pathological at **both** ends: too many outputs (each doing no work) and too few (leaving
   the GPU idle).

## Why this matters downstream

The PULSE repo (`~/Documents/dev/pulse`) is pinned `torch<2.13`. It was re-benchmarked on
this fixed build and **did not recover** (~16.4 it/s vs 19.4 on 2.12): its own fix already
replaced the `expand()`-based upsample with `F.interpolate`, so the small-extent fix has
nothing left to bite on there, while its `StyleMod`/`NoiseLayer` broadcasts hit precisely
this residual. **The `torch<2.13` pin stays until §8 is fixed too.**
