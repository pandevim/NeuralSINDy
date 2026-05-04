# The Big Picture

You're trying to solve **system identification** — given time-series data of a physical system, automatically discover the governing differential equations. If you have measurements `x(t)` of a pendulum, can a machine recover `ẍ = -sin(x)` without you telling it that gravity exists?

This is the SINDy problem: **Sparse Identification of Nonlinear Dynamics** (Brunton, Proctor, Kutz 2016). It matters because most real-world systems have governing equations that are _sparse_ — only a few terms out of an infinite possible library actually appear in the true dynamics. Newton's laws, Lotka-Volterra, Navier-Stokes — all sparse in some natural basis.

## How standard SINDy works

You write the dynamics as $\dot{X} = \Theta(X) \cdot \xi$ where:

- $\dot{X}$ is the time derivative you measure (or compute numerically from $X$)
- $\Theta(X)$ is a **library matrix**: each column is a candidate function evaluated on your data — `[1, x, x², sin(x), cos(x), x·y, ...]`
- $\xi$ is a sparse coefficient vector: most entries are zero, the nonzero ones tell you which functions appear in the true equation

You solve for $\xi$ with sparse regression (typically STLSQ — Sequential Thresholded Least Squares: solve the regression, zero out small coefficients, refit, repeat).

## The limitation you're attacking

Standard SINDy uses a **hand-crafted, fixed library**. You decide a priori that the basis is polynomials up to degree 5 plus sines and cosines. If the true dynamics involve a term you didn't include, you can't find it. If your library is too big, you have collinearity and conditioning problems. The library is a human guess and the whole pipeline is brittle to it.

## Your idea: Neural SINDy

Replace the hand-crafted basis functions with **MLPs that have been trained to perfectly approximate primitive operations**. Each MLP $f_\theta$ is a neural network that has internalized one operation — one for `identity`, one for `sin`, one for `cos`, one for `x+y`, one for `x·y`. The library matrix becomes:

$$\Theta_{\text{neural}}(X) = [\text{MLP}_{\text{identity}}(x), \text{MLP}_{\text{sin}}(x), \text{MLP}_{\text{cos}}(x), \text{MLP}_{\text{add}}(x,v), \text{MLP}_{\text{mul}}(x,v), \dots]$$

Because the MLPs are differentiable, you can do end-to-end gradient-based discovery instead of just sparse regression. The promise: a swappable, composable, learnable library — and eventually a path to MLPs that have grokked _more general_ operations than `sin`/`cos`/`mul`.

The reason this is interesting (and not obviously redundant — why use an MLP to compute `sin(x)` when you have `np.sin`?) is the longer-term bet: if you can train MLPs that **grok** operations from data, you can extend the library to operations no one has analytic forms for. The damped oscillator is a sanity-check benchmark, not the destination.

---

# How the file is structured

The notebook walks the full pipeline in five phases. Phase 1–4 are the SINDy pipeline. Phase 5 is your novel contribution — replacing sparse regression with a differentiable router — and it's where the iteration happens.

## Phase 1 — Grok the MLP library

Train five small MLPs to perfectly compute primitive operations:
`cos(x)`, `sin(x)`, `identity(x)`, `x+y`, `x·y`.

The interesting choice here is **grokking**: training way past the point of overfitting (10k–20k epochs) with heavy AdamW weight decay. Grokking is the phenomenon where validation loss stays flat (or worsens) while training loss drops, and then _much_ later the network suddenly generalizes. The internal weights restructure into a clean algorithmic circuit. The grokked MLPs become near-exact approximators on their domains, not noisy interpolators.

These get saved as checkpoints (`mlp_identity.pt`, `mlp_cos.pt`, ...) and frozen for everything downstream.

## Phase 2 — Generate test data

Pick a known system to test discovery against. You chose the **damped harmonic oscillator**:

$$\ddot{x} = -k x - c \dot{x}$$

with $k=1.0$, $c=0.1$. Rewritten as a first-order system:

$$\dot{x} = v, \quad \dot{v} = -1.0 \cdot x - 0.1 \cdot v$$

Simulate 600 time points with `solve_ivp`, add small Gaussian noise. Save as `(t, x, v, ẋ, v̇)`. The ground truth is two linear terms in the first equation and two linear terms in the second (with damping 10× weaker than the restoring force — this becomes the hardest part to recover).

## Phase 3 — Build the library

Load all five grokked MLPs. Construct the 8-column library matrix by evaluating each MLP on every state:

```
['identity(x)', 'identity(v)', 'cos(x)', 'cos(v)', 'sin(x)', 'sin(v)', 'add(x,v)', 'mul(x,v)']
```

This is the basis the discovery algorithm gets to choose from. The true equations live entirely in `identity(x)` and `identity(v)` — everything else should ideally come back with coefficient ≈ 0.

## Phase 4 — STLSQ baseline

Run classical SINDy on top of the neural library. Solve $\dot{X} = \Theta_{\text{neural}}(X) \cdot \xi$ with thresholded least squares.

**Result:** Numerically perfect (MSE 1e-6) but **uninterpretable**. The discovered equations spread coefficients across `identity(x)`, `identity(v)`, **and `add(x,v)`**:

$$\dot{x} = -0.334\,x + 0.666\,v + 0.334\,(x+v)$$

Algebraically, this simplifies to `≈ 1.0·v` — correct! — but as a structural statement it's a mess.

**Why it failed:** The library has an exact collinearity. `add(x,v) = identity(x) + identity(v)`, so the column `MLP_add(x,v)` lies in the span of two other columns. STLSQ has no way to break the tie and distributes the coefficient arbitrarily.

The lesson from Phase 4: the library _design_ is the bottleneck, not the regression algorithm. And — critically — STLSQ's reliance on solving normal equations makes it brittle to rank deficiency. This motivates Phase 5.

## Phase 5 — Gumbel-Softmax Router

Replace STLSQ with a learned **router** that picks which MLP(s) explain each component of the dynamics. Gumbel-Softmax provides a differentiable approximation to discrete sampling: the forward pass routes through one (or k) MLPs, the backward pass uses the soft distribution for gradients via the Straight-Through Estimator.

A temperature τ anneals from 5.0 (soft, exploratory) down to 0.05 (near-discrete). The hope is that you converge to a clean one-hot selection per derivative, with no spread, no collinearity pathology, and a globally optimal sparse equation.

This is where you iterated.

### Exp 1 — Naive Gumbel router

Router is a state-dependent MLP (~9,760 params) producing per-sample logits. No entropy penalty.

**Failed.** Max activation 43.8% on the right term. The router stayed in the soft regime — at high τ it computed a weighted mixture of all MLPs and learned per-MLP coefficients that combined to fit the data without ever concentrating on one term. Damping was lost. Final val MSE ~5e-3, _1000× worse_ than STLSQ.

**Diagnosis:** Nothing in the loss pushed the logits apart. Gumbel-Softmax without sparsity pressure is just a fancy softmax.

### Exp 2 — State-independent router + entropy penalty

Two changes: (1) router becomes a single learnable logit vector per derivative (32 params total — matches SINDy's assumption that one term governs the dynamics globally), (2) add a temperature-scheduled entropy penalty `0.05·(1 − τ/τ_start)·H(π)` to the loss to actively push toward one-hot.

**Committed cleanly — to the wrong terms.** Picked `sin(v)` for `dx/dt` and `sin(x)` for `dv/dt`, both at 99%+ activation. Coefficients +1.0265 and −1.0138.

**Diagnosis:** The library has a _second_ collinearity, this one approximate: for `|x|, |v| ≲ 1`, `sin(z) ≈ z`. Entropy penalty broke the tie arbitrarily. The coefficient inflation (1.0265 vs 1.0) is the giveaway — the coefficient absorbed the Taylor-series shortfall of `sin` vs `identity`. And one-hot routing structurally cannot represent `v̇ = -1·x - 0.1·v` (two terms), so damping was lost regardless.

### Exp 3 — Top-2 routing + complexity prior

Two changes: (1) top-k routing — sample 2 basis functions per derivative via iterative masked Gumbel-Softmax, each with its own coefficient. (2) complexity prior on the logits: `identity` gets a +1.0 bonus, nonlinear terms get a penalty. Occam's razor baked in.

**Structure correct, magnitude wrong.** Slot 1 locked: `identity(v)` for `ẋ` at +1.000, `identity(x)` for `v̇` at −0.985. Slot 2 partially worked: `identity(v)` at 27% selection, coefficient −0.027 (true: −0.1). The damping was recovered at _27% of its true magnitude_.

**Diagnosis:** The entropy penalty was operating on the **base** distribution before slot masking. Once slot 1 committed, slot 2 faced a near-flat conditional distribution and the entropy gradient pointed nowhere useful.

### Exp 4 — Conditional slot entropy + stronger prior (α=1.5)

Two changes: (1) entropy penalty is now computed _per slot, conditional on prior picks_ — iterate through slots, compute entropy of the masked distribution at each step, sum. This gives slot 2 a direct entropy gradient to concentrate on the residual. (2) bump complexity prior α from 1.0 to 1.5 to more decisively break the sin/identity tie.

**Per the printed output, this solved it:**

```
ẋ = +0.9999 · identity(v)
v̇ = −0.9981 · identity(x) − 0.0995 · identity(v)
```

Errors of 0.01%, 0.19%, and 0.5%. Both slots at ~100% selection. Val MSE 1e-6, matching STLSQ's accuracy with full structural interpretability.

**This is the discrepancy I flagged in my last reply** — the analysis text in your file describes a _different_, worse Exp 4 run (val MSE 2.4e-4, damping at −0.034, with spurious `sin(v)`). The printed training log shows the better result. Verify which is real before committing to an Exp 5.

---

# The narrative arc

Read together, the experiments tell a consistent story about _where the difficulty lives_ in this kind of discovery problem:

1. **Phase 4 → Exp 1**: Sparse regression fails on rank-deficient libraries; differentiable routing replaces it but adds a new failure mode (no commitment).
2. **Exp 1 → Exp 2**: Entropy regularization fixes commitment but exposes a _second_ collinearity (sin ≈ identity at small amplitudes) that the regression formulation hid.
3. **Exp 2 → Exp 3**: Top-k routing makes multi-term equations representable; complexity prior breaks approximate collinearity.
4. **Exp 3 → Exp 4**: A subtle bug in _where_ the entropy penalty is computed makes the difference between recovering 27% of the damping and recovering 99.5% of it.

Each iteration didn't just tune hyperparameters — it diagnosed a specific structural problem and added a targeted mechanism. The end state (if Exp 4's printed result holds) is a method that matches STLSQ's numerical accuracy while producing a clean, interpretable equation.

# What you've actually demonstrated

If the Exp 4 printed result is real:

- A grokked-MLP library can serve as a drop-in replacement for analytic basis functions
- Differentiable routing via Gumbel-Softmax can recover correct sparse equations from this library
- The mechanism design that makes it work has three non-trivial pieces: top-k routing, conditional slot entropy, and a complexity prior
- It works on a 2D linear system with a 10× weak secondary term

What you have _not_ yet demonstrated:

- That this generalizes to genuinely nonlinear systems where `mul` and `cos` need to fire (Lorenz, Van der Pol, Lotka-Volterra)
- That the grokked MLPs are _necessary_ — could naive MLPs or analytic functions do the same job?
- Robustness to noise levels beyond the small Gaussian you currently inject
- That any of this scales beyond a 5-MLP library

The work that's left is validation and scope expansion. The core algorithm appears (modulo the analysis discrepancy) to work.
