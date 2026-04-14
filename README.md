# Neural SINDy: Sparse System Identification with Grokked MLP Libraries

> **Can "grokked" neural networks replace hand-crafted mathematical functions in data-driven equation discovery — and can we extract human-readable equations back out?**

---

## Table of Contents

- [Introduction](#introduction)
- [The Problem](#the-problem)
- [Why This Matters](#why-this-matters)
- [Why It's Hard](#why-its-hard)
- [Research Question](#research-question)
- [Proposed Approach](#proposed-approach)
- [Experimental Description](#experimental-description)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Expected Results](#expected-results)
- [References](#references)

---

## Introduction

Discovering governing equations from data is one of the most important problems in scientific machine learning. The **Sparse Identification of Nonlinear Dynamics (SINDy)** framework (Brunton et al., 2016) addresses this by fitting data to a hand-crafted library of candidate mathematical functions — polynomials, trigonometric functions, exponentials — and using sparse regression to select only the active terms.

But what if we replaced those rigid mathematical functions with _neural networks that have learned to be those functions_?

This experiment investigates a novel hybrid: using **small MLPs trained to "grok"** basic mathematical operations (cosine, sine, addition, multiplication) as the basis functions in a SINDy-style discovery framework. The hypothesis is that these neural approximations carry three unique advantages over their symbolic counterparts.

---

## The Problem

**Standard SINDy** requires the user to pre-specify a library of candidate functions. Given state data **X** and its time derivatives **Ẋ**, SINDy solves:

```
Ẋ = Θ(X) · ξ
```

where **Θ(X) = [1, x, x², sin(x), cos(x), ...]** is a hand-designed feature matrix and **ξ** is a sparse coefficient vector found via thresholded regression.

This works remarkably well for textbook systems, but breaks down in practice for three reasons:

### 1. The Library Design Problem

The user must guess which functions belong in the library _before_ seeing the data. If the true dynamics involve `tanh(x)` but the library only includes polynomials and trig functions, SINDy will fail silently — producing a plausible-looking but incorrect equation. There is no systematic way to build the "right" library.

### 2. The Combinatorial Search Problem

Searching over the space of possible equations is inherently discrete — you cannot do gradient descent on the choice between `sin(x)` and `cos(x)`. Standard SINDy sidesteps this with sparse regression, but this limits it to linear combinations of library terms. More complex compositions (e.g., `sin(cos(x))`) require exponentially large libraries.

### 3. The Noise Sensitivity Problem

Real-world measurement data contains noise. Numerical differentiation amplifies this noise, and the rigid mathematical library offers no flexibility to adapt — `cos(x)` is `cos(x)`, regardless of whether the physical system produces a signal that is _almost_ cosine but with systematic sensor-dependent distortions.

---

## Why This Matters

If successful, this approach could:

- **Automate library design** — instead of guessing functions, train a universal set of neural "modules" that cover common mathematical operations
- **Enable gradient-based search** — because MLPs are differentiable, the selection process can leverage gradient descent rather than combinatorial search
- **Improve robustness to noise** — neural approximations carry tiny, idiosyncratic imperfections from training that may actually better fit messy real-world data than perfect mathematical ideals
- **Maintain interpretability** — via symbolic distillation (feeding clean data through the winning MLPs and using symbolic regression to extract the math), we can recover human-readable equations from the neural pipeline

This would bridge the gap between the interpretability of SINDy and the flexibility of neural networks.

---

## Why It's Hard

### Grokking is Fragile

"Grokking" — where a neural network suddenly generalizes long after memorizing the training data — requires careful hyperparameter tuning. The balance between learning rate, weight decay, dataset size, and training duration must be precise. If the MLP memorizes without grokking, it will fail to generalize and the SINDy library becomes useless.

### Neural Basis Functions are Redundant

Unlike perfectly orthogonal mathematical functions (`sin` and `cos` are linearly independent), neural approximations of these functions will be _approximately_ orthogonal at best. This near-collinearity in the library matrix **Θ** can confuse sparse regression, leading to incorrect term selection.

### Symbolic Extraction is Not Guaranteed

Even if the neural pipeline discovers the correct dynamics, converting the opaque weight matrices back into symbolic equations is a separate, hard problem. Symbolic regression (PySR) works well on clean, low-dimensional data but struggles with complex or multi-variate functions.

### Scalability

Each MLP in the library must be trained to grokking, which is computationally expensive. A library of 10 mathematical operations requires training 10 separate networks, each for potentially tens of thousands of epochs.

---

## Research Question

> **Can a SINDy-style sparse regression framework, using grokked MLPs as basis functions instead of analytical mathematical terms, correctly identify the governing equations of a dynamical system — and can symbolic distillation recover the human-readable form of those equations?**

Specifically, we investigate:

1. **MLP Grokking Fidelity** — Do small MLPs reliably grok basic mathematical operations (`cos`, `sin`, `+`, `×`, `identity`) with sufficient accuracy to serve as basis functions?

2. **Neural Library Selection** — Can STLSQ (Sequentially Thresholded Least Squares) sparse regression correctly identify which grokked MLPs explain the dynamics when the library contains both relevant and irrelevant terms?

3. **End-to-End Differentiable Selection** — Can a Gumbel-Softmax router learn to make discrete MLP selections via gradient descent, and does it outperform the classical STLSQ approach?

4. **Coefficient Recovery** — Are the discovered coefficients quantitatively accurate (within 10% of the true parameters)?

5. **Symbolic Distillation** — Can PySR recover the exact symbolic form (e.g., `f(x) = cos(x)`) from the input-output behavior of a grokked MLP with R² > 0.999?

---

## Proposed Approach

### Three Key Insights

**1. "Soft Math" Advantage** — A grokked MLP's approximation of `cos(x)` carries tiny neural imperfections. In real-world scenarios where physical friction, dampening, and sensor noise produce signals that are _almost-but-not-quite_ perfect cosines, these soft approximations may fit better than the rigid mathematical ideal.

**2. Smooth Differentiability via Gumbel-Softmax** — Standard SINDy treats function selection as discrete. Because our library consists of MLPs (continuous, differentiable functions), we can use Gumbel-Softmax routing to make discrete library selection fully differentiable. A learned router network scores each MLP, and the Gumbel-Softmax trick with temperature annealing smoothly transitions from soft exploration (all MLPs contribute) to hard selection (one MLP wins). The Straight-Through Estimator (STE) ensures compute efficiency: only the winning MLP runs during the forward pass, while smooth gradients flow during backpropagation.

**3. Grokked Circuits Generalize** — When an MLP "groks" arithmetic, its weights form an algorithmic circuit (often based on discrete Fourier transforms). This guarantees perfect generalization to unseen inputs, solving the extrapolation problem that normally plagues neural networks.

### Pipeline Architecture

```
                     ┌─────────────────────────────────────────┐
                     │         PHASE 1: GROK THE MLPS          │
                     │                                         │
                     │   cos(x) → MLP_cos    (Tanh, 128h)     │
                     │   sin(x) → MLP_sin    (Tanh, 128h)     │
                     │   x + y  → MLP_add    (ReLU, 128h)     │
                     │   x · y  → MLP_mul    (ReLU, 128h)     │
                     │   x      → MLP_id     (ReLU, 128h)     │
                     └──────────────┬──────────────────────────┘
                                    │
                     ┌──────────────▼──────────────────────────┐
                     │      PHASE 2: NEURAL SINDY LIBRARY      │
                     │                                         │
                     │   Θ = [ MLP_id(x), MLP_id(v),          │
                     │         MLP_cos(x), MLP_cos(v),         │
                     │         MLP_sin(x), MLP_sin(v),         │
                     │         MLP_add(x,v), MLP_mul(x,v) ]    │
                     └──────────────┬──────────────────────────┘
                                    │
┌───────────────────┐    ┌──────────▼──────────────────────────┐
│   PHASE 3: DATA   │    │     PHASE 4: DISCOVERY (2 paths)    │
│                   │───▶│                                     │
│  Damped Harmonic  │    │  Path A: STLSQ sparse regression    │
│  Oscillator data  │    │    Ẋ = Θ(X) · ξ  (classical)        │
│  [t, x(t), v(t)] │    │                                     │
└───────────────────┘    │  Path B: Gumbel-Softmax Router      │
                         │    Router → GS gate → MLP (E2E)     │
                         │    τ: 5.0 → 0.05 (anneal)           │
                         └──────────────┬──────────────────────┘
                                        │
                         ┌──────────────▼──────────────────────┐
                         │    PHASE 5: SYMBOLIC DISTILLATION    │
                         │                                     │
                         │  Winning MLP → clean synthetic data  │
                         │  → PySR symbolic regression          │
                         │  → Recovered equation: f(x) = x     │
                         └─────────────────────────────────────┘
```

---

## Experimental Description

### Datasets

| Dataset                      | Type           | Source                               | Size                               | Purpose                                               |
| ---------------------------- | -------------- | ------------------------------------ | ---------------------------------- | ----------------------------------------------------- |
| Grokking training data       | Synthetic      | Random samples from target functions | 500–1000 samples per function      | Train MLPs to grok `cos`, `sin`, `+`, `×`, `identity` |
| Grokking validation data     | Synthetic      | Held-out random samples              | 200–400 samples per function       | Verify grokking (delayed generalization)              |
| Damped oscillator trajectory | Synthetic ODE  | `scipy.integrate.solve_ivp` (RK45)   | 600 time points, dt=0.05, t∈[0,30] | Test SINDy discovery on known dynamics                |
| Distillation probes          | Synthetic grid | Uniform linspace                     | 500–2500 points                    | Feed clean data through MLPs for PySR extraction      |

**Ground truth system:**

```
ẍ = -k·x - c·ẋ     (damped harmonic oscillator)
k = 1.0, c = 0.1
x(0) = 1.0, ẋ(0) = 0.0
```

Gaussian noise (σ = 0.001) is added to the state measurements to simulate real-world sensor noise.

### Machine Learning Techniques

| Technique                      | Role in Experiment                                                                                                   | Implementation                                   |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **MLP Training with Grokking** | Train neural approximations of math functions using AdamW with heavy weight decay to induce grokking                 | PyTorch, custom `GrokMLP` class                  |
| **Sparse Regression (STLSQ)**  | _Path A:_ Select which MLP basis functions are active in the governing equations                                     | Custom implementation following Brunton et al.   |
| **Ridge Regression**           | Sub-routine within STLSQ for numerically stable least-squares fitting                                                | scikit-learn                                     |
| **Gumbel-Softmax Routing**     | _Path B:_ End-to-end differentiable discrete MLP selection with temperature annealing and Straight-Through Estimator | PyTorch, custom `GumbelRouter` class             |
| **Symbolic Regression (PySR)** | Extract human-readable symbolic equations from grokked MLP input-output behavior                                     | PySR (evolutionary algorithm with Julia backend) |

### Experiments

#### Experiment 1: Grokking Verification

- **Training data:** Random samples from each target function
- **Test data:** Held-out validation set (different random samples, same distribution)
- **Evaluation metrics:**
  - Train MSE vs. Validation MSE over epochs (grokking curve shape)
  - Final validation MSE (target: < 1e-4 for trig functions, < 1e-6 for arithmetic)
  - Relative error: RMSE / output range × 100% (target: < 5%)
- **Success criterion:** Characteristic grokking curve — validation loss drops sharply well after training loss has plateaued

#### Experiment 2: Neural SINDy Discovery

- **Training data:** Damped oscillator state measurements `[x(t), v(t)]` with noise
- **Test data:** Ground-truth derivative values `[ẋ(t), v̇(t)]` (computed analytically)
- **Evaluation metrics:**
  - Reconstruction MSE: `‖Ẋ - Θξ‖²`
  - Sparsity: number of nonzero coefficients in ξ (target: 2 per equation)
  - Coefficient accuracy: `|ξ_discovered - ξ_true| / |ξ_true|` (target: < 10% relative error)
  - Trajectory MSE: forward-simulate discovered system and compare to ground truth
- **Success criterion:** SINDy selects `identity(v)` for ẋ and `identity(x)` + `identity(v)` for v̇, with coefficients close to `+1.0`, `-1.0`, and `-0.1`

#### Experiment 3: Symbolic Distillation

- **Training data:** Clean synthetic grid through each grokked MLP (no noise)
- **Test data:** Same grid (deterministic — the MLP is the oracle)
- **Evaluation metrics:**
  - R² between PySR prediction and MLP output (target: > 0.999)
  - Equation complexity (prefer parsimony)
  - Exact symbolic match to ground truth function
- **Success criterion:** PySR recovers `f(x) = x` for identity, `f(x) = cos(x)` for cosine, etc.

#### Experiment 4: Gumbel-Softmax Router Discovery

- **Training data:** 80% of damped oscillator measurements (shuffled)
- **Test data:** 20% held-out oscillator measurements
- **Evaluation metrics:**
  - Training/validation MSE convergence
  - Router gate activations (should converge to one-hot per derivative)
  - Learned coefficients vs true parameters
  - Head-to-head comparison with STLSQ (Experiment 2)
- **Success criterion:** Router selects the same MLPs as STLSQ with comparable coefficient accuracy. Temperature annealing should show smooth transition from exploration to hard selection.

---

## Project Structure

```
NeuralSINDy/
├── runner.ipynb            # Dependencies, Phase 1, Phase 2, Phase 3, Phase 4
├── distill.py                  # Phase 5: Symbolic distillation
├── visualize.py                # Generate all plots
├── checkpoints/                # (auto) saved MLP weights
├── data/                       # (auto) saved ODE data
└── plots/                      # (auto) all figures
```

---

## How to Run

```bash
# just run the runner.ipynb

# 4. Symbolic distillation via PySR (optional, ~2-5 min per MLP)
python distill.py

# 5. Generate visualizations (seconds)
python visualize.py
```

---

## Expected Results

If the experiment succeeds, `run_experiment.py` should output:

```
DISCOVERED EQUATIONS
============================================================
  ẋ = dx/dt = +1.000 · identity(v)
  v̇ = dv/dt = -1.000 · identity(x) -0.100 · identity(v)

TRUE EQUATIONS
============================================================
  ẋ = +1.0 · v
  v̇ = -1.0 · x - 0.1 · v
```

This would confirm that:

1. Grokked MLPs can serve as valid SINDy basis functions
2. Sparse regression correctly identifies which neural modules explain the dynamics
3. The framework is a viable alternative to hand-crafted symbolic libraries

---

## References

1. **SINDy:** Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). _Discovering governing equations from data by sparse identification of nonlinear dynamical systems._ PNAS, 113(15), 3932-3937.

2. **Grokking:** Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). _Grokking: Generalization beyond overfitting on small algorithmic datasets._ arXiv:2201.02177.

3. **PySR:** Cranmer, M. (2023). _Interpretable machine learning for science with PySR and SymbolicRegression.jl._ arXiv:2305.01582.

4. **PySINDy:** de Silva, B. M., et al. (2020). _PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data._ JOSS, 5(49), 2104.

5. **Mechanistic Interpretability of Grokking:** Nanda, N., et al. (2023). _Progress measures for grokking via mechanistic interpretability._ ICLR 2023.

6. **KANs:** Liu, Z., et al. (2024). _KAN: Kolmogorov-Arnold Networks._ arXiv:2404.19756.

7. **Gumbel-Softmax:** Jang, E., Gu, S., & Poole, B. (2017). _Categorical Reparameterization with Gumbel-Softmax._ ICLR 2017.

8. The main _SINDy_ paper my idea extend [Discovering governing equations from data: Sparse identification of nonlinear dynamical systems](https://arxiv.org/pdf/1509.03580)
9. Library Generation: [GROKKING: GENERALIZATION BEYOND OVERFIT-TING ON SMALL ALGORITHMIC DATASETS](https://arxiv.org/pdf/2201.02177)
10. Arithmetics: [Grokking modular arithmetic](https://arxiv.org/pdf/2301.02679)
11. Router: [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144)

---

## Citation

This is a research proof-of-concept. Use freely for academic purposes.

```
@unpublished{Aniruddha2026,
  title  = {{Neural SINDy}: Sparse System Identification with Grokked MLP Libraries},
  author = {Aniruddha Pandey},
  year   = {2026},
  note   = {Manuscript in preparation},
}
```
