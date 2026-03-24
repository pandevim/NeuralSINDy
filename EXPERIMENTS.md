# Experiment Log

## Experiment 1: STLSQ Discovery (Run 1 — Baseline)

**Date:** 2026-03-23  
**Script:** `run_experiment.py`  
**System:** Damped harmonic oscillator — `ẍ = -1.0·x - 0.1·ẋ`

### Configuration
- Library: 8 terms — `[identity(x), identity(v), cos(x), cos(v), sin(x), sin(v), add(x,v), mul(x,v)]`
- Data: 600 time points, dt=0.05, noise σ=0.001
- Optimizer: STLSQ with ridge α=0.01
- Best threshold: 0.2

### Discovered Equations

```
ẋ = dx/dt = -0.3325·identity(x) + 0.6675·identity(v) + 0.3324·add(x,v)
v̇ = dv/dt = -0.6322·identity(x) + 0.2677·identity(v) - 0.3676·add(x,v)
```

### True Equations
```
ẋ = +1.0·v
v̇ = -1.0·x - 0.1·v
```

### Analysis

**Result: ✅ Mathematically correct, but coefficients are spread across redundant terms.**

Since `add(x,v) ≈ x + v`, substituting and simplifying:

| Equation | Discovered (simplified) | True | Match? |
|----------|------------------------|------|--------|
| ẋ | (-0.3325 + 0.3324)·x + (0.6675 + 0.3324)·v = **-0.0001·x + 0.9999·v** | +1.0·v | ✅ |
| v̇ | (-0.6322 - 0.3676)·x + (0.2677 - 0.3676)·v = **-0.9998·x - 0.0999·v** | -1.0·x - 0.1·v | ✅ |

**MSE:** 0.000001 (essentially exact fit)  
**Active terms:** 6/16

### Key Finding: Collinearity Problem

The `add(x,v)` MLP is mathematically identical to `identity(x) + identity(v)`, creating a rank-deficient library matrix. STLSQ cannot distinguish between them, so it distributes the coefficients arbitrarily across the three redundant terms. The *combined* answer is correct, but the individual coefficients are uninterpretable.

**Implication:** The library design must avoid including terms that are linear combinations of other terms. For the next run, removing `add` from the library (or adding a collinearity detection step) should produce clean, interpretable coefficients.

---

## Experiment 2: *(pending — remove `add` from library)*

---

## Experiment 3: *(pending — Gumbel-Softmax router)*
