# SINDy + Grokked MLPs — Step-by-Step Execution Guide

---

## Step 4: Symbolic Distillation via PySR (~2-5 min per MLP)

```bash
python distill.py
```

> [!IMPORTANT]
> This step requires PySR (and Julia). **Skip this step if you didn't install PySR** — it will print a warning and show input-output statistics instead.

**What it does:**

- Feeds clean synthetic data through each grokked MLP
- Runs PySR symbolic regression on the input-output pairs
- Because there's zero noise, PySR should easily recover the exact function

**What to look for:**

```
MLP_identity → f(x) = x         (R² ≈ 1.0)
MLP_cos      → f(x) = cos(x)    (R² ≈ 1.0)
MLP_sin      → f(x) = sin(x)    (R² ≈ 1.0)
MLP_add      → f(x,y) = x + y   (R² ≈ 1.0)
MLP_mul      → f(x,y) = x * y   (R² ≈ 1.0)
```

---

## Step 5: Generate All Visualizations (~seconds)

```bash
python visualize.py
```

**Outputs:**

- `plots/mlp_approximations.png` — each MLP vs ground truth function
- `plots/full_summary.png` — combined summary figure with discovered equations

---
