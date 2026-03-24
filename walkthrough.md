# SINDy + Grokked MLPs — Step-by-Step Execution Guide

## Project Structure
```
MLPLibexp1/
├── requirements.txt            # Dependencies
├── grok_mlps.py                # Phase 1: Train grokked MLPs
├── generate_data.py            # Phase 3: Generate ODE data
├── neural_sindy.py             # Phase 2: Neural SINDy library (imported)
├── neural_router.py            # Gumbel-Softmax router (imported)
├── run_experiment.py           # Phase 4A: STLSQ discovery
├── run_router_experiment.py    # Phase 4B: Gumbel-Softmax discovery
├── distill.py                  # Phase 5: Symbolic distillation
├── visualize.py                # Generate all plots
├── checkpoints/                # (auto) saved MLP weights
├── data/                       # (auto) saved ODE data
└── plots/                      # (auto) all figures
```

---

## Step 0: Install Dependencies

```bash
cd /home/aniruddha/Personal/repos/MLPLibexp1
pip install torch numpy matplotlib scipy scikit-learn pysr
```

> [!NOTE]
> **PySR** requires Julia. On first run it will auto-install Julia — this takes a few minutes. If you want to skip PySR for now, everything else works without it (distill.py will print a warning and show statistics instead of equations).

---

## Step 1: Train the Grokked MLPs (~5-10 min GPU, ~30-60 min CPU)

```bash
python grok_mlps.py
```

**What it does:**
- Trains 5 small MLPs to learn: `identity(x)`, `cos(x)`, `sin(x)`, [add(x,y)](file:///home/aniruddha/Personal/repos/MLPLibexp1/neural_sindy.py#71-75), [mul(x,y)](file:///home/aniruddha/Personal/repos/MLPLibexp1/run_experiment.py#22-38)
- Uses AdamW with **heavy weight decay** (the key ingredient for grokking)
- Trains for 10k-20k epochs each

**What to look for:**
- Training loss drops fast, validation loss drops later → **that's grokking!**
- Final relative error should be <5% for each MLP

**Outputs:**
- `checkpoints/mlp_*.pt` — saved model weights
- `plots/grokking_curves.png` — train vs val loss curves

---

## Step 2: Generate Synthetic ODE Data (~instant)

```bash
python generate_data.py
```

**What it does:**
- Simulates a damped harmonic oscillator: `ẍ = -1.0·x - 0.1·ẋ`
- 600 time points, small Gaussian noise added

**Outputs:**
- `data/oscillator_data.npz` — time series `[t, x, v, dx/dt, dv/dt]`
- `plots/oscillator_data.png` — phase portrait and time series

---

## Step 3A: Run STLSQ Discovery — Path A (~seconds)

```bash
python run_experiment.py
```

**What it does:**
- Loads grokked MLPs and ODE data
- Evaluates each MLP on the state data to build library matrix **Θ**
- Runs STLSQ sparse regression to find coefficients **ξ**
- Tries multiple sparsity thresholds, picks the sparsest model with low error
- Simulates the discovered system and compares to ground truth

**What to look for (the money shot):**
```
DISCOVERED EQUATIONS
============================================================
  ẋ = dx/dt = +1.000 · identity(v)          ← should pick identity(v) ≈ v
  v̇ = dv/dt = -1.000 · identity(x) -0.100 · identity(v)  ← should pick identity(x) and identity(v)
```
If SINDy picks `identity(x)` and `identity(v)` with coefficients near `-1.0` and `-0.1`, **the experiment worked!**

**Outputs:**
- `data/discovery_results.npz` — coefficients and selected terms
- `plots/discovery_comparison.png` — true vs discovered trajectories
- `plots/coefficients.png` — coefficient bar chart (should be very sparse)

---

## Step 3B: Run Gumbel-Softmax Discovery — Path B (~seconds)

```bash
python run_router_experiment.py
```

**What it does:**
- Builds a router network that learns to select MLPs using Gumbel-Softmax
- Temperature anneals from τ=5.0 (soft, exploratory) → τ=0.05 (hard, discrete)
- Uses Straight-Through Estimator: forward pass routes through 1 MLP, backward pass gets smooth gradients
- Compares results head-to-head with STLSQ (Step 3A)

**What to look for:**
- Training loss should decrease smoothly as temperature anneals
- Final routing should converge to one-hot (one MLP per derivative)
- Should select the same MLPs as STLSQ with similar coefficients

**Outputs:**
- `plots/router_results.png` — training curves, temperature schedule, routing selections

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

## Quick Reference (all commands)

```bash
cd /home/aniruddha/Personal/repos/MLPLibexp1
pip install torch numpy matplotlib scipy scikit-learn pysr   # Step 0
python grok_mlps.py                                          # Step 1 (slow)
python generate_data.py                                      # Step 2 (fast)
python run_experiment.py                                     # Step 3A: STLSQ (fast)
python run_router_experiment.py                              # Step 3B: Gumbel (fast)
python distill.py                                            # Step 4 (optional)
python visualize.py                                          # Step 5 (fast)
```
