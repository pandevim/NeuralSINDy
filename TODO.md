Deviation 1 — all MLPs still compute in the forward pass

**What I emailed to the professor**: Because all other gates are 0, the data only routes through the single winning MLP. The other 49 MLPs simply don't compute.

# BUT

```python
# Evaluate ALL MLPs on the state data
for i, (model, cols) in enumerate(zip(self.mlp_models, self.input_columns)):
    x_in = X[:, cols]
    out = model(x_in)  # (batch, 1)
    mlp_outputs.append(out)
```

Every MLP runs on every batch sample, then the hard gate multiplies the outputs by 1 or 0 after the fact. You get the gradient behavior of STE but not the compute savings. The correctness math is identical; the FLOPs are not.

Why it's done this way: different samples in a batch route to different MLPs (see Deviation 2 below), so you can't trivially skip any MLP at the batch level. Fixing this properly needs conditional computation (e.g., grouping samples by selected MLP, or sparse dispatch). For 8 MLPs this is not urgent; for a 50‑MLP library it will be.
