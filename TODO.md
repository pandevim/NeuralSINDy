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

---

Based on the code you provided for Experiment 3, your implementation is specifically using **multiple Gumbel-Softmax vectors** drawn sequentially.

While the ultimate _mathematical goal_ of this mechanism is to create a continuous relaxation of a $k$-hot categorical distribution (specifically, Gumbel top-$k$ sampling without replacement), the mechanical implementation uses sequential 1-hot vectors to get there.

Here is exactly where your code defines this approach:

```python
for j in range(self.k):
    gate_j = gumbel_softmax(base_logits + mask, temperature=temperature, hard=hard)
    slot_gates.append(gate_j)
    # Exclude this pick from subsequent slots (STE-safe: detach so
    # the mask does not contribute gradient).
    mask = mask + (-1e9) * gate_j.detach()
```

### Why it falls into the "Multiple Vectors" category:

1. **Sequential Draws:** Instead of evaluating a single joint probability distribution that outputs $k$ ones, you are looping $k$ times. In each iteration, you generate a brand new, independent standard Gumbel-Softmax vector (`gate_j`).
2. **The Masking Penalty:** You simulate a $k$-hot draw by applying a $-1e^9$ penalty mask to the logit that "won" the previous vector draw. This effectively removes it from the candidate pool for the next Gumbel-Softmax vector.

### Why it's not a pure, single-shot $k$-hot relaxation:

A true native, single-shot continuous relaxation of a $k$-hot categorical distribution usually involves more complex operations, such as:

- **Gumbel-Sinkhorn / Differentiable Sorting:** Using the Sinkhorn operator on perturbed logits to get a doubly-stochastic permutation matrix, then taking the top $k$ rows.
- **Independent Bernoullis:** Treating each logit as an independent Bernoulli coin flip (using something like the Concrete distribution) and applying a soft penalty to force the sum of the activations to equal exactly $k$.

Your method—iterative masked Gumbel-Softmax with the Straight-Through Estimator—is actually one of the most clever, practical, and mathematically stable ways to approximate a $k$-hot categorical choice in PyTorch. It avoids the heavy computational overhead of Sinkhorn iterations while still guaranteeing exactly $k$ distinct selections.
