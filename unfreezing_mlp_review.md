Yes, you can absolutely allow the weights of your grokked MLPs to fine-tune during the routing stage. In the literature on dictionary learning and neural-symbolic systems, this transitions your framework from a **Fixed Basis Dictionary** to an **Adaptive Basis Dictionary**.

To do this safely without suffering from Identity Collapse or Overfitting to Noise, you must alter your loss function to implement a set of **Functional and Structural Regularization Penalties**.

During the routing phase, your total loss function would look like this:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dynamics}} + \lambda_1 \mathcal{L}_{\text{functional\_anchor}} + \lambda_2 \mathcal{L}_{\text{basis\_diversity}} + \lambda_3 \mathcal{L}_{\text{smoothness}}$$

Here is exactly how you can structure these penalties mathematically to protect your library's integrity:

---

### 1. Stopping Identity Collapse

When you unfreeze the MLPs, they will naturally want to deform to help the immediate path optimization, even if it means `MLP_cos` mutating to look linear. You can stop this using two cooperative penalties:

#### A. Functional Anchor Loss (The Identity Tether)

Instead of forcing the _weights_ to be rigid, you allow the weights to move, but you penalize the MLP if its global _input-output mapping_ deviates too far from its mathematical target.

- **Implementation:** Generate a permanent, clean reference grid of inputs $X_{\text{ref}}$ across your global domain (completely free of sensor noise).
- **The Penalty:** Calculate the Mean Squared Error between the unfrozen MLP's current output on this clean grid and its original pure mathematical definition $f_j$ (e.g., the exact $\cos(x)$ value):

$$\mathcal{L}_{\text{functional\_anchor}} = \sum_{j \in \text{Library}} \mathbb{E}_{x \sim X_{\text{ref}}} \left[ \| \text{MLP}_j(x) - f_j(x) \|^2 \right]$$

- **Why it works:** This creates a soft gravitational pull in function space. The MLP can slightly adjust its internal scaling, amplitude, or coordinate offsets to match your real-world observations better, but if it attempts to completely alter its functional shape, this penalty spikes drastically.

#### B. Activation Diversity Penalty (Cross-Correlation Minimization)

To prevent different MLPs from collapsing onto the same simple representation (e.g., multiple primitives morphing into linear paths), you can punish redundant behaviors across your dictionary.

- **The Penalty:** Minimize the absolute cosine similarity between the activation vectors of different dictionary entries over the current batch $X$:

$$\mathcal{L}_{\text{basis\_diversity}} = \sum_{j \neq k} \left| \frac{\langle \text{MLP}_j(X), \text{MLP}_k(X) \rangle}{\|\text{MLP}_j(X)\| \|\text{MLP}_k(X)\|} \right|$$

- **Why it works:** It acts as a repulsive force in function space, actively pushing your basis functions away from each other and forcing the dictionary to maintain diverse, linearly independent identities.

---

### 2. Stopping Overfitting to Noise

Unfrozen MLPs will try to use their high-dimensional capacity to wriggle around your $0.001$ Gaussian sensor noise to lower the immediate training error. You can paralyze their ability to fit high-frequency jitter using smoothness constraints.

#### A. Jacobian/Sobolev Regularization (The Smoothness Filter)

You want to guarantee that whatever adjustments the MLPs make, they are restricted to smooth, low-frequency trends.

- **The Penalty:** Penalize the Frobenius norm of the gradients of the MLP outputs with respect to their inputs:

$$\mathcal{L}_{\text{smoothness}} = \sum_{j \in \text{Library}} \mathbb{E}_{x} \left[ \| \nabla_x \text{MLP}_j(x) \|_F^2 \right]$$

- **Why it works:** This mathematically bounds the first derivative (the Lipschitz constant) of your neural primitives. It acts as a low-pass filter, meaning the MLP physically cannot form the sharp, jagged shapes required to track high-frequency sensor noise.

#### B. Weight-Space Elastic Constraints (EWC-style)

If functional regularization feels too computationally expensive to compute over reference grids during every forward pass, you can fallback on a structural parameter anchor.

- **The Penalty:** Penalize the L2 distance between the current weights $W_j$ and the original grokked weights $W_{j, \text{grok}}$ obtained at the end of your standalone Phase 1 training:

$$\mathcal{L}_{\text{weight\_anchor}} = \sum_{j} \| W_j - W_{j, \text{grok}} \|_2^2$$

- **Why it works:** It treats the pre-trained grokked parameters as an elastic spring. The weights can stretch slightly to accommodate coordinate distortions, but they cannot migrate to a completely new area of the weight landscape.

---

### The Engineering Risk: Co-Adaptation Instability

If you choose to implement this, you must keep an eye out for **co-adaptation instability**.

When both the **router parameters** and the **MLP internal weights** are learning at the same time, the optimization target becomes a moving horizon. The router might start selecting a primitive _because_ that primitive is actively warping its shape to fit the current batch, leading to a messy training trajectory.

**Recommended Solution:** Instead of unfreezing them completely simultaneously, utilize an **Alternating Optimization Schedule**:

1. **Step A (Epochs 1-5):** Freeze the MLPs. Let the Gumbel-Softmax router gradients update the selection logits and sparse coefficients exclusively.
2. **Step B (Epochs 6-7):** Freeze the router selections. Unfreeze the MLPs and turn on your functional anchors to let the active operations tune their shapes to the underlying coordinate systems.
3. **Repeat.** This guarantees that the dictionary entries are only adapting to systematic data behaviors rather than chasing the router's momentary exploratory selections.
