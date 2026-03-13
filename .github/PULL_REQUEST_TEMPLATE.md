## Pull Request Checklist — GDP2-WorldModel

**Branch:** `feature/member-X-description` → `main`
**Member:** <!-- Your name/role e.g. Member B — World Model Architect -->
**Task(s) completed:** <!-- e.g. B.2, B.3 -->

---

### 1. Interface Contract Compliance
> These values are IMMUTABLE. If you changed any of these, stop and discuss with Member E first.

- [ ] Latent dimension is still **384** (DINOv2 ViT-S/14)
- [ ] Sequence length `T` is still **16**
- [ ] Input image size is still **(64, 64, 3) RGB**
- [ ] Batch tensors are shaped **(Batch_Size, 16, 384)**

---

### 2. Tests Passed
> Run these locally before opening the PR. Paste the output below.

- [ ] `python -m pytest tests/test_shapes.py -v` — **PASSED**
- [ ] `python -m pytest tests/test_env.py -v` — **PASSED**
- [ ] `python -m pytest tests/test_integration.py -v` — **PASSED**

<details>
<summary>Paste test output here</summary>

```
# paste your terminal output here
```

</details>

---

### 3. Files Changed
> List the files you modified and briefly explain why.

| File | What changed |
|------|-------------|
| `src/...` | |

---

### 4. Member-Specific Checks

**Member A (Environment & Baseline):**
- [ ] `env.step()` still returns `(64, 64, 3)` uint8 RGB array
- [ ] `env.reset()` works without crash
- [ ] Data saved to `data/raw/` in correct `.npz` format

**Member B (World Model):**
- [ ] `model.forward(z_in, a_in)` returns `pred_next (B, T-1, 384)`, `pred_rew (B, T-1, 1)`, `pred_val (B, T-1, 1)`
- [ ] `model.rollout(z0, actions)` works correctly
- [ ] Action embeddings are injected (not raw integers)
- [ ] `forward()` docstring is up to date so Member E knows what to pass

**Member C (Data & Vision):**
- [ ] `encoder.encode(img)` returns vector of shape `(384,)` with no NaN
- [ ] `buffer.sample(B, seq_len=T)` returns `(B, T, 384)` float32 tensor
- [ ] Latent variance > 0.01 (not all-zeros bug)

**Member D (Validation & Theory):**
- [ ] `metrics.py` functions return correct scalar values
- [ ] Plots saved to `evaluation/` (not committed to git)
- [ ] No hardcoded paths — uses relative paths only

---

### 5. Notes for Member E (Integration Lead)
> Anything I need to know before merging? Any dependency on another member's code? Any known issues?

<!-- Write here -->

---

### 6. Pre-merge Checklist
- [ ] No `data/` files committed (check `.gitignore`)
- [ ] No `checkpoints/` files committed
- [ ] No print debug statements left in production code
- [ ] `requirements.txt` updated if new libraries were added
- [ ] Branch is up to date with `main` (no merge conflicts)
