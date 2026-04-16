# PLAM Uncertainty Patch Design: Global Uncertainty and Residual-Aware Variance

## Purpose

This note specifies mathematically grounded and implementation-ready options to extend PLAM uncertainty in `gtex_gp` beyond the current per-latent independent GP variance. It focuses on two goals:

1. Derive a **single parcel-level uncertainty** ("global uncertainty") suitable for shrinkage, ranking, and reporting.
2. Add a principled mechanism for **residual-informed uncertainty** (capturing "higher residual => higher variance") without leakage in LORO.

Primary code context:
- `src/models/unified_generative.py`
- `src/workflows/loro_cache.py`

---

## Current Implementation (Baseline)

In `infer_subject_unified(...)`, PLAM estimates a latent residual field:

$$
\Delta_{\text{obs}} = Z_{\text{obs}} R^\top - U_{\text{obs}}.
$$

For each latent component $j \in \{1,\dots,k\}$, an independent GP is fit on
$(\mathbf{c}_{\text{obs}}, \Delta_{\text{obs},j})$, where $\mathbf{c}$ denotes 3D parcel coordinates.
Predictive standard deviations $s_j(p)$ at parcel $p$ are converted to variance:

$$
\nu_j(p) = s_j(p)^2.
$$

Stacking over components gives

$$
U_{\text{var}} \in \mathbb{R}^{P \times k}, \quad U_{\text{var}}[p,j] = \nu_j(p).
$$

So current uncertainty is:
- Shape: $(P,k)$
- Type: GP **predictive variance** per latent component
- Structure: diagonal-only across latent dimensions (no cross-latent covariance)

---

## Notation

- $p \in \{1,\dots,P\}$: parcel index
- $k$: latent rank
- $g$: number of genes
- $W \in \mathbb{R}^{g \times k}$: decoder loading matrix
- $\Sigma_z(p) \in \mathbb{R}^{k \times k}$: latent predictive covariance at parcel $p$
- $\Sigma_x(p) \in \mathbb{R}^{g \times g}$: induced expression-space predictive covariance
- $\nu_j(p)$: latent variance for component $j$ at parcel $p$

Current model implies

$$
\Sigma_z(p) = \operatorname{diag}(\nu_1(p),\dots,\nu_k(p)).
$$

Expression-space propagation:

$$
\Sigma_x(p) = W\,\Sigma_z(p)\,W^\top.
$$

---

## A. Global Parcel Uncertainty from Latent Variance

### A1. Diagonal-Latent Projected Trace (recommended first patch)

Define a scalar parcel uncertainty as total decoder-projected variance:

$$
U_{\text{global}}(p) = \operatorname{tr}\!\left(\Sigma_x(p)\right)
= \operatorname{tr}\!\left(W\,\operatorname{diag}(\nu(p))\,W^\top\right).
$$

Efficient equivalent form:

$$
U_{\text{global}}(p) = \sum_{j=1}^{k} \nu_j(p)\,\lVert W_{:,j} \rVert_2^2.
$$

Interpretation: each latent variance is weighted by how much that latent contributes to decoded gene space.

Why this is better than latent mean:
- Uses decoder geometry $W$.
- Gives one scalar per parcel with principled gene-space meaning.
- Cost is negligible relative to GP fitting.

#### Supplemental pseudocode

```text
function compute_global_uncertainty_trace_diag(uvar_full[P,k], W[g,k]):
    w2[j] = ||W[:, j]||_2^2  for j=1..k
    for p in 1..P:
        u_global[p] = sum_j uvar_full[p,j] * w2[j]
    return u_global
```

### A2. Alternative scalarizations

- Latent mean:
  $$U_{\text{mean}}(p) = \frac{1}{k}\sum_j \nu_j(p).$$
- Latent max:
  $$U_{\max}(p) = \max_j \nu_j(p).$$
- Log-volume proxy:
  $$U_{\log\det}(p) = \sum_j \log(\nu_j(p)+\epsilon).$$

Recommendation: use $U_{\text{global}}(p)=\operatorname{tr}(\Sigma_x(p))$ as default.

---

## B. Joint Latent Covariance (Off-Diagonals)

### B1. Full multi-output GP (future/heavier)

Instead of independent scalar GPs per component, model

$$
\Delta(p) \in \mathbb{R}^k
$$

jointly with a multi-output GP to estimate full

$$
\Sigma_z(p) \text{ with off-diagonal terms } \Sigma_z^{ij}(p),\; i\neq j.
$$

Then

$$
\Sigma_x(p) = W\Sigma_z(p)W^\top,
\qquad
U_{\text{global}}(p)=\operatorname{tr}(\Sigma_x(p)).
$$

Tradeoff:
- Better coupling fidelity.
- Much higher engineering/runtime complexity; less stable in tiny folds.

---

## C. Residual-Aware Uncertainty (Aleatoric + Epistemic)

Goal: uncertainty should rise where fit quality is poor, not only where spatial support is weak.

### C1. Variance decomposition

$$
U_{\text{total}}(p) = U_{\text{epi}}(p) + \lambda_{\text{ale}}\,U_{\text{ale}}(p).
$$

- $U_{\text{epi}}(p)$: epistemic component from latent GP predictive variance (e.g., projected trace)
- $U_{\text{ale}}(p)$: aleatoric component from residual noise model
- $\lambda_{\text{ale}}$: weight hyperparameter

### C2. Residual GP on log-variance (minimal principled patch)

For each fold, using training observations only:

1. Compute residual energy at observed parcel $i$:

$$
r_i = \frac{1}{g}\sum_{m=1}^{g}\left(x_{i,m}^{\text{obs}} - x_{i,m}^{\text{pred}}\right)^2.
$$

2. Fit scalar GP to

$$
y_i = \log(r_i + \epsilon)
$$

as a function of coordinates.

3. Predict at parcel $p$: mean/variance in log-space

$$
m_r(p),\; s_r^2(p).
$$

4. Convert to aleatoric variance via lognormal moment:

$$
U_{\text{ale}}(p) = \exp\!\big(m_r(p) + \tfrac{1}{2}s_r^2(p)\big).
$$

5. Combine with epistemic term via $U_{\text{total}}$ above.

#### Supplemental pseudocode

```text
function residual_aware_uncertainty(train_coords, x_obs_train, x_pred_train,
                                    epi_uncertainty, lambda_ale, eps):
    r[i] = mean_over_genes( (x_obs_train[i]-x_pred_train[i])^2 )
    y[i] = log(r[i] + eps)
    fit GP_logvar on (train_coords, y)
    for each parcel p:
        m[p], s[p] = GP_logvar.predict(coord[p], return_std=True)
        u_ale[p] = exp(m[p] + 0.5 * s[p]^2)
        u_total[p] = epi_uncertainty[p] + lambda_ale * u_ale[p]
    return u_total, u_ale
```

### C3. Leakage constraint (critical)

In LORO, residual model fitting must use fold-train data only. Held-out parcel residuals must never be used to fit uncertainty for that fold.

---

## D. Runtime and Scaling Tradeoffs

Let average train-observed parcels per fold be $n$, latent rank be $k$.

- Current: roughly $k$ GP fits/fold.
- Add projected-trace scalar (A1): $\mathcal{O}(Pk)$ arithmetic, negligible.
- Add residual logvar GP (C2): +1 GP fit/fold (moderate overhead).
- Add multi-output latent GP (B1): major overhead and complexity increase.

Practical expectation:
- A1: near-zero runtime increase.
- C2: meaningful but manageable increase (typically sub-2x for GP stage, data-dependent).
- B1: potentially several-fold slower.

---

## E. Proposed Config Additions

In `UnifiedGenerativeConfig`:

- `uncertainty_scalar_mode: str = "trace_projected_diag"`
  - `mean_latent | max_latent | trace_projected_diag`
- `uncertainty_include_residual: bool = False`
- `residual_uncertainty_mode: str = "logvar_gp"`
  - `none | logvar_gp | local_smoother`
- `lambda_aleatoric: float = 1.0`
- `residual_gp_length_scale: float | None = None`
- `residual_gp_noise: float | None = None`

Optional cache additions (`loro_cache.py`):
- `plam_u_global_full` shape $(P,)$
- `plam_u_epi_full` shape $(P,)$
- `plam_u_ale_full` shape $(P,)$ when enabled
- `plam_uvar_latent_full` shape $(P,k)$ behind debug/size flag

---

## F. Evaluation Plan and Expected Outcomes

### F1. Reliability metrics

At held-out folds/parcels, evaluate:
- Correlation between uncertainty and absolute error:
  $$\operatorname{corr}(U(\text{hold}), |e_{\text{hold}}|).$$
- Binned reliability slope/intercept.
- Optional Gaussian NLL proxy when residual scale is modeled.

Expected:
- A1 should improve interpretability/consistency over latent mean.
- C2 should improve monotonic error-uncertainty coupling where residual heterogeneity exists.

### F2. Predictive impact when shrinkage uses new scalar uncertainty

Compare against current PLAM across coverage tiers:
- Pearson $r$, RMSE, $R^2$.
- Check low-coverage gains vs potential high-coverage over-shrinkage.

### F3. Runtime acceptance targets

- A1: <3% total runtime delta expected.
- C2: keep fold-level runtime increase within acceptable experimental budget.

---

## G. Risks and Mitigations

1. Residual noise overfit in tiny folds.
- Mitigate with noise floor, priors, clipping log-targets, and fallback to constant aleatoric term for small $n$.

2. Leakage in residual uncertainty.
- Strict fold-train-only residual fitting.

3. Scale mismatch between epistemic and aleatoric terms.
- Normalize components before combination; tune $\lambda_{\text{ale}}$.

4. Cache bloat.
- Store scalar uncertainty by default; gate full latent uncertainty matrix behind debug flag.

---

## H. Recommended Phased Rollout

1. **Phase 1 (low risk, high value)**
- Implement projected-trace scalar uncertainty (A1).
- Route shrinkage to this scalar.
- Export diagnostics for uncertainty-error calibration.

2. **Phase 2 (experimental)**
- Add residual log-variance GP (C2) under flags.
- Benchmark on subset runs before cluster-wide rollout.

3. **Phase 3 (if justified)**
- Explore multi-output latent GP for full joint covariance (B1).

---

## I. Concrete Code Touchpoints

- `src/models/unified_generative.py`
  - add scalar uncertainty computation from `uvar_full` and `W`
  - add optional residual uncertainty model
  - return `u_global_full`, `u_epi_full`, `u_ale_full`

- `src/workflows/loro_cache.py`
  - optionally persist uncertainty arrays in `.npz`
  - expose fold-level diagnostics for uncertainty calibration

- `src/eval/diagnostics.py` or `src/eval_utils/results_eda.py`
  - add reliability/calibration plots using new scalar uncertainty

- `scripts/run_loro_cache_batch.py` and sbatch wrappers
  - pass through new uncertainty config flags when enabled

---

## Decision Summary

If we want a principled single uncertainty value per parcel with minimal disruption, implement

$$
U_{\text{global}}(p)=\operatorname{tr}\!\left(W\operatorname{diag}(\nu(p))W^\top\right)
$$

first.

If we also want uncertainty to reflect local fit quality ("higher residual => higher variance"), add residual-aware aleatoric modeling:

$$
U_{\text{total}}(p)=U_{\text{epi}}(p)+\lambda_{\text{ale}}U_{\text{ale}}(p)
$$

behind an experimental flag and validate calibration gains under leakage-safe LORO.
