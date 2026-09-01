# 2026-08-30 · M03-PORT-001 — Individual portfolio and cross-reference to the EDA

| Field | Value |
| --- | --- |
| **Date** | 2026-08-30 |
| **Member** | Ilias El Hamri (Member 03) |
| **Ticket ID** | M03-PORT-001 |
| **Branch** | member_03 → Feature/Logistic_Regression/PCA |
| **Time spent** | ≈ 3 h (estimate) |
| **Basis for time** | One commit only, so the session length cannot be derived from timestamps; this figure is an estimate and should be treated as such. |
| **Related meeting** | 2026-08-28 |

## Objective

Write up the work package, and explain the null result rather than merely reporting it.

## Work performed

- Drafted the Member 03 portfolio section and committed it
  (`docs(portfolio): add Member 03 individual portfolio`).
- Re-read Member 02's EDA logbook entry (`ADA-DATA-EDA-01`) and Member 01's L1 sparsity
  audit (`reports/tables/logistic_l1_sparsity_summary.json`) to check whether the null
  result was consistent with the rest of the project's evidence.

## Interpretation

Three independent artefacts agree. Member 02 reported weak pairwise linear correlations
across the 200 features. Member 01's sparsity audit found that 196–200 of 200 coefficients
remained non-zero under an L1 penalty. My two experiments found no ranking gain from either
reduction. Together these say the same thing: the dataset contains little linear redundancy,
so there is little for PCA to compress or for an L1 selector to drop.

This reframes the result. It is not that the methods were applied badly, but that the data
does not offer what they exploit.

## Limitations recorded

- One configuration per method; neither `C` nor the variance threshold was swept.
- The number of retained components and selected features per fold was never logged, so the
  compression achieved cannot be quantified.
- The conclusion is bound to a linear classifier; no reduced representation was tested in
  front of a nonlinear model.
- PCA is unsupervised and can discard a low-variance but discriminative direction.
- The comparison is descriptive; no paired statistical test across folds was performed.

## Files changed

`reports/portfolio/member_03_ilias_el_hamri.md`

## Next step

Assemble the e-portfolio document and prepare the presentation section.
