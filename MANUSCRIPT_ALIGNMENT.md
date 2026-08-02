# Manuscript-to-repository alignment audit

Audit date: 2026-08-02

## Outcome

The committed aggregate results reconstruct the numerical values reported in manuscript Tables 6–10. The five repository figures use the same model variants, condition definitions, aggregation order and displayed metrics as manuscript Figures 1–5.

## Figure alignment

| Manuscript figure | Repository output | Reconstruction source |
|---|---|---|
| Figure 1: source-only workflow | `figures/Figure_1.pdf` and `.png` | Frozen protocols and study design |
| Figure 2: condition-level F1 | `figures/Figure_2.pdf` and `.png` | Corrected ablation non-selective table |
| Figure 3: component ablation | `figures/Figure_3.pdf` and `.png` | Corrected ablation non-selective table |
| Figure 4: augmentation-only F1 changes | `figures/Figure_4.pdf` and `.png` | Corrected ablation non-selective table |
| Figure 5: selective prediction | `figures/Figure_5.pdf` and `.png` | Corrected ablation selective table |

## Table alignment

| Manuscript table | Reconstructed file | Key checks |
|---|---|---|
| Table 6 | `manuscript_tables/Table_6.csv` | Complete, single-loss and pairwise F1 for all four ablation models |
| Table 7 | `manuscript_tables/Table_7.csv` | Seed-paired effects, 95% t intervals, paired t tests, exact Wilcoxon tests and wins |
| Table 8 | `manuscript_tables/Table_8.csv` | All six condition-level baseline, augmentation and delta values |
| Table 9 | `manuscript_tables/Table_9.csv` | Nominal 90% selective coverage, attack coverage, risk, accepted F1 and AURC |
| Table 10 | `manuscript_tables/Table_10.csv` | Five-seed XGBoost and Random Forest pairwise comparison |

The repository verifier performs exact row-count, seed, condition, model, protocol-status and SHA-256 checks. It also checks representative manuscript values, including both unfavorable target reversals and the Random Forest target-F1 decrease.

## Provenance alignment

- Corrected component-ablation manifest audit: PASS.
- Component models audited: 10/10.
- Missing component files: 0.
- Component parameter mismatches: 0.
- Random Forest pretraining audit: PASS.
- Frozen authoritative Random Forest artifacts: 7/7.
- Target use: final evaluation only.

## Items requiring author action

These are publication-administration items, not experimental mismatches:

1. Create the final GitHub repository and replace the manuscript's `[REPOSITORY URL]` placeholders.
2. Remove the duplicated “Writing—review and editing” role from the CRediT statement.
3. Select a software license approved by all authors.
4. Complete and rename `CITATION.cff.template` after the final repository URL and publication metadata are known.
