# GitHub upload checklist

## Before public release

1. Confirm the final paper title and author order.
2. Complete `CITATION.cff.template` and rename it to `CITATION.cff`.
3. Select and add a software `LICENSE` approved by all authors.
4. Replace any placeholder repository URL, journal, and DOI fields.
5. Do not add raw datasets, Parquet partitions, model binaries, or Google Drive credentials.

## Upload through the GitHub website

1. Sign in to GitHub and choose **New repository**.
2. Suggested repository name: `semantic-feature-group-mask-nids`.
3. Add a short description based on the first paragraph of `README.md`.
4. Select **Private** while the manuscript is under review unless the target journal requires a public repository.
5. Do not ask GitHub to create another README, `.gitignore`, or license during repository creation because this package already contains the first two and the license choice is pending.
6. Open the empty repository and choose **uploading an existing file**.
7. Extract the supplied ZIP locally, open the extracted repository folder, and upload its contents while preserving the folder structure.
8. Use the commit message: `Initial reproducibility package`.
9. Open the repository after committing and confirm that the README renders and the `figures/`, `scripts/`, `results/`, `config/`, and `protocols/` folders are visible.

## Final verification

Run locally or in Colab from the repository root:

```bash
python scripts/aggregate_analysis.py
python scripts/generate_figures.py
python scripts/verify_repository.py --require-figures --require-tables
```

The required final status is `PASS` with an empty `problems` list.
