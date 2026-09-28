# Software verification — 27 September 2026

Synthetic data only. No real AI-READI clinical results were estimated in this verification.

`python test_no_cgm.py`: **7 tests passed** (59 seconds in the execution environment).

1. Full-roster preparation with no CGM files; absence of CGM does not prevent inclusion. Source-specific oxygen and sleep features are exercised; nominal diet special codes become missing.
2. Changing laboratory HbA1c values leaves wearable features and selected start times unchanged.
3. Actual missing values and explicit missing tokens are preserved; unexpected text, duplicate participant IDs, naive timestamps without a documented timezone, and unavailable referenced Garmin files raise errors.
4. Conflicting scalar timestamps are withheld; identical duplicates do not multiply recordings. Sleep overlaps are counted once, and conflicting stages count as unresolved time.
5. The Monte Carlo attribution implementation returns the exact known Shapley contributions for an additive toy predictor and satisfies additivity.
6. End-to-end nested training, final model fitting, artifact locking, evaluation on separate splits, all-eligible validation attribution, and preparation resumption succeed. Evaluation/explanation do not change model file hashes. Edited evaluation predictors fail verification.
7. The additional quality sensitivity runs separately, keeps the same window boundaries, and saves matched-participant comparisons.

Notebook code cells were parsed and checked for accidental control characters in Windows paths. The synthetic cohort-flow and SHAP figures were visually inspected for readable labels and clipping. The package contains no synthetic figures that could be confused with real results.

Verified environment: Python runtime with numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8. Each real run writes its own package versions and input/code/model hashes.

Limitations: these tests establish software behavior on the schemas represented by your recovered source code and synthetic cases. They do not verify every raw file in your Windows dataset, establish the clinical validity of quality thresholds, validate education response semantics, or establish Monte Carlo attribution convergence on real models. Source audits and the revised real cohort flow still require review after local execution.
