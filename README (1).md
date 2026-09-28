# AI-READI: CGM-independent HbA1c reanalysis

## What this revision fixes

The old workflow selected people and wearable windows using CGM recording boundaries, CGM quality, next-day CGM outcomes, and the intersection of three-, five-, and seven-day forecast windows. Those requirements are inappropriate for the current laboratory-HbA1c question.

This workflow reads the full participant roster and original Garmin files. It does **not** read a CGM manifest, CGM recording, CGM quality mask, CGM-derived outcome, old aligned daily cache, old forecast window, or old model. All roster participants receive a preparation/status row. Every model is retrained on the newly eligible **training** participants.

**This does not guarantee better predictive accuracy.** It corrects the target population and removes a scientifically irrelevant source of selection. The old test results have already been inspected. The new validation and test analyses are explicitly labeled **post-test revision / exploratory internal evaluation**; neither is untouched or external validation. Keep the old results and disclose this change in the manuscript.

## Run in order

1. Extract this folder. Keep `AI_READI_No_CGM_Reanalysis.ipynb`, `aireadi_no_cgm.py`, and `aireadi_no_cgm_models.py` together. Open the notebook.
2. Run the configuration and source-check cell. It contains your corrected `D:\POSTDOC\YSN\AimAhead\...` paths. No participant data are transmitted.
3. Run preparation. This reads original Garmin JSON files and can take substantially longer than reading the old caches. Progress prints every 100 people. Expected missing sources are retained; malformed source schemas, unsupported units, conflicting clinical responses, and manifest paths pointing to absent files stop with an explanation.
4. Review the cohort flow, source-quality status, predictor missingness, and static predictor timing audit. A participant with missing CGM may now enter. **Do not change quality settings in response to predictive performance.** If a source mapping genuinely needs correction, create a new run rather than editing a locked run.
5. Run nested cross-validation and final training. All eight feature groups use the same new common training cohort and outer folds. The previous models and ensemble weights are not reusable.
6. Evaluate validation and then test with the new locked training-fitted pipelines. Results are exploratory because the original test was already examined.
7. Run the all-eligible-validation feature attribution cell and inspect its figures/table. You can explain the test split by an explicit separate call; there is no default 50-person sampling limit.
8. Optionally run the one additional quality sensitivity. It is labeled as a new post-test sensitivity, not an earlier prespecified/frozen analysis.

Output folders are timestamped under `AI_READI_Analysis/hba1c_no_cgm_v1/`. No previous analysis is overwritten. Fitting/evaluation/explanation destinations intentionally reject a second run into an existing folder. To inspect completed results, read their CSVs. To restart after a failure, use a new output run; the notebook includes a resume cell for completed preparation/training.

## Input sources and what is reused

| Input | Use | Why it does not reintroduce the old CGM cohort |
|---|---|---|
| `dataset/participants.tsv` | Full roster, recommended split, visit calendar date, age | Starting population is the full roster |
| `dataset/wearable_activity_monitor/manifest.tsv` and original JSONs | New scalar and sleep histories | No old CGM-aligned cache or selected window is loaded |
| `clinical_data/measurement.csv`, `observation.csv` | Clinical measures and nine diet items | Values extracted again for all roster IDs |
| `expanded_predictor_inventory/TRAIN_all_clinical_variables.csv` | Verified source-label/concept/unit mapping only | This file supplies mapping keys, not participant eligibility or feature values |
| `staging/cleaning/hba1c_sources_UNIT_CHECKED.csv` | Unit-checked laboratory outcome and timing/operator metadata | Must include every roster ID, including missing outcomes; a selected cohort is rejected |
| `staging/sdh/sdh_predictor_candidates.csv` | Previously cleaned nominal SDH responses and stored education | Must include every roster ID; missing SDH never excludes anyone |

The cleaned HbA1c and SDH tables still need their original source-cleaning provenance. Requiring full-roster membership prevents accidentally feeding an already selected participant table, but cannot prove that a supplied file's values were cleaned correctly. The code records input SHA-256 hashes.

## Exact revised eligibility and timing

- Laboratory HbA1c must be finite and positive in `hba1c_percent_unit_confirmed` (a value, **not a Boolean flag**), with `hba1c_days_from_visit == 0`, a known visit date, and the same expected operator representation as the original verified workflow. Missing outcomes are never imputed. No exclusion is based on the HbA1c magnitude beyond positive/finite availability.
- Search the interval from the study-visit calendar date through **10 elapsed days**, left inclusive/right exclusive. The earliest usable scalar wearable record or recognized, nonconflicting sleep segment in that interval determines the start. The window ends exactly seven elapsed days later. **No search for the best-quality or best-performing window.** A start on day 9 can yield an end on day 16; this is deliberate and documented, not a claim of exact coincidence with HbA1c.
- The 10-day initiation search is a new explicit replacement for the CGM anchor, not a previously validated clinical threshold. A visit calendar date is represented as midnight UTC for reproducibility. That is **not** the visit's measured clock time or local midnight. Timestamp-aware source records are converted to UTC. Naive timestamps require a documented timezone; the default stops instead of guessing.
- Common analysis cohort: visit-day laboratory HbA1c plus at least one source passing wearable quality. Missing diet/SDH and partially missing wearables do not exclude a participant. Participants without quality-passing wearables remain in the status/selection tables. This remains a wearable-eligible population, **not** an all-comers population.
- Predictors usually extend after the blood draw. The task is **retrospective estimation of recorded HbA1c**, not prospective forecasting or a causal effect of lifestyle.

## Wearable preparation

| Source | Reading validity | Daily quality | Seven-day source quality |
|---|---|---|---|
| Heart rate | Finite >0, beats/min | At least 600 occupied elapsed-minute bins | At least five passing days spanning at least five day-index units |
| Respiratory rate | Finite >0, breaths/min | At least 600 minutes | Same |
| Device stress | Finite 0–100, source `stress level` | At least 600 minutes | Same |
| Oxygen saturation | Finite >0 and ≤100, % | At least 60 minutes | Same |
| Sleep | Recognized light/deep/REM/awake intervals | Episode rules below | At least three accepted episodes, end-time span ≥5 days, last episode ends within two days of window end |

Scalar timestamps with identical duplicate values are deduplicated. If different valid values share a timestamp, all records at that conflicting timestamp are withheld and counted. Out-of-range readings are counted and withheld. Unsupported units or unexpected text trigger a review stop.

Per scalar source, use quality-passing daily means to calculate: (1) mean of daily means, (2) ordinary least-squares slope against actual elapsed day positions, (3) sample SD of daily means, and (4) final-day mean minus the prior passing-day mean when the final day passes. Each day receives equal weight in the history mean. Source-quality failure makes that source's four features missing, rather than removing the participant.

Sleep intervals are deduplicated and processed as a union of nonoverlapping segments. Conflicting stage overlaps are unresolved, not double-counted. Gaps greater than 60 minutes separate episodes; shorter gaps count toward unresolved elapsed time. An episode must be fully contained in the selected seven-day history, have positive recorded sleep, last ≤24 elapsed hours, and have ≤10% unresolved time. Features are mean/sample SD of episode sleep duration, slope of duration against episode end time, and latest episode duration minus the previous-episode mean. These are episode-duration measures, not validated nightly sleep scores.

No step count feature is added in this revision. Garmin step/activity endpoints require a separate verified extraction and scientific decision.

## Clinical, diet, and SDH preparation

Clinical measurements are matched using the original source concept, source label, and units. Distinct conflicting repeated values stop for visit/source review; no favorable measurement is chosen. Nonpositive body/BP values are set missing. Age, BMI, waist, first systolic BP, and first diastolic BP are the clinical predictors. Other anthropometry/BP values are extracted for auditing, not silently added to the models. Clinical and diet collection dates relative to the visit are saved for review.

Nine diet items remain **nominal categories**. The previously audited `777` representation becomes missing; valid 0/1/2 categories are not converted to servings or a diet score. Numeric/source-response disagreements and conflicting repeated responses stop.

Fourteen SDH fields are retained to isolate the effect of correcting CGM-based selection. In particular, **the stored education numeric value is preserved, but its semantic coding/units remain unresolved**. The new labels say “Education (stored numeric value)” rather than claiming verified years. Resolve the education codebook before substantive interpretation/publication; if its coding changes, document another model revision. Questionnaire labels are short descriptions inherited from the prior label mapping, not newly reconstructed verbatim questionnaire text. Nominal categories are not assigned an invented low-to-high order in SHAP figures.

## Models, validation, uncertainty

| Group | Predictors |
|---|---|
| W | 20 wearable features |
| WD | Wearables + nine diet items |
| WDS | Wearables + diet + 14 SDH fields |
| C | Five clinical predictors |
| CD | Clinical + diet |
| CW | Clinical + wearables |
| CWD | Clinical + wearables + diet |
| CWDS | Clinical + wearables + diet + SDH |

There is one row per participant. Recommended splits stay separate. Five outer training folds and three inner folds are shared across predictor groups. Ridge, random forest, and histogram gradient boosting are tuned by inner out-of-fold MAE. Median imputation, numeric missingness indicators, scaling, and nominal one-hot encoding are fitted **inside** each training fold. New held-out nominal categories are handled without learning from the held-out outcomes.

Nonnegative ensemble weights summing to one minimize MAE using selected learners' inner out-of-fold predictions. The outer held-out fold evaluates tuning and weight fitting together. Final learner settings and ensemble weights are derived anew from all eligible training participants through inner cross-validation, then saved with fitted preprocessing. No validation/test refitting or recalibration occurs.

Reports include MAE, RMSE, R², bias, 95% bootstrap intervals, paired predictor-group differences, learned versus equal weighting, weighted ensemble versus every individual learner, and both training-mean and training-median baselines. The designated main contrast remains WD minus W in MAE, explicitly exploratory in this revised project. Other contrasts are exploratory. Participant bootstrap uses paired resampling; OOF resampling preserves outer-fold strata. Intervals condition on saved predictions, exclude refitting/selection uncertainty, and do not eliminate dependence caused by overlapping CV training sets.

Observed/predicted and calibration figures are descriptive. Error strata based on recorded HbA1c are diagnostic summaries, not new eligibility thresholds. Selection reports cover included versus excluded participants both in the full roster and among visit-day-HbA1c-eligible people, with standardized differences and missingness rather than significance-driven selection.

## Held-out SHAP-style attribution

The package includes an explicit Monte Carlo estimator of **interventional Shapley values**, so it does not require the `shap` or `jinja2` packages. It explains the saved weighted ensemble on **all eligible participants in the chosen held-out split**, using a fixed sample of 32 training participants as the background. A random permutation path and its reverse are evaluated for every background row, repeated twice by default. The model is only called to predict; it is never fitted to the explanation set.

Individual attribution values are approximate. A telescoping additivity check verifies that the background mean plus contributions recovers each prediction, but this **does not prove Monte Carlo convergence**. For a formal attribution-focused analysis, assess rank/value stability with additional permutations, clearly separate from model-performance evaluation. Interventional hybrids can be unrealistic for correlated features. No causal or clinical-usefulness inference follows from attribution magnitude. Numeric color uses within-feature 5th–95th percentiles; nominal and missing values are gray. Figures omit participant-count subtitles; metadata retains exact counts for transparent reporting. CSV, HTML, PNG, PDF, and SVG outputs are saved.

## Additional sensitivity

The optional additional sensitivity relaxes scalar source coverage to at least four days spanning three day-index units. Minute thresholds, the seven-day window anchor, sleep rules, HbA1c rules, algorithms, and feature groups remain unchanged. Training is repeated only on the alternative training cohort. Reports include both the alternative eligible population and matched main/sensitivity held-out participants. This is new additional analysis after the original test was examined. It must not be presented as previously prespecified or used to pick the most favorable result.

## Software verification and limits

Run `python test_no_cgm.py` for synthetic tests. They check a dataset containing **no CGM files**, full-roster preservation, eligibility independent of CGM, HbA1c-independent wearable windows, source conflicts, sleep overlap accounting, parsing, train/evaluation separation, locked artifacts, and the complete training/evaluation/attribution workflow. Tests also verify exact attributions for an additive toy predictor. Small test forests/folds are only used with `software_test_only=True`; the research notebook uses the full settings.

This package was tested with synthetic data in the assistant's environment. It has **not** been executed against your Windows raw AI-READI files. New real cohort sizes, performance estimates, and feature rankings are unknown until you run it. Raw records and output files containing participant information must remain in your approved local environment. Files marked `LOCAL_ONLY` are not publication tables.

## Sources and implementation provenance

- Your recovered original analysis code, originally uploaded as `Pasted text(20260924-140438).txt`: source schema, units, source-code mappings, original thresholds, sleep logic, nominal diet handling.
- Your original `aireadi_lifestyle_diet_ensemble.py`: preprocessing, candidate models, exact MAE weight optimization, and nested training structure. This revision replaces cohort/window construction and extends reporting and provenance checks.
- AI-READI wearable documentation: https://docs.aireadi.org/docs/3/dataset/wearable-activity-monitor/
- AI-READI dataset documentation: https://docs.aireadi.org/docs/3/
- Scikit-learn imputation: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- Scikit-learn categorical encoding: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

The quality thresholds are analysis choices inherited from your prior code, not clinical recommendations from these documentation pages.
