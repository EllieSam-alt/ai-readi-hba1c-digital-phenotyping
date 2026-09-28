# AI READI reviewer completion toolkit

Open AI_READI_Reviewer_Completion.ipynb with a fresh Python kernel from this extracted folder. The notebook includes the relocated project path. Read the Markdown instructions before running each numbered section.

Contents
- AI_READI_Reviewer_Completion.ipynb: step-by-step local workflow.
- reviewer_completion.py: extraction, integrity checks and additional analyses.
- reviewed_source/: the two reviewed CGM-independent source modules. They are accepted only if their hashes match the executed training lock.

Outputs are saved under the existing run in reviewer_completion/<timestamp>/. Original models and reports are never overwritten. Participant-level outputs remain in LOCAL_ONLY. The aggregate review archive uses an explicit file allowlist.

What this can recover
- Exact selected and effective parameters of final saved learners.
- Exact encoded dimensions, including missingness indicators and learned categories.
- Agreement between current artifacts and the archived execution lock after relocation.
- Original recorded environment and current recovery environment as separate records.
- Education source candidates for local investigation.
- Descriptive age-group MAE and bias.
- Three-seed Monte Carlo attribution stability with a fixed original training background.
- Optional separately fitted no-education WDS and CWDS sensitivity models.

What still needs documentary evidence
- Exact dataset release and original acquisition date.
- Continuity from the environment recorded at run creation to actual model fitting.
- Full education derivation and verified coding; a candidate label match is insufficient.
- A real public repository and immutable version or DOI. This toolkit does not publish one.

The manuscript cannot be finalized by treating missing records as completed analyses. All revised and additional evaluations remain exploratory after earlier test exposure.

The module syntax and selected functions are checked on synthetic data in the authoring environment. The notebook has not been run against the participant files or current model bundles on your computer. Do not interpret synthetic checks as study results.

Technical documentation
- Fitted transformed output names: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- Spearman rank correlation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
- AI-READI questionnaire documentation: https://docs.aireadi.org/docs/3/dataset/clinical-data/questionnaires

The AI-READI documentation version linked above is not an assertion about the version downloaded for this study.
