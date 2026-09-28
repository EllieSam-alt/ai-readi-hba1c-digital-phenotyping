"""Notebook-equivalent sequential execution; edit CODE_DIR/CFG paths as needed."""
if __name__ == "__main__":
    # Notebook code cell 2
    from pathlib import Path
    import sys
    import json
    
    CODE_DIR = Path.cwd()
    if not (CODE_DIR / "aireadi_no_cgm.py").is_file():
        CODE_DIR = Path(r"D:\POSTDOC\YSN\AimAhead\AI_READI_No_CGM_Reanalysis")
    for name in ["aireadi_no_cgm.py", "aireadi_no_cgm_models.py"]:
        if not (CODE_DIR / name).is_file():
            raise FileNotFoundError(f"Set CODE_DIR to the extracted folder containing {name}. Current choice: {CODE_DIR}")
    sys.path.insert(0, str(CODE_DIR))
    
    import pandas as pd
    from IPython.display import display, Image
    from aireadi_no_cgm import Config, new_run, source_paths, prepare
    from aireadi_no_cgm_models import (
        selection_reports, train_revision, evaluate_revision, explain_revision,
        load_prepared, verify_lock, additional_quality_sensitivity,
    )
    
    CFG = Config(
        dataset=r"D:\POSTDOC\YSN\AimAhead\DATA\aireadi\11a60cdd-6588-4f77-80d8-4e2718613172\dataset",
        analysis_base=r"D:\POSTDOC\YSN\AimAhead\AI_READI_Analysis",
        history_days=7,
        start_search_days=10,
        min_observed_minutes_per_day=600,
        oxygen_min_observed_minutes_per_day=60,
        min_good_days=5,
        min_day_span=5,
        outer_folds=5,
        inner_folds=3,
        bootstrap_draws=2000,
        software_test_only=False,
    )
    checks = pd.DataFrame([
        {"input": name, "exists": path.is_file(), "path": str(path)}
        for name, path in source_paths(CFG).items()
    ])
    display(checks)
    if not checks["exists"].all():
        raise FileNotFoundError("Restore or correct the missing source paths above before continuing.")
    print("No CGM files, old CGM-aligned caches, old selected windows, or old frozen models are required.")
    # Notebook code cell 4
    RUN_DIR = new_run(CFG)
    print("NEW ANALYSIS FOLDER:", RUN_DIR)
    print("Original analysis files remain in their original folders.")
    print("Analysis status:", CFG.analysis_status)
    # Notebook code cell 6
    analysis_status = prepare(CFG, RUN_DIR)
    print("Preparation completed:", RUN_DIR / "preparation_complete.json")
    # Notebook code cell 8
    selection = selection_reports(analysis_status, RUN_DIR)
    display(pd.read_csv(RUN_DIR / "cohort_flow.csv"))
    display(pd.read_csv(RUN_DIR / "exclusion_reasons.csv"))
    missingness = pd.read_csv(RUN_DIR / "selection_and_dictionary" / "missingness_by_selection_and_split.csv")
    display(missingness.loc[missingness["status"].eq("Included")])
    display(Image(filename=str(RUN_DIR / "selection_and_dictionary" / "cohort_flow.png")))
    print("Review source timing:", RUN_DIR / "static_predictor_timing_LOCAL_ONLY.csv")
    print("Review wearable quality:", RUN_DIR / "wearable_source_status_LOCAL_ONLY.csv")
    print("New eligible participants by split:")
    display(analysis_status.groupby("recommended_split")["eligible_common"].sum().rename("n_eligible"))
    # Notebook code cell 10
    oof_predictions = train_revision(CFG, analysis_status, RUN_DIR)
    training_performance = pd.read_csv(RUN_DIR / "training" / "oof_reports" / "performance_with_conditional_CI.csv")
    display(training_performance.loc[
        training_performance["model"].str.endswith("__ensemble") & training_performance["metric"].eq("MAE")
    ])
    display(pd.read_csv(RUN_DIR / "training" / "final_ensemble_weights.csv"))
    print("Revised lock:", RUN_DIR / "revised_frozen_models" / "revision_lock.json")
    # Notebook code cell 12
    validation_predictions = evaluate_revision(CFG, analysis_status, RUN_DIR, "validation")
    validation_reports = RUN_DIR / "validation_exploratory_evaluation" / "reports"
    display(pd.read_csv(validation_reports / "performance_with_conditional_CI.csv"))
    display(pd.read_csv(validation_reports / "paired_comparisons.csv").query("role == 'designated_primary_exploratory'"))
    # Notebook code cell 14
    test_predictions = evaluate_revision(CFG, analysis_status, RUN_DIR, "test")
    test_reports = RUN_DIR / "test_exploratory_evaluation" / "reports"
    display(pd.read_csv(test_reports / "performance_with_conditional_CI.csv"))
    paired = pd.read_csv(test_reports / "paired_comparisons.csv")
    display(paired.loc[paired["metric"].eq("MAE")])
    display(pd.read_csv(test_reports / "calibration_diagnostics.csv"))
    display(Image(filename=str(test_reports / "CWDS_calibration.png")))
    # Notebook code cell 16
    importance = explain_revision(
        CFG, analysis_status, RUN_DIR,
        split="validation",
        group="CWDS",
        background_n=32,
        n_permutations=2,
    )
    display(importance.drop(columns="feature"))
    SHAP_DIR = RUN_DIR / "validation_CWDS_SHAP"
    display(Image(filename=str(SHAP_DIR / "SHAP_beeswarm_and_importance.png")))
    print("Vector figures:", SHAP_DIR / "SHAP_beeswarm_and_importance.pdf")
    print("Formatted table:", SHAP_DIR / "SHAP_importance.html")
    # Notebook code cell 18
    outputs = {
        "Cohort flow": RUN_DIR / "cohort_flow.csv",
        "Selection comparison": RUN_DIR / "selection_and_dictionary" / "included_versus_excluded.csv",
        "Missingness": RUN_DIR / "selection_and_dictionary" / "missingness_by_selection_and_split.csv",
        "Feature dictionary": RUN_DIR / "selection_and_dictionary" / "feature_dictionary.csv",
        "Nested CV": RUN_DIR / "training" / "oof_reports" / "performance_with_conditional_CI.csv",
        "Validation": RUN_DIR / "validation_exploratory_evaluation" / "reports" / "performance_with_conditional_CI.csv",
        "Test (exploratory revision)": RUN_DIR / "test_exploratory_evaluation" / "reports" / "performance_with_conditional_CI.csv",
        "Paired and individual-learner comparisons": RUN_DIR / "test_exploratory_evaluation" / "reports" / "paired_comparisons.csv",
        "SHAP table": RUN_DIR / "validation_CWDS_SHAP" / "SHAP_importance.csv",
    }
    display(pd.DataFrame([{"output": k, "saved": v.is_file(), "path": str(v)} for k, v in outputs.items()]))
    print("All outputs:", RUN_DIR)
    print("Keep participant-level files marked LOCAL_ONLY in your approved local environment.")
    # Notebook code cell 20
    RUN_ADDITIONAL_SENSITIVITY = False
    if RUN_ADDITIONAL_SENSITIVITY:
        SENSITIVITY_DIR = additional_quality_sensitivity(CFG, RUN_DIR)
        print("Additional post-test sensitivity:", SENSITIVITY_DIR)
    else:
        print("Additional sensitivity not run.")
    # Notebook code cell 22
    # RUN_DIR = Path(r"D:\POSTDOC\YSN\AimAhead\AI_READI_Analysis\hba1c_no_cgm_v1\YOUR_ACTUAL_RUN_FOLDER")
    # protocol = json.loads((RUN_DIR / "protocol.json").read_text(encoding="utf-8"))
    # CFG = Config(**{key: protocol[key] for key in Config.__dataclass_fields__})
    # analysis_status = load_prepared(RUN_DIR)
    # if (RUN_DIR / "revised_frozen_models" / "revision_lock.json").is_file():
    #     verify_lock(RUN_DIR)
    # print("Resumed:", RUN_DIR)
