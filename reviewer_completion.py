"""Additional analyses for the reviewed CGM-independent AI-READI run.

All participant-level outputs stay in LOCAL_ONLY. Never edits original run files.
New checks do not turn the previously examined test split into untouched data.
"""
from pathlib import Path, PureWindowsPath
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import platform
import sys
import warnings
import zipfile
import numpy as np
import pandas as pd

GROUPS = ['W', 'WD', 'WDS', 'C', 'CD', 'CW', 'CWD', 'CWDS']
EXPECTED_COUNTS = dict(zip(GROUPS, [20, 29, 43, 5, 14, 25, 34, 48]))
REVIEWED_HASHES = {'aireadi_no_cgm.py': 'e9976a62575a1801439fde414548a26a308b78dfde1540f7def3b0943b40e2ba', 'aireadi_no_cgm_models.py': '91cfd9aa70bf60b6e72a1545df705e39a46a351438dc40a91d081d98d1032aa4'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, obj):
    Path(path).write_text(json.dumps(native(obj), indent=2, allow_nan=False), encoding='utf-8')


def versions():
    out = {'python': platform.python_version()}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'joblib']:
        out[name] = importlib.metadata.version(name)
    return out


def resolve_locked_path(old, run, code):
    # The run was moved into Redo3. Keep the saved expected hash unchanged.
    parts = PureWindowsPath(str(old)).parts if '\\' in str(old) else Path(old).parts
    candidates = []
    name = parts[-1]
    if name in ['aireadi_no_cgm.py', 'aireadi_no_cgm_models.py']:
        candidates.append(code / name)
    elif run.name in parts:
        position = max(i for i, v in enumerate(parts) if v == run.name)
        candidates.append(run.joinpath(*parts[position + 1:]))
    candidates.append(Path(old))
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def setup(run_dir, code_dir):
    run, code = Path(run_dir), Path(code_dir)
    if not run.is_dir():
        raise FileNotFoundError(f'Run folder not found: {run}')
    required = ['protocol.json', 'environment.json', 'preparation_complete.json',
                'all_roster_analysis_status_LOCAL_ONLY.csv',
                'revised_frozen_models/revision_lock.json']
    missing = [str(run / name) for name in required if not (run / name).is_file()]
    missing += [str(code / name) for name in ['aireadi_no_cgm.py', 'aireadi_no_cgm_models.py']
                if not (code / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing existing run inputs:\n' + '\n'.join(missing))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out = run / 'reviewer_completion' / stamp
    local, share = out / 'LOCAL_ONLY', out / 'aggregate_for_review'
    local.mkdir(parents=True, exist_ok=False)
    share.mkdir()
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    lock = json.loads((run / 'revised_frozen_models/revision_lock.json').read_text(encoding='utf-8'))
    if lock.get('software_test_only') or protocol.get('software_test_only'):
        raise ValueError('This is a software-test run, not the manuscript run.')
    if lock.get('training_n') != 1227:
        raise ValueError('Expected the revised 1,227-participant training run.')
    audit, passed = [], set()
    for old, expected in lock['hashes'].items():
        path = resolve_locked_path(old, run, code)
        actual = sha(path) if path.is_file() else None
        ok = actual == expected
        audit.append({'artifact': path.name, 'recorded_sha256': expected,
                      'current_sha256': actual, 'matches_lock': ok,
                      'matches_reviewed_source': actual == REVIEWED_HASHES.get(path.name)
                      if path.name in REVIEWED_HASHES else None})
        if ok:
            passed.add(path.resolve())
    pd.DataFrame(audit).to_csv(share / 'artifact_integrity.csv', index=False)
    # A mismatch is reported; it is never fixed by rewriting the original lock.
    if not all(item['matches_lock'] for item in audit):
        raise ValueError(f'An archived hash did not match. Inspect {share / "artifact_integrity.csv"}. '
                         'Restore the corresponding original file before executing this workflow.')
    critical = [run / 'protocol.json', run / 'all_roster_analysis_status_LOCAL_ONLY.csv']
    critical += [run / 'revised_frozen_models' / f'frozen_{g}.joblib' for g in GROUPS]
    critical += [code / 'aireadi_no_cgm.py', code / 'aireadi_no_cgm_models.py']
    if any(p.resolve() not in passed for p in critical):
        raise ValueError('The lock does not cover every required model, table, protocol and source file.')
    # Require a fresh kernel rather than silently using previously imported, different code.
    for name in ['aireadi_no_cgm', 'aireadi_no_cgm_models']:
        if name in sys.modules:
            raise RuntimeError('Restart the kernel and run this notebook first; an analysis module is already imported.')
    sys.path.insert(0, str(code.resolve()))
    prep = importlib.import_module('aireadi_no_cgm')
    mdl = importlib.import_module('aireadi_no_cgm_models')
    cfg = prep.Config(**{f.name: protocol[f.name] for f in fields(prep.Config) if f.name in protocol})
    mdl.configure(cfg)
    table = mdl.load_prepared(run)
    train = table.loc[table.recommended_split.eq('train') & table.eligible_common].reset_index(drop=True)
    for split, expected in [('train', 1227), ('validation', 257), ('test', 276)]:
        if int((table.recommended_split.eq(split) & table.eligible_common).sum()) != expected:
            raise ValueError(f'Unexpected eligible count for {split}; do not mix analysis versions.')
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    bundles = {}
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        for g in GROUPS:
            bundle = joblib.load(run / 'revised_frozen_models' / f'frozen_{g}.joblib')
            if set(bundle['training_ids']) != set(train.person_id.astype(str)):
                raise ValueError(f'{g}: saved training membership differs from the prepared table.')
            if len(bundle['features']) != EXPECTED_COUNTS[g]:
                raise ValueError(f'{g}: unexpected original feature count.')
            if set(bundle['algorithms']) != {'ridge', 'random_forest', 'hist_gradient_boosting'}:
                raise ValueError(f'{g}: unexpected learner set; reconcile the executed run before proceeding.')
            bundles[g] = bundle
    recorded = json.loads((run / 'environment.json').read_text(encoding='utf-8'))
    write_json(share / 'recorded_run_environment.json', {
        'record': recorded, 'sha256': sha(run / 'environment.json'),
        'capture_stage_in_reviewed_code': 'new_run, before preparation',
        'was_environment_file_covered_by_training_lock': (run / 'environment.json').resolve() in passed,
        'training_environment_continuity': 'Requires contemporaneous training log or author confirmation; '
        'these are not automatically verified fit-time versions.'})
    write_json(share / 'current_recovery_environment.json', versions())
    write_json(share / 'verified_protocol_for_review.json', {
        k: v for k, v in protocol.items() if k not in ['dataset', 'analysis_base']})
    write_json(local / 'source_locations.json', {'run': run, 'code': code, 'protocol': protocol})
    ctx = dict(run=run, code=code, out=out, local=local, share=share, cfg=cfg,
               prep=prep, mdl=mdl, table=table, train=train, bundles=bundles, lock=lock)
    print(f'Run integrity checked. New outputs: {out}')
    print('Original models, prepared rows and historical reports are preserved.')
    return ctx


def model_specifications(ctx):
    rows, full_params, names = [], {}, {}
    searches = pd.read_csv(ctx['run'] / 'training/final_inner_search_records.csv')
    for g, b in ctx['bundles'].items():
        full_params[g], names[g] = {}, {}
        for learner in b['algorithms']:
            pipe = b['models'][learner]
            preprocess, estimator = pipe.named_steps['preprocess'], pipe.named_steps['model']
            transformed = preprocess.transform(ctx['train'].iloc[:8][b['features']])
            encoded_names = list(map(str, preprocess.get_feature_names_out()))
            dimension = int(transformed.shape[1])
            if len(encoded_names) != dimension or estimator.n_features_in_ != dimension:
                raise AssertionError(f'Dimensionality mismatch: {g}/{learner}')
            selected = b['selected_params'][learner]
            effective = estimator.get_params(deep=False)
            if any(effective[k] != v for k, v in selected.items()):
                raise AssertionError(f'Stored selection differs from fitted learner: {g}/{learner}')
            candidates = searches.loc[searches.stage.eq(f'full training/{g}') & searches.algorithm.eq(learner)]
            if candidates.empty:
                raise ValueError(f'Final search record absent: {g}/{learner}')
            best = candidates.sort_values('inner_MAE_for_selection_only', kind='stable').iloc[0]
            agrees = json.loads(best['params']) == selected
            if not agrees:
                raise ValueError(f'Selection/search record disagreement: {g}/{learner}')
            numeric = preprocess.named_transformers_.get('numeric')
            n_indicators = len(numeric.named_steps['impute'].indicator_.features_) if numeric is not None else 0
            rows.append({'Group': g, 'Learner': learner, 'Original variables': len(b['features']),
                         'Final encoded columns': dimension, 'Numeric missing indicators': n_indicators,
                         'Selected hyperparameters': json.dumps(selected, sort_keys=True),
                         'Final search record agrees': agrees,
                         'Training participants': len(ctx['train'])})
            full_params[g][learner] = {'selected_hyperparameters': selected, 'all_fitted_estimator_parameters': effective}
            names[g][learner] = encoded_names
    result = pd.DataFrame(rows)
    result.to_csv(ctx['share'] / 'final_model_specifications_24_learners.csv', index=False)
    write_json(ctx['share'] / 'final_model_parameters_full.json', full_params)
    # Encoded names can contain response categories. Retain locally for provenance review.
    write_json(ctx['local'] / 'encoded_column_names.json', names)
    summary = []
    for g in GROUPS:
        z = result.loc[result.Group.eq(g)]
        dims = z['Final encoded columns'].unique()
        summary.append({'Group': g, 'Original variables': EXPECTED_COUNTS[g],
                        'Final encoded columns': int(dims[0]) if len(dims) == 1 else 'Learner-specific; see full table',
                        'All three learners agree': len(dims) == 1})
    summary = pd.DataFrame(summary)
    summary.to_csv(ctx['share'] / 'S13_final_encoded_dimensions.csv', index=False)
    print(summary.to_string(index=False))
    return result


def provenance_inputs(ctx, dataset_dir, author_record=None):
    """Find documentary candidates; never infer release or acquisition date from mtimes."""
    root = Path(dataset_dir)
    rows = []
    if root.is_dir():
        paths = list(root.iterdir()) + list(root.parent.iterdir())
        for p in sorted(set(paths)):
            if p.is_file() and any(s in p.name.lower() for s in ['dataset_description', 'readme', 'release', 'citation', 'license', 'version']):
                rows.append({'filename': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size})
    pd.DataFrame(rows, columns=['filename', 'sha256', 'bytes']).to_csv(ctx['share'] / 'dataset_document_candidates.csv', index=False)
    record = author_record or {}
    write_json(ctx['share'] / 'author_provenance_record.json', {
        'dataset_release_identifier': record.get('dataset_release_identifier'),
        'dataset_release_doi': record.get('dataset_release_doi'),
        'acquisition_date': record.get('acquisition_date'),
        'acquisition_evidence': record.get('acquisition_evidence'),
        'training_environment_evidence': record.get('training_environment_evidence'),
        'public_code_repository': record.get('public_code_repository'),
        'immutable_commit_or_doi': record.get('immutable_commit_or_doi'),
        'note': 'Null fields are unresolved. User-entered fields require cited documentary support. '
        'Dataset folder UUID, documentation version, file modification time and paper date are not substitutes.'})
    print('Dataset and publication provenance template saved; no dates or release identifiers inferred.')


def education_audit(ctx, dataset_dir, source_code_files=()):
    """Collect candidates. A text match or plausible numeric range is NOT verification."""
    observation = Path(dataset_dir) / 'clinical_data/observation.csv'
    if not observation.is_file():
        raise FileNotFoundError(observation)
    schema = pd.read_csv(observation, nrows=0).columns.tolist()
    allowed = ['observation_concept_id', 'observation_source_concept_id', 'observation_source_value',
               'unit_concept_id', 'unit_source_value', 'value_as_concept_id']
    cols = [c for c in allowed if c in schema]
    if 'observation_source_value' not in cols:
        raise ValueError('OMOP observation_source_value was not found.')
    counts = {}
    for chunk in pd.read_csv(observation, usecols=cols, dtype=str, keep_default_na=False, chunksize=100000):
        mask = chunk.observation_source_value.str.contains(r'educat|school|degree|moca.*edu', case=False, regex=True, na=False)
        for values, n in chunk.loc[mask, cols].value_counts(dropna=False).items():
            key = values if isinstance(values, tuple) else (values,)
            counts[key] = counts.get(key, 0) + int(n)
    candidates = pd.DataFrame([dict(zip(cols, k), n_observation_rows=v) for k, v in counts.items()],
                              columns=cols + ['n_observation_rows'])
    candidates.to_csv(ctx['local'] / 'education_source_candidates_LOCAL_ONLY.csv', index=False)
    snippets = []
    for file in source_code_files:
        p = Path(file)
        if not p.is_file():
            continue
        if p.suffix.lower() == '.ipynb':
            cells = [c for c in json.loads(p.read_text(encoding='utf-8')).get('cells', []) if c.get('cell_type') == 'code']
            source = '\n'.join(''.join(c.get('source', [])) for c in cells)
        else:
            source = p.read_text(encoding='utf-8', errors='replace')
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if 'years_of_education' in line:
                snippets.append({'source_file': str(p), 'line': i + 1,
                                 'snippet': '\n'.join(lines[max(0, i-8): i+9]), 'source_sha256': sha(p)})
    write_json(ctx['local'] / 'education_code_snippets_LOCAL_ONLY.json', snippets)
    write_json(ctx['share'] / 'education_verification_status.json', {
        'status': 'UNRESOLVED', 'candidate_observation_mappings_found': len(candidates),
        'requirements': ['Exact source item/concept', 'Response representation and units',
                         'Derivation into years_of_education', 'Missing/special code rules',
                         'Record-level agreement with staged and prepared education values'],
        'interpretation': 'Candidate text matches, distributions, and column names alone do not verify the field.'})
    print(f'{len(candidates)} education mapping candidates saved for LOCAL review.')
    print('If the derivation cannot be verified, run the separate education-exclusion sensitivity cell.')
    return candidates


def saved_predictions(ctx, split):
    path = ctx['run'] / f'{split}_exploratory_evaluation/predictions_LOCAL_ONLY.csv'
    pred = pd.read_csv(path, dtype={'person_id': str})
    cohort = ctx['table'].loc[ctx['table'].recommended_split.eq(split) & ctx['table'].eligible_common]
    if pred.person_id.duplicated().any() or set(pred.person_id) != set(cohort.person_id):
        raise ValueError(f'{split}: prediction membership mismatch.')
    pred = pred.set_index('person_id').loc[cohort.person_id].reset_index()
    if not np.allclose(pred.observed_hba1c_percent, cohort.target_hba1c_percent, rtol=0, atol=1e-10):
        raise ValueError(f'{split}: saved outcomes differ from the prepared table.')
    # Reproduce predictions from each original bundle, without fitting.
    for g, b in ctx['bundles'].items():
        current = ctx['mdl'].bundle_predictions(b, cohort)['ensemble']
        if not np.allclose(current, pred[g + '__ensemble'], rtol=1e-9, atol=1e-9):
            raise ValueError(f'{split}/{g}: saved predictions not reproduced.')
    return pred, cohort.reset_index(drop=True)


def age_strata(ctx):
    rows = []
    for split in ['validation', 'test']:
        pred, cohort = saved_predictions(ctx, split)
        age = pd.to_numeric(cohort.age_years, errors='raise').to_numpy(float)
        if not np.isfinite(age).all() or (age < 40).any():
            raise ValueError('Expected available roster age >=40; inspect unexpected ages locally before defining bins.')
        strata = [('40–54', (age >= 40) & (age < 55)),
                  ('55–64', (age >= 55) & (age < 65)), ('65 and older', age >= 65)]
        models = [g + '__ensemble' for g in GROUPS] + ['baseline__training_mean', 'baseline__training_median']
        rng = np.random.default_rng(ctx['cfg'].seed + 3000)
        for label, mask in strata:
            y = pred.loc[mask, 'observed_hba1c_percent'].to_numpy(float)
            n = len(y)
            idx = rng.integers(0, n, size=(ctx['cfg'].bootstrap_draws, n)) if n >= 2 else None
            for model in models:
                p = pred.loc[mask, model].to_numpy(float)
                err = p-y
                for metric, values in [('MAE', np.abs(err)), ('Bias predicted minus observed', err)]:
                    ci = np.quantile(values[idx].mean(axis=1), [.025, .975]) if idx is not None else [np.nan, np.nan]
                    rows.append({'split': split, 'age_group_years': label, 'model': model, 'n': n,
                                 'observed_HbA1c_mean': y.mean() if n else np.nan,
                                 'metric': metric, 'estimate': values.mean() if n else np.nan,
                                 'ci_low': ci[0], 'ci_high': ci[1],
                                 'small_group_under_20': n < 20,
                                 'status': 'Additional descriptive analysis after test exposure'})
    result = pd.DataFrame(rows)
    result.to_csv(ctx['share'] / 'S16_age_stratified_MAE_bias.csv', index=False)
    print(result.loc[result.model.eq('CWDS__ensemble')].to_string(index=False))
    return result


def shap_stability(ctx):
    """Three new seeds, fixed documented training background, no outcome use."""
    folder = ctx['run'] / 'validation_CWDS_SHAP'
    meta_path = folder / 'SHAP_metadata.json'
    bg_path = folder / 'background_ids_LOCAL_ONLY.csv'
    if not meta_path.is_file() or not bg_path.is_file():
        raise FileNotFoundError('To test the published Figure 5, its original SHAP_metadata.json and '
                                'background_ids_LOCAL_ONLY.csv are needed. No different background is silently substituted.')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    original_ids = pd.read_csv(bg_path, dtype={'person_id': str}).person_id.tolist()
    if len(original_ids) != 32 or len(set(original_ids)) != 32:
        raise ValueError('The original background is not 32 unique participants.')
    b = ctx['bundles']['CWDS']
    if not set(original_ids).issubset(set(b['training_ids'])):
        raise ValueError('Background includes a non-training participant.')
    background = ctx['train'].set_index('person_id').loc[original_ids, b['features']]
    cohort = ctx['table'].loc[ctx['table'].recommended_split.eq('validation') & ctx['table'].eligible_common]
    for key, expected in {'group': 'CWDS', 'split': 'validation', 'n_explained': len(cohort),
                          'n_background_training_only': 32, 'permutations_per_background': 2}.items():
        if meta.get(key) != expected:
            raise ValueError(f'Original SHAP metadata does not confirm {key}={expected!r}; '
                             'reconcile the published estimator before running a stability comparison.')
    X = cohort[b['features']].reset_index(drop=True)
    def predict(x):
        return ctx['mdl'].bundle_predictions(b, x)['ensemble']
    seeds = [ctx['cfg'].seed + offset for offset in [1001, 1002, 1003]]
    importance, checks = {}, []
    for seed in seeds:
        print(f'SHAP stability seed {seed}; same 32 training background participants; 128 paths.', flush=True)
        values, base, predictions = ctx['mdl'].permutation_shapley(predict, background, X, n_permutations=2, seed=seed)
        importance[seed] = np.abs(values).mean(axis=0)
        pd.DataFrame(values, columns=b['features']).assign(person_id=cohort.person_id.to_numpy()).to_csv(
            ctx['local'] / f'SHAP_seed_{seed}_LOCAL_ONLY.csv', index=False)
        checks.append({'seed': seed, 'n_explained': len(X), 'background_n': len(background),
                       'paths_per_participant': 128, 'max_additivity_error': float(np.abs(base + values.sum(1) - predictions).max())})
    from scipy.stats import spearmanr
    comparisons = []
    for a, bseed in itertools.combinations(seeds, 2):
        ia, ib = importance[a], importance[bseed]
        # Break exact ties deterministically by original feature order; report no p value.
        topa = set(np.argsort(-ia, kind='stable')[:10])
        topb = set(np.argsort(-ib, kind='stable')[:10])
        comparisons.append({'seed_1': a, 'seed_2': bseed, 'Spearman_rho': float(spearmanr(ia, ib).statistic),
                            'top10_overlap_count': len(topa & topb), 'top10_overlap_fraction': len(topa & topb)/10})
    pd.DataFrame(comparisons).to_csv(ctx['share'] / 'S15_SHAP_repeat_seed_stability.csv', index=False)
    importance_table = pd.DataFrame({'feature': ctx['bundles']['CWDS']['features'],
        'Predictor': [ctx['mdl'].LABELS[f] for f in ctx['bundles']['CWDS']['features']],
        **{f'mean_absolute_SHAP_seed_{seed}': importance[seed] for seed in seeds}})
    importance_table.to_csv(ctx['share'] / 'SHAP_stability_importance.csv', index=False)
    allowed = ['method', 'group', 'split', 'n_explained', 'n_background_training_only', 'permutations_per_background',
               'seed', 'max_additivity_error', 'all_eligible_split_participants_explained']
    write_json(ctx['share'] / 'SHAP_original_and_stability_metadata.json', {
        'original_metadata': {k: meta[k] for k in allowed if k in meta},
        'original_metadata_sha256': sha(meta_path), 'background_ids_file_sha256': sha(bg_path),
        'stability_runs': checks, 'same_training_background_all_runs': True,
        'scope': 'Monte Carlo seed sensitivity conditional on this background and frozen model; '
                 'does not establish stability to another background, training resample, or model fit.',
        'status': 'Additional post-test analysis; original Figure 5 is not overwritten.'})
    print(pd.DataFrame(comparisons).to_string(index=False))
    return pd.DataFrame(comparisons)


def education_exclusion(ctx):
    """Refit WDS and CWDS without education, training only. No historical artifact edits."""
    from sklearn.model_selection import KFold
    import joblib
    cfg, mdl, train = ctx['cfg'], ctx['mdl'], ctx['train']
    mdl.configure(cfg)
    if mdl.INCLUDE_XGBOOST or mdl.QUICK:
        raise ValueError('Sensitivity requires the three production learners from the manuscript.')
    target = 'target_hba1c_percent'
    y = train[target].to_numpy(float)
    # Validate the fold memberships against the original saved OOF record.
    original_oof = pd.read_csv(ctx['run'] / 'training/nested_oof_predictions_LOCAL_ONLY.csv', dtype={'person_id': str})
    if original_oof.person_id.duplicated().any() or set(original_oof.person_id) != set(train.person_id):
        raise ValueError('OOF training membership mismatch.')
    original_oof = original_oof.set_index('person_id').loc[train.person_id].reset_index()
    outer = list(KFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.seed).split(train))
    for fold, (_, va) in enumerate(outer, 1):
        if not original_oof.iloc[va].outer_fold.eq(fold).all():
            raise ValueError('Regenerated folds do not reproduce the saved OOF assignments.')
    groups = {g: [f for f in ctx['bundles'][g]['features'] if f != 'years_of_education'] for g in ['WDS', 'CWDS']}
    folder = ctx['local'] / 'education_exclusion'
    folder.mkdir(exist_ok=False)
    write_json(ctx['share'] / 'education_exclusion_plan.json', {
        'status': 'Additional post-test exploratory sensitivity', 'removed_feature': 'years_of_education',
        'groups': {g + '_without_education': len(f) for g, f in groups.items()},
        'same_eligibility_features_other_than_education_folds_and_candidate_grid': True,
        'tuning_and_ensemble_weight_estimation': 'Training only; repeated in the original nested partitions',
        'primary_models_modified': False,
        'interpretation': 'No-education models may replace unresolved SDH specifications only after all affected tables '
        'and attributions are regenerated. They remain revised exploratory analyses.'})
    grid = mdl.model_grid()
    executed_search = pd.read_csv(ctx['run'] / 'training/final_inner_search_records.csv')
    for g in groups:
        for learner, candidates in grid.items():
            recorded = executed_search.loc[executed_search.stage.eq(f'full training/{g}') &
                                            executed_search.algorithm.eq(learner)].sort_values('candidate')
            if [json.loads(v) for v in recorded.params] != candidates:
                raise ValueError(f'{g}/{learner}: current grid differs from the executed search. '
                                 'Restore the original candidate grid; do not silently change it.')
    oof = {g: np.full(len(train), np.nan) for g in groups}
    records, weights, fitted = [], [], {}
    for fold, (tr, va) in enumerate(outer, 1):
        inner = list(KFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.seed+fold).split(tr))
        for g, features in groups.items():
            bundle, search = mdl.tune_fit_bundle(train.iloc[tr], y[tr], features, inner, grid,
                                                  f'education exclusion outer {fold}/{g}')
            oof[g][va] = mdl.bundle_predictions(bundle, train.iloc[va])['ensemble']
            records.extend(search)
    inner = list(KFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.seed+100).split(train))
    for g, features in groups.items():
        bundle, search = mdl.tune_fit_bundle(train, y, features, inner, grid, f'education exclusion full training/{g}')
        bundle.update(training_ids=train.person_id.tolist(), analysis_status='additional_post_test_education_exclusion')
        fitted[g] = bundle
        joblib.dump(bundle, folder / f'{g}_without_education.joblib', compress=3)
        records.extend(search)
        weights += [{'group': g, 'learner': a, 'weight': float(w)} for a, w in zip(bundle['algorithms'], bundle['weights'])]
    pd.DataFrame(records).to_csv(ctx['share'] / 'education_exclusion_inner_search.csv', index=False)
    pd.DataFrame(weights).to_csv(ctx['share'] / 'education_exclusion_weights.csv', index=False)
    perf, comparisons = [], []
    for split in ['train', 'validation', 'test']:
        if split == 'train':
            original, cohort = original_oof, train
            new = oof
        else:
            original, cohort = saved_predictions(ctx, split)
            new = {g: mdl.bundle_predictions(b, cohort)['ensemble'] for g, b in fitted.items()}
        ysplit = original.observed_hba1c_percent.to_numpy(float)
        idx = mdl.bootstrap_indices(original, cfg)
        allp = {g + '_without_education__ensemble': p for g, p in new.items()}
        for g in groups:
            allp[g + '__ensemble'] = original[g + '__ensemble'].to_numpy(float)
        for ref in ['WD', 'CWD']:
            allp[ref + '__ensemble'] = original[ref + '__ensemble'].to_numpy(float)
        estimates, boots = {}, {}
        for name, p in allp.items():
            estimates[name] = mdl.metrics(ysplit, p)
            boots[name] = mdl.vector_metrics(ysplit[idx], p[idx])
            for metric in ['MAE', 'RMSE', 'R2']:
                finite = boots[name][metric][np.isfinite(boots[name][metric])]
                ci = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
                perf.append({'split': split, 'model': name, 'n': len(ysplit), 'metric': metric,
                             'estimate': estimates[name][metric], 'ci_low': ci[0], 'ci_high': ci[1]})
        for g, ref in [('WDS', 'WD'), ('CWDS', 'CWD')]:
            first = g + '_without_education__ensemble'
            for reference in [g + '__ensemble', ref + '__ensemble']:
                for metric in ['MAE', 'RMSE', 'R2']:
                    delta = boots[first][metric] - boots[reference][metric]
                    finite = delta[np.isfinite(delta)]
                    ci = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
                    comparisons.append({'split': split, 'first': first, 'reference': reference, 'metric': metric,
                        'delta_first_minus_reference': estimates[first][metric] - estimates[reference][metric],
                        'ci_low': ci[0], 'ci_high': ci[1], 'finite_draws': len(finite),
                        'status': 'Additional post-test exploratory sensitivity'})
        pd.DataFrame({'person_id': cohort.person_id, 'observed_hba1c_percent': ysplit, **allp}).to_csv(
            folder / f'{split}_predictions_LOCAL_ONLY.csv', index=False)
    pd.DataFrame(perf).to_csv(ctx['share'] / 'S17_education_exclusion_performance.csv', index=False)
    pd.DataFrame(comparisons).to_csv(ctx['share'] / 'S17_education_exclusion_paired_comparisons.csv', index=False)
    print('Education-exclusion sensitivity complete. Original models and Figure 5 remain unchanged.')
    return pd.DataFrame(comparisons)


def export_review(ctx):
    # Explicit output allowlist: no participant tables, model files, background IDs or code snippets.
    allowed = ['artifact_integrity.csv', 'recorded_run_environment.json', 'current_recovery_environment.json',
               'verified_protocol_for_review.json', 'final_model_specifications_24_learners.csv',
               'final_model_parameters_full.json', 'S13_final_encoded_dimensions.csv',
               'dataset_document_candidates.csv', 'author_provenance_record.json', 'education_verification_status.json',
               'S16_age_stratified_MAE_bias.csv', 'S15_SHAP_repeat_seed_stability.csv',
               'SHAP_stability_importance.csv', 'SHAP_original_and_stability_metadata.json',
               'education_exclusion_plan.json', 'education_exclusion_inner_search.csv', 'education_exclusion_weights.csv',
               'S17_education_exclusion_performance.csv', 'S17_education_exclusion_paired_comparisons.csv']
    present = [ctx['share']/name for name in allowed if (ctx['share']/name).is_file()]
    write_json(ctx['share'] / 'recovery_manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer_completion_module_sha256': sha(Path(__file__)),
        'source_run': ctx['run'].name, 'artifacts': {p.name: sha(p) for p in present},
        'not_completed': [name for name in allowed if not (ctx['share']/name).is_file()],
        'participant_data_or_models_in_archive': False,
        'note': 'Review aggregate outputs against applicable data-use terms before sharing.'})
    target = ctx['out'] / 'AI_READI_Reviewer_Completion_Results.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in present + [ctx['share'] / 'recovery_manifest.json']:
            archive.write(path, path.name)
    print(f'Aggregate results archive: {target}')
    return target
