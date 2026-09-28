"""Training-only fitting and descriptive evaluation after the original test was seen."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from aireadi_no_cgm import *

SEED=20260927
QUICK=False
INCLUDE_XGBOOST=False
N_JOBS_RF=2

def configure(cfg):
    global SEED, QUICK
    SEED=cfg.seed; QUICK=cfg.software_test_only

def make_pipeline(features, algorithm, params):
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]
    transformers = []
    if num:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), num))
    if cat:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    prep = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    if algorithm == "ridge":
        model = Ridge(**params)
    elif algorithm == "random_forest":
        model = RandomForestRegressor(n_estimators=12 if QUICK else 250,
                    random_state=SEED, n_jobs=N_JOBS_RF, **params)
    elif algorithm == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(max_iter=15 if QUICK else 200,
                    early_stopping=False, random_state=SEED, **params)
    elif algorithm == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=12 if QUICK else 250, objective="reg:squarederror",
                    tree_method="hist", random_state=SEED, n_jobs=N_JOBS_RF, **params)
    else:
        raise ValueError(algorithm)
    return Pipeline([("preprocess", prep), ("model", model)])

def model_grid():
    grid = {
        "ridge": [{"alpha": a} for a in [1.0, 10.0, 100.0]],
        "random_forest": [{"max_depth": 6, "min_samples_leaf": 10, "max_features": 1.0},
                          {"max_depth": None, "min_samples_leaf": 20, "max_features": 1.0}],
        "hist_gradient_boosting": [
            {"learning_rate": 0.03, "max_leaf_nodes": 7, "min_samples_leaf": 20, "l2_regularization": 5.0},
            {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 30, "l2_regularization": 10.0}],
    }
    if INCLUDE_XGBOOST:
        import xgboost  # fail early, rather than silently omit a learner
        grid["xgboost"] = [
            {"max_depth": 2, "learning_rate": 0.03, "min_child_weight": 10, "reg_lambda": 10},
            {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 20, "reg_lambda": 10}]
    return {k: v[:1] if QUICK else v for k, v in grid.items()}

def fit_mae_weights(predictions, y):
    """Nonnegative weights sum to one. Exact convex MAE minimization.
    Weights are estimated only using predictions internal to training data.
    """
    P = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(P).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite inputs to ensemble-weight fitting.")
    n, m = P.shape
    # min sum(u)/n, subject to |P w - y| <= u, w>=0, sum(w)=1.
    A = hstack([csr_matrix(P), -eye(n, format="csr")], format="csr")
    B = hstack([csr_matrix(-P), -eye(n, format="csr")], format="csr")
    from scipy.sparse import vstack
    eq = csr_matrix(np.r_[np.ones(m), np.zeros(n)].reshape(1, -1))
    res = linprog(np.r_[np.zeros(m), np.ones(n) / n], A_ub=vstack([A, B]),
        b_ub=np.r_[y, -y], A_eq=eq, b_eq=[1.0], bounds=[(0, None)] * (m + n), method="highs")
    if not res.success:
        raise RuntimeError(f"Ensemble optimization failed: {res.message}")
    weights = np.maximum(res.x[:m], 0)
    weights /= weights.sum()
    return weights

def tune_fit_bundle(X, y, features, inner_splits, grid, label):
    """Tune each learner by inner OOF MAE, then learn blend weights on the
    selected inner OOF predictions. Both choices are evaluated by the OUTER
    holdout. Inner blend MAE is intentionally never reported as validation.
    """
    models, selected, oof_columns, records = {}, {}, [], []
    for algorithm, candidates in grid.items():
        best = None
        for candidate_idx, params in enumerate(candidates):
            pred = np.full(len(y), np.nan)
            for tr, va in inner_splits:
                estimator = make_pipeline(features, algorithm, params)
                estimator.fit(X.iloc[tr][features], y[tr])
                pred[va] = estimator.predict(X.iloc[va][features])
            loss = float(mean_absolute_error(y, pred))
            records.append({"stage": label, "algorithm": algorithm, "candidate": candidate_idx,
                            "params": json.dumps(params, sort_keys=True), "inner_MAE_for_selection_only": loss})
            if best is None or loss < best[0]:
                best = (loss, params, pred)
        selected[algorithm] = best[1]
        oof_columns.append(best[2])
        fitted = make_pipeline(features, algorithm, best[1]).fit(X[features], y)
        models[algorithm] = fitted
        log(f"  {label}: {algorithm} fitted")
    P = np.column_stack(oof_columns)
    bundle = {"models": models, "weights": fit_mae_weights(P, y),
              "algorithms": list(models), "features": features, "selected_params": selected}
    return bundle, records

def bundle_predictions(bundle, X):
    pred = {a: bundle["models"][a].predict(X[bundle["features"]]) for a in bundle["algorithms"]}
    P = np.column_stack([pred[a] for a in bundle["algorithms"]])
    pred["ensemble"] = P @ bundle["weights"]
    pred["equal_weight_ensemble"] = P.mean(axis=1)
    return pred

def metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)), "bias_prediction_minus_observed": float(np.mean(pred - y))}

LABELS = {
 'age_years':'Age', 'bmi_source':'Body mass index', 'waist_source_cm':'Waist circumference',
 'bp1_systolic_source':'Systolic blood pressure', 'bp1_diastolic_source':'Diastolic blood pressure',
 'years_of_education':'Education (stored numeric value)', 'marital_status_source_code':'Marital status',
 'pxhi1__source_category':'Current housing stability',
 'diet1__source_category':'Fast-food meals/snacks', 'diet2__source_category':'Fruit intake',
 'diet3__source_category':'Vegetable intake', 'diet4__source_category':'Regular soda/sweet tea',
 'diet5__source_category':'Beans, chicken, or fish', 'diet6__source_category':'Regular chips/crackers',
 'diet7__source_category':'Desserts/other sweets', 'diet8__source_category':'Added margarine/butter/meat fat',
 'diet9__source_category':'Fruit-juice intake',
 'pxfi1__source_category':'Food exhausted; unable to afford more',
 'pxfi2__source_category':'Unable to afford balanced meals',
 'pxfi3__source_category':'Reduced/skipped meals because of cost',
 'pxfi4__source_category':'Ate less because of food cost',
 'pxfi5__source_category':'Hungry but unable to afford food',
 'pxhic1__source_category':'Employer/union health coverage',
 'pxhic2__source_category':'Direct-purchase/marketplace coverage',
 'pxhic3__source_category':'Medicare coverage',
 'pxhic4__source_category':'Medicaid/CHIP/other public assistance coverage',
 'pxhic5__source_category':'Military/VA health coverage',
 'pxhic7__source_category':'Other health coverage',
 'sleep__mean_hours_per_episode':'Sleep duration: episode average',
 'sleep__slope_hours_per_day':'Sleep duration: trend',
 'sleep__sd_hours_per_episode':'Sleep duration: episode variability',
 'sleep__latest_minus_previous_mean':'Sleep duration: latest change',
}
for _s,_label in [('heart_rate','Heart rate'),('oxygen_saturation','Oxygen saturation'),
                  ('respiratory_rate','Respiratory rate'),('stress','Device stress score')]:
    for _f,_meaning in [('history_mean','average'),('slope_per_day','trend'),
                        ('sd_daily_means','daily variability'),('last_day_minus_prior_mean','final-day change')]:
        LABELS[f'{_s}__{_f}']=f'{_label}: {_meaning}'
GROUP_LABELS={'W':'Wearables','WD':'Wearables + diet','WDS':'Wearables + diet + SDH',
 'C':'Clinical','CD':'Clinical + diet','CW':'Clinical + wearables',
 'CWD':'Clinical + wearables + diet','CWDS':'Clinical + wearables + diet + SDH'}


def save_figure(fig,path):
    fig.savefig(str(path)+'.png',dpi=400,bbox_inches='tight',facecolor='white')
    fig.savefig(str(path)+'.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(str(path)+'.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def check_analysis_table(table):
    ids(table,'analysis table')
    require(table,['recommended_split','eligible_common','target_hba1c_percent']+ALL_FEATURES,'analysis table')
    if table.eligible_common.dtype != bool:
        raise ValueError('eligible_common must be Boolean. Use load_prepared() to resume a run.')
    if not table.recommended_split.isin(['train','validation','test']).all(): raise ValueError('Unknown split.')


def load_prepared(out):
    out=Path(out); receipt=json.loads((out/'preparation_complete.json').read_text())
    path=out/'all_roster_analysis_status_LOCAL_ONLY.csv'
    if digest(path)!=receipt['table_sha256']: raise ValueError('Prepared table has changed since its receipt.')
    table=pd.read_csv(path,dtype={'person_id':str, **{f:object for f in CATEGORICAL}})
    for f in CATEGORICAL: table[f]=category(table[f])
    for col in ['eligible_common','laboratory_hba1c_available','visit_day_hba1c_eligible','any_quality_wearable','has_selected_window']:
        if table[col].dtype!=bool:
            table[col]=table[col].astype(str).str.lower().map({'true':True,'false':False})
            if table[col].isna().any(): raise ValueError(f'Invalid Boolean {col}')
            table[col]=table[col].astype(bool)
    check_analysis_table(table)
    return table


def train_revision(cfg,table,out):
    configure(cfg); out=Path(out); check_analysis_table(table)
    saved_protocol=json.loads((out/'protocol.json').read_text())
    for key,value in asdict(cfg).items():
        if saved_protocol[key]!=value: raise ValueError(f'Configuration changed after preparation: {key}')
    original=load_prepared(out)
    pd.testing.assert_frame_equal(table[['person_id','recommended_split','eligible_common','target_hba1c_percent']+ALL_FEATURES].reset_index(drop=True),
        original[['person_id','recommended_split','eligible_common','target_hba1c_percent']+ALL_FEATURES].reset_index(drop=True),check_dtype=False)
    train=table.loc[table.recommended_split.eq('train')&table.eligible_common].reset_index(drop=True)
    if len(train)<max(cfg.outer_folds*2,cfg.inner_folds*2): raise ValueError('Insufficient training participants for configured CV.')
    if not np.isfinite(train.target_hba1c_percent).all(): raise ValueError('Missing/nonfinite training outcomes.')
    folder=out/'training'; folder.mkdir(exist_ok=False)
    train.to_csv(folder/'training_cohort_LOCAL_ONLY.csv',index=False)
    y=train.target_hba1c_percent.to_numpy(float)
    outer=list(KFold(n_splits=cfg.outer_folds,shuffle=True,random_state=cfg.seed).split(train))
    grid=model_grid(); names=list(grid)+['ensemble','equal_weight_ensemble']
    preds={f'{g}__{a}':np.full(len(y),np.nan) for g in FEATURE_SETS for a in names}
    preds.update({f'baseline__training_{a}':np.full(len(y),np.nan) for a in ['mean','median']})
    fold_values=np.full(len(y),-1); searches=[]; weights=[]
    for fold,(tr,va) in enumerate(outer,1):
        fold_values[va]=fold
        inner=list(KFold(n_splits=cfg.inner_folds,shuffle=True,random_state=cfg.seed+fold).split(tr))
        for group,features in FEATURE_SETS.items():
            bundle,records=tune_fit_bundle(train.iloc[tr],y[tr],features,inner,grid,f'outer {fold}/{group}')
            searches.extend(records)
            for alg,p in bundle_predictions(bundle,train.iloc[va]).items(): preds[f'{group}__{alg}'][va]=p
            weights.extend({'outer_fold':fold,'group':group,'learner':a,'weight':float(w)} for a,w in zip(bundle['algorithms'],bundle['weights']))
        preds['baseline__training_mean'][va]=np.mean(y[tr]); preds['baseline__training_median'][va]=np.median(y[tr])
        pd.DataFrame({'person_id':train.person_id,'outer_fold':fold_values,'observed_hba1c_percent':y,**preds}).to_csv(folder/'oof_checkpoint_LOCAL_ONLY.csv',index=False)
        pd.DataFrame(searches).to_csv(folder/'inner_search_records.csv',index=False)
        log(f'Completed outer fold {fold}/{cfg.outer_folds}')
    if any(not np.isfinite(v).all() for v in preds.values()): raise AssertionError('Incomplete OOF predictions.')
    predictions=pd.DataFrame({'person_id':train.person_id,'outer_fold':fold_values,'observed_hba1c_percent':y,**preds})
    predictions.to_csv(folder/'nested_oof_predictions_LOCAL_ONLY.csv',index=False)
    pd.DataFrame(weights).to_csv(folder/'outer_fold_ensemble_weights.csv',index=False)
    frozen=out/'revised_frozen_models'; frozen.mkdir(exist_ok=False)
    final_records=[]; final_weights=[]
    inner=list(KFold(n_splits=cfg.inner_folds,shuffle=True,random_state=cfg.seed+100).split(train))
    for group,features in FEATURE_SETS.items():
        bundle,records=tune_fit_bundle(train,y,features,inner,grid,f'full training/{group}')
        bundle.update(training_ids=train.person_id.tolist(),baseline_training_mean=float(y.mean()),
                      baseline_training_median=float(np.median(y)),analysis_status=cfg.analysis_status)
        joblib.dump(bundle,frozen/f'frozen_{group}.joblib',compress=3)
        final_records.extend(records)
        final_weights.extend({'group':group,'learner':a,'weight':float(w)} for a,w in zip(bundle['algorithms'],bundle['weights']))
    pd.DataFrame(final_records).to_csv(folder/'final_inner_search_records.csv',index=False)
    pd.DataFrame(final_weights).to_csv(folder/'final_ensemble_weights.csv',index=False)
    # Hash trained artifacts, prepared rows, protocol, and executing source. No old models imported.
    hashed=[out/'protocol.json',out/'input_hashes.json',out/'preparation_complete.json',
            out/'wearable_file_hashes_LOCAL_ONLY.csv',out/'all_roster_analysis_status_LOCAL_ONLY.csv',folder/'training_cohort_LOCAL_ONLY.csv']
    hashed+=list(frozen.glob('*.joblib'))+[Path(__file__),Path(__file__).with_name('aireadi_no_cgm.py')]
    write_json(frozen/'revision_lock.json',{'analysis_status':cfg.analysis_status,
        'not_an_untouched_test':True,'software_test_only':cfg.software_test_only,
        'training_n':len(train),'hashes':{str(p.resolve()):digest(p) for p in hashed}})
    report_predictions(cfg,predictions,folder/'oof_reports','Nested training cross-validation')
    log('Revised training-fitted models saved. Historical test exposure remains documented.')
    return predictions


def verify_lock(out):
    folder=Path(out)/'revised_frozen_models'
    receipt=json.loads((folder/'revision_lock.json').read_text())
    for path,expected in receipt['hashes'].items():
        if not Path(path).is_file() or digest(path)!=expected:
            raise ValueError(f'Locked artifact missing/modified: {path}. Do not reuse this lock after editing.')
    return receipt


def evaluate_revision(cfg,table,out,split):
    if split not in ['validation','test']: raise ValueError('Expected validation or test.')
    receipt=verify_lock(out); check_analysis_table(table)
    # Prevent accidentally evaluating edited in-memory predictors.
    original=load_prepared(out)
    pd.testing.assert_frame_equal(table[['person_id','recommended_split','eligible_common','target_hba1c_percent']+ALL_FEATURES].reset_index(drop=True),
       original[['person_id','recommended_split','eligible_common','target_hba1c_percent']+ALL_FEATURES].reset_index(drop=True),check_dtype=False)
    hold=table.loc[table.recommended_split.eq(split)&table.eligible_common].reset_index(drop=True)
    if len(hold)<3: raise ValueError('Fewer than three eligible evaluation participants.')
    out=Path(out); dest=out/f'{split}_exploratory_evaluation'; dest.mkdir(exist_ok=False)
    pred={'person_id':hold.person_id,'observed_hba1c_percent':hold.target_hba1c_percent}
    for group in FEATURE_SETS:
        bundle=joblib.load(out/'revised_frozen_models'/f'frozen_{group}.joblib')
        if set(bundle['training_ids']) & set(hold.person_id): raise ValueError('Training/evaluation ID overlap.')
        for alg,p in bundle_predictions(bundle,hold).items(): pred[f'{group}__{alg}']=p
    for stat in ['mean','median']: pred[f'baseline__training_{stat}']=np.repeat(bundle[f'baseline_training_{stat}'],len(hold))
    predictions=pd.DataFrame(pred)
    predictions.to_csv(dest/'predictions_LOCAL_ONLY.csv',index=False)
    hold.to_csv(dest/'analysis_cohort_LOCAL_ONLY.csv',index=False)
    report_predictions(cfg,predictions,dest/'reports',f'{split.capitalize()}: post-test revision')
    write_json(dest/'evaluation_complete.json',{'split':split,'n':len(hold),'analysis_status':cfg.analysis_status,
        'models_refitted':False,'test_previously_examined':True,'software_test_only':receipt['software_test_only']})
    return predictions


def bootstrap_indices(pred,cfg):
    rng=np.random.default_rng(cfg.seed); n=len(pred)
    strata=[np.flatnonzero(pred.outer_fold.to_numpy()==f) for f in sorted(pred.outer_fold.unique())] if 'outer_fold' in pred else [np.arange(n)]
    return np.array([np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]) for _ in range(cfg.bootstrap_draws)])


def vector_metrics(y,p):
    # y and p have shape (bootstrap draws, participants).
    errors=p-y; var=((y-y.mean(axis=1,keepdims=True))**2).sum(axis=1)
    r2=np.full(len(y),np.nan)
    np.divide((errors**2).sum(axis=1),var,out=r2,where=var>0)
    return {'MAE':np.abs(errors).mean(axis=1),'RMSE':np.sqrt((errors**2).mean(axis=1)),
            'R2':1-r2,'bias_prediction_minus_observed':errors.mean(axis=1)}


def report_predictions(cfg,pred,out,title):
    out=Path(out); out.mkdir(exist_ok=True,parents=True)
    y=pred.observed_hba1c_percent.to_numpy(float); idx=bootstrap_indices(pred,cfg)
    cols=[c for c in pred if '__' in c]; boots={}; estimates={}; rows=[]
    for col in cols:
        p=pred[col].to_numpy(float)
        if not np.isfinite(p).all(): raise ValueError(f'Nonfinite predictions: {col}')
        estimates[col]=metrics(y,p); boots[col]=vector_metrics(y[idx],p[idx])
        for m,v in estimates[col].items():
            finite=boots[col][m][np.isfinite(boots[col][m])]
            ci=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
            rows.append({'model':col,'n':len(y),'metric':m,'estimate':v,'ci_low':ci[0],'ci_high':ci[1],'finite_draws':len(finite)})
    pd.DataFrame(rows).to_csv(out/'performance_with_conditional_CI.csv',index=False)
    pairs=[(f'{a}__ensemble',f'{b}__ensemble',label,'designated_primary_exploratory' if (a,b)==('WD','W') else 'exploratory') for a,b,label in COMPARISONS]
    for group in FEATURE_SETS:
        for ref in [c for c in cols if c.startswith(group+'__') and c!=group+'__ensemble']+['baseline__training_mean','baseline__training_median']:
            pairs.append((group+'__ensemble',ref,'Weighted ensemble versus comparator','exploratory'))
    paired=[]
    for a,b,label,role in pairs:
        for m in ['MAE','RMSE','R2']:
            delta=boots[a][m]-boots[b][m]; finite=delta[np.isfinite(delta)]
            ci=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
            paired.append({'first':a,'reference':b,'comparison':label,'role':role,'metric':m,
                'delta_first_minus_reference':estimates[a][m]-estimates[b][m],'ci_low':ci[0],'ci_high':ci[1], 'finite_draws':len(finite)})
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False)
    calibration=[]; strata=[]
    for group in FEATURE_SETS:
        col=group+'__ensemble'; p=pred[col].to_numpy(float)
        slope,intercept=np.polyfit(p,y,1) if np.ptp(p)>1e-12 else (np.nan,np.nan)
        calibration.append({'model':col,'n':len(y),'intercept_observed_on_predicted':intercept,'slope_observed_on_predicted':slope,
                            'predicted_SD_divided_by_observed_SD':np.std(p,ddof=1)/np.std(y,ddof=1) if np.std(y)>0 else np.nan})
        for name,mask in [('Recorded HbA1c <6.5%',y<6.5),('Recorded HbA1c >=6.5%',y>=6.5),('Recorded HbA1c >=8%',y>=8)]:
            strata.append({'model':col,'descriptive_stratum':name,'n':int(mask.sum()),
                'MAE':float(np.abs(p[mask]-y[mask]).mean()) if mask.any() else np.nan,
                'bias_predicted_minus_observed':float((p[mask]-y[mask]).mean()) if mask.any() else np.nan})
        fig,axes=plt.subplots(1,3,figsize=(13,4)); lo=min(y.min(),p.min())-.2; hi=max(y.max(),p.max())+.2
        axes[0].scatter(y,p,s=20,alpha=.6,color='#287d9a',edgecolors='none'); axes[0].plot([lo,hi],[lo,hi],'--',c='#77828c')
        axes[0].set(xlabel='Observed HbA1c (%)',ylabel='Predicted HbA1c (%)',title='Observed versus predicted',xlim=(lo,hi),ylim=(lo,hi))
        axes[1].scatter(p,y,s=18,alpha=.18,color='#287d9a')
        q=pd.qcut(pd.Series(p),q=min(5,len(p)),duplicates='drop')
        bins=pd.DataFrame({'p':p,'y':y,'bin':q}).groupby('bin',observed=True)[['p','y']].mean()
        axes[1].plot(bins.p,bins.y,'o-',color='#b45767'); axes[1].plot([lo,hi],[lo,hi],'--',c='#77828c')
        axes[1].set(xlabel='Predicted HbA1c (%)',ylabel='Observed HbA1c (%)',title='Descriptive calibration')
        axes[2].scatter(y,p-y,s=20,alpha=.6,color='#287d9a',edgecolors='none'); axes[2].axhline(0,ls='--',c='#77828c')
        axes[2].set(xlabel='Observed HbA1c (%)',ylabel='Prediction − observation (pp)',title='Below zero: underestimation')
        for ax in axes: ax.spines[['top','right']].set_visible(False)
        fig.suptitle(GROUP_LABELS[group]+' | '+title,fontsize=12); fig.tight_layout()
        save_figure(fig,out/f'{group}_calibration')
    pd.DataFrame(calibration).to_csv(out/'calibration_diagnostics.csv',index=False)
    pd.DataFrame(strata).to_csv(out/'descriptive_error_by_recorded_hba1c.csv',index=False)
    (out/'interpretation.txt').write_text('All intervals condition on the saved predictions and exclude model-refitting/model-selection uncertainty. OOF bootstrap stratifies by outer fold but does not remove dependence caused by overlapping training sets.\nMAE/RMSE: negative paired differences favor first model; R2: positive favors first.\nCalibration is descriptive, not a fitted correction. Strata were defined by observed HbA1c for error description, not eligibility or model selection.\nThe original test split was previously examined; revised split results are exploratory internal evidence.\n',encoding='utf-8')
    log(f'Reports saved: {out}')
    return pd.DataFrame(rows),pd.DataFrame(paired)


def selection_reports(table,out):
    check_analysis_table(table); folder=Path(out)/'selection_and_dictionary'; folder.mkdir(exist_ok=True)
    rows=[]; missing=[]
    for split,z in table.groupby('recommended_split'):
        # Both comparisons matter: all exclusions, and wearable selection among HbA1c-eligible people.
        for contrast,pool in [('All roster',z),('Visit-day HbA1c eligible',z.loc[z.visit_day_hba1c_eligible])]:
            for feature in C+['years_of_education','target_hba1c_percent']:
                a=pd.to_numeric(pool.loc[pool.eligible_common,feature],errors='coerce').dropna()
                b=pd.to_numeric(pool.loc[~pool.eligible_common,feature],errors='coerce').dropna()
                denom=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2) if min(len(a),len(b))>=2 else np.nan
                rows.append({'split':split,'comparison_population':contrast,'feature':feature,'label':LABELS.get(feature,'Recorded HbA1c'),
                    'included_n_observed':len(a),'excluded_n_observed':len(b),'included_mean':a.mean(),'excluded_mean':b.mean(),
                    'included_SD':a.std(),'excluded_SD':b.std(),'included_median':a.median(),'excluded_median':b.median(),
                    'included_Q1':a.quantile(.25),'included_Q3':a.quantile(.75),'excluded_Q1':b.quantile(.25),'excluded_Q3':b.quantile(.75),
                    'standardized_mean_difference':(a.mean()-b.mean())/denom if denom>0 else np.nan})
        for status,pool in [('Included',z.loc[z.eligible_common]),('Excluded',z.loc[~z.eligible_common])]:
            for f in ALL_FEATURES+['target_hba1c_percent']:
                missing.append({'split':split,'status':status,'feature':f,'label':LABELS.get(f,'Recorded HbA1c'),
                    'n':len(pool),'n_missing':int(pool[f].isna().sum()),'percent_missing':100*pool[f].isna().mean() if len(pool) else np.nan})
    pd.DataFrame(rows).to_csv(folder/'included_versus_excluded.csv',index=False)
    pd.DataFrame(missing).to_csv(folder/'missingness_by_selection_and_split.csv',index=False)
    dictionary=[]
    for f in ALL_FEATURES:
        domain='Clinical' if f in C else 'Wearable' if f in W else 'Diet' if f in D else 'SDH'
        dictionary.append({'feature':f,'publication_label':LABELS[f],'domain':domain,'model_representation':'nominal' if f in CATEGORICAL else 'numeric',
             'qualification':'Stored coding requires verification; do not assert confirmed years' if f=='years_of_education' else
             'Short descriptive label, not the full questionnaire wording' if f in D+S else ''})
    pd.DataFrame(dictionary).to_csv(folder/'feature_dictionary.csv',index=False)
    write_json(folder/'publication_labels.json',LABELS)
    # Compact exact-count flow figure: no invented sample sizes.
    flow=pd.read_csv(Path(out)/'cohort_flow.csv'); splits=['train','validation','test']
    fig,axes=plt.subplots(1,3,figsize=(13,7))
    from textwrap import fill
    for ax,split in zip(axes,splits):
        z=flow.loc[flow.split.eq(split)].reset_index(drop=True); ax.axis('off'); ax.set_title(split.capitalize(),fontweight='bold',color='#244B6B')
        for i,r in z.iterrows():
            yy=.9-i*.18
            text=fill(r['step'],28)+f"\nn = {int(r['n']):,}"
            if i: text+=f"  (−{int(z.loc[i-1,'n']-r['n']):,})"
            ax.text(.5,yy,text,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.65',fc='#eaf2f6',ec='#94b3c4'),transform=ax.transAxes)
            if i<len(z)-1: ax.annotate('',xy=(.5,yy-.105),xytext=(.5,yy-.065),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#55778c'))
    fig.suptitle('CGM-independent selection for recorded HbA1c estimation',fontsize=15,fontweight='bold',color='#244B6B')
    fig.tight_layout(); save_figure(fig,folder/'cohort_flow')
    return pd.DataFrame(rows)


def permutation_shapley(predict,background,X,n_permutations=2,seed=20260927,progress=True):
    """Monte Carlo interventional Shapley estimator with antithetic paths.
    All background rows participate equally. A reverse path accompanies each
    random permutation. Predictions only: no fitting, labels, or test background.
    Correlated-feature hybrids may be unrealistic; no conditional/causal claim.
    """
    if list(background.columns)!=list(X.columns): raise ValueError('Background/feature order mismatch.')
    if len(background)==0 or n_permutations<1: raise ValueError('Empty background or invalid permutations.')
    bg=background.to_numpy(object); xs=X.to_numpy(object); rng=np.random.default_rng(seed)
    n,p=xs.shape; values=np.zeros((n,p)); base=float(np.asarray(predict(background),float).mean())
    for i,x in enumerate(xs):
        paths=[]; orders=[]
        for _ in range(n_permutations):
            for b in bg:
                perm=rng.permutation(p)
                for order in [perm,perm[::-1]]:
                    path=np.tile(b,(p+1,1))
                    for k,f in enumerate(order): path[k+1]=path[k].copy(); path[k+1,f]=x[f]
                    paths.append(path); orders.append(order)
        combined=pd.DataFrame(np.vstack(paths),columns=X.columns)
        # Restore numeric dtypes expected by sklearn's numeric preprocessing.
        for f in X:
            if pd.api.types.is_numeric_dtype(X[f]): combined[f]=pd.to_numeric(combined[f],errors='raise')
        scores=np.asarray(predict(combined),float).reshape(len(paths),p+1)
        for d,order in zip(np.diff(scores,axis=1),orders): values[i,order]+=d/len(paths)
        if progress and (i==0 or (i+1)%20==0 or i==n-1): log(f'Feature attribution {i+1:,}/{n:,}')
    pred=np.asarray(predict(X),float)
    if not np.allclose(base+values.sum(axis=1),pred,atol=1e-7,rtol=1e-7):
        raise AssertionError('Shapley additivity check failed.')
    return values,base,pred


def explain_revision(cfg,table,out,split='validation',group='CWDS',background_n=32,n_permutations=2):
    verify_lock(out); check_analysis_table(table)
    if split not in ['validation','test'] or group not in FEATURE_SETS: raise ValueError('Invalid explanation split/group.')
    # Use locked saved inputs, not an unverified in-memory change.
    locked=load_prepared(out)
    features=FEATURE_SETS[group]
    pd.testing.assert_frame_equal(table[['person_id','recommended_split','eligible_common']+features].reset_index(drop=True),
                                  locked[['person_id','recommended_split','eligible_common']+features].reset_index(drop=True),check_dtype=False)
    train=table.loc[table.recommended_split.eq('train')&table.eligible_common]
    hold=table.loc[table.recommended_split.eq(split)&table.eligible_common]
    if len(hold)==0: raise ValueError('No eligible held-out participants.')
    bundle=joblib.load(Path(out)/'revised_frozen_models'/f'frozen_{group}.joblib')
    if set(hold.person_id)&set(bundle['training_ids']): raise ValueError('Explanation/training overlap.')
    bg=train.sample(n=min(background_n,len(train)),random_state=cfg.seed)
    X=hold[features].reset_index(drop=True)
    def predict(x): return bundle_predictions(bundle,x)['ensemble']
    dest=Path(out)/f'{split}_{group}_SHAP'; dest.mkdir(exist_ok=False)
    values,base,pred=permutation_shapley(predict,bg[features],X,n_permutations,cfg.seed)
    result=pd.DataFrame(values,columns=features); result.insert(0,'person_id',hold.person_id.to_numpy())
    result.to_csv(dest/'SHAP_values_LOCAL_ONLY.csv',index=False)
    hold[['person_id']+features].to_csv(dest/'SHAP_features_LOCAL_ONLY.csv',index=False)
    bg[['person_id']].to_csv(dest/'background_ids_LOCAL_ONLY.csv',index=False)
    write_json(dest/'SHAP_metadata.json',{'method':'Monte Carlo interventional Shapley values; random and reverse permutation paths',
        'group':group,'split':split,'n_explained':len(hold),'n_background_training_only':len(bg),'permutations_per_background':n_permutations,
        'seed':cfg.seed,'expected_prediction':base,'all_eligible_split_participants_explained':True,
        'max_additivity_error':float(np.abs(base+values.sum(axis=1)-pred).max()),'analysis_status':cfg.analysis_status,
        'uncertainty':'Monte Carlo attributions; additivity alone does not establish convergence of individual attributions.',
        'interpretation':'Model behavior, not association significance, clinical usefulness, or causality. Nominal values have no low/high color ordering.'})
    importance=pd.DataFrame({'feature':features,'Predictor':[LABELS[f] for f in features],
        'Mean absolute SHAP':np.mean(np.abs(values),axis=0),'Mean signed SHAP':values.mean(axis=0)}).sort_values('Mean absolute SHAP',ascending=False)
    importance.insert(0,'Rank',np.arange(1,len(importance)+1))
    importance.to_csv(dest/'SHAP_importance.csv',index=False)
    # Plain HTML avoids the pandas Styler/jinja2 dependency that failed previously.
    html='<html><meta charset="utf-8"><style>body{font-family:Arial;color:#244B6B;margin:32px}table{border-collapse:collapse}th{background:#244B6B;color:white;padding:12px}td{padding:9px;border-bottom:1px solid #dfe8ee}tr:nth-child(even){background:#f1f6f8}</style><h2>Feature attribution for recorded HbA1c estimation</h2>'
    html+=importance.drop(columns='feature').to_html(index=False,float_format=lambda x:f'{x:.4f}',escape=True)+'</html>'
    (dest/'SHAP_importance.html').write_text(html,encoding='utf-8')
    plot_shap(values,X,importance,dest)
    return importance


def plot_shap(values,X,importance,out,top=20):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    cmap=LinearSegmentedColormap.from_list('teal_rose',['#267c9e','#d8e4e8','#cb4569'])
    order=[list(X.columns).index(f) for f in importance.feature.iloc[:top]][::-1]
    fig,axes=plt.subplots(1,2,figsize=(14,max(7,len(order)*.34)),gridspec_kw={'width_ratios':[1.6,1]})
    labels=[LABELS[X.columns[j]] for j in order]; rng=np.random.default_rng(914)
    for y,j in enumerate(order):
        v=values[:,j]; spread=max(float(np.ptp(v)),1e-12)
        # Deterministic density stacking within narrow SHAP bins.
        bins=np.floor((v-v.min())/spread*55).astype(int); offsets=np.zeros(len(v))
        for b in np.unique(bins):
            ids_=np.flatnonzero(bins==b); rng.shuffle(ids_)
            offsets[ids_]=(np.arange(len(ids_))-(len(ids_)-1)/2)*min(.055,.7/max(1,len(ids_)))
        f=X.columns[j]
        if f in CATEGORICAL:
            axes[0].scatter(v,y+offsets,s=13,c='#8b91a1',alpha=.75,edgecolors='none')
        else:
            data=pd.to_numeric(X[f],errors='coerce').to_numpy(float); finite=np.isfinite(data)
            low,high=np.nanquantile(data,[.05,.95]) if finite.any() else (0,1)
            colors=np.clip((data-low)/max(high-low,1e-12),0,1)
            axes[0].scatter(v[finite],y+offsets[finite],s=13,c=colors[finite],cmap=cmap,vmin=0,vmax=1,alpha=.8,edgecolors='none')
            axes[0].scatter(v[~finite],y+offsets[~finite],s=13,c='#c0c3c8',alpha=.6,edgecolors='none')
    axes[0].set_yticks(range(len(order)),labels,fontsize=9); axes[0].axvline(0,color='#acb8c1',lw=.8)
    axes[0].set(xlabel='SHAP value (HbA1c percentage points)',title='A  Participant-level contributions')
    vals=np.mean(np.abs(values[:,order]),axis=0)
    axes[1].barh(range(len(order)),vals,color='#287f8f',height=.62); axes[1].set_yticks(range(len(order)),[])
    axes[1].set(xlabel='Mean absolute SHAP value',title='B  Average contribution magnitude')
    for ax in axes: ax.spines[['top','right','left']].set_visible(False); ax.grid(axis='x',alpha=.12); ax.set_axisbelow(True)
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),ax=axes[0],pad=.02,shrink=.5)
    cb.set_ticks([0,1],labels=['Low','High']); cb.set_label('Numeric feature value')
    fig.suptitle('Feature contributions to recorded HbA1c estimation',fontsize=16,fontweight='bold',color='#244B6B')
    fig.text(.5,.015,'Gray: nominal categories or missing values. Contributions describe the model, not causal effects.',ha='center',fontsize=9,color='#687885')
    fig.subplots_adjust(left=.29,right=.98,top=.9,bottom=.10,wspace=.12)
    save_figure(fig,Path(out)/'SHAP_beeswarm_and_importance')


def additional_quality_sensitivity(cfg,main_out):
    """Optional new post-test sensitivity; never described as previously frozen.
    Relax only scalar source coverage from >=5 days/span>=5 to >=4/span>=3.
    Same seven-day anchor, minute thresholds, sleep rules, and clinical outcome.
    Each revised model is refitted on sensitivity-eligible TRAINING people only.
    """
    from dataclasses import replace
    sensitivity=replace(cfg,min_good_days=4,min_day_span=3)
    out=new_run(sensitivity)
    write_json(out/'additional_analysis.json',{'parent_run':str(Path(main_out).resolve()),
        'label':'Additional quality-rule sensitivity defined after original test evaluation',
        'changes':{'min_good_days':4,'min_day_span':3},
        'interpretation':'Not prospectively frozen. Different cohorts can change marginal metrics; also report overlap comparisons.'})
    table=prepare(sensitivity,out); selection_reports(table,out); train_revision(sensitivity,table,out)
    for split in ['validation','test']:
        result=evaluate_revision(sensitivity,table,out,split)
        previous=Path(main_out)/f'{split}_exploratory_evaluation/predictions_LOCAL_ONLY.csv'
        if not previous.is_file(): continue
        original=pd.read_csv(previous,dtype={'person_id':str})
        common=original.merge(result,on='person_id',suffixes=('_main','_sensitivity'),validate='one_to_one')
        if not np.allclose(common.observed_hba1c_percent_main,common.observed_hba1c_percent_sensitivity):
            raise ValueError('Outcome disagreement in matched quality-rule comparison.')
        rows=[]
        for a,b,label in COMPARISONS:
            for rule in ['main','sensitivity']:
                y=common.observed_hba1c_percent_main.to_numpy(float)
                delta=np.abs(common[f'{a}__ensemble_{rule}'].to_numpy()-y)-np.abs(common[f'{b}__ensemble_{rule}'].to_numpy()-y)
                rng=np.random.default_rng(cfg.seed)
                boot=np.array([delta[rng.integers(0,len(y),len(y))].mean() for _ in range(cfg.bootstrap_draws)]) if len(y) else np.array([])
                ci=np.quantile(boot,[.025,.975]) if len(boot) else [np.nan,np.nan]
                rows.append({'split':split,'rule':rule,'comparison':label,'common_n':len(y),
                    'delta_MAE_added_minus_reference':float(delta.mean()) if len(y) else np.nan,'ci_low':ci[0],'ci_high':ci[1]})
        pd.DataFrame(rows).to_csv(out/f'{split}_matched_quality_comparisons.csv',index=False)
    return out
