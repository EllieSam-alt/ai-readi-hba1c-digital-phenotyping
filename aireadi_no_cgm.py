"""CGM-independent, post-test redevelopment of recorded HbA1c estimation.
No CGM files, CGM outcome masks, prior common windows, or frozen models are used.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd

MISSING = {'', 'nan', 'na', 'n/a', 'none', 'null', '<na>'}
SCALARS = ['heart_rate', 'oxygen_saturation', 'respiratory_rate', 'stress']
SIGNALS = {
 'heart_rate': ('heartrate_filepath', 'heart_rate', 'heart_rate', 'beats/min'),
 'oxygen_saturation': ('oxygen_saturation_filepath', 'breathing', 'oxygen_saturation', '%'),
 'respiratory_rate': ('respiratory_rate_filepath', 'breathing', 'respiratory_rate', 'breaths/min'),
 'stress': ('stress_level_filepath', 'stress', 'stress', 'stress level'),
}
C = ['age_years', 'bmi_source', 'waist_source_cm', 'bp1_systolic_source', 'bp1_diastolic_source']
W = [f'{s}__{f}' for s in SCALARS for f in
     ['history_mean', 'slope_per_day', 'sd_daily_means', 'last_day_minus_prior_mean']]
W += ['sleep__mean_hours_per_episode', 'sleep__slope_hours_per_day',
      'sleep__sd_hours_per_episode', 'sleep__latest_minus_previous_mean']
D = [f'diet{i}__source_category' for i in range(1, 10)]
S = ['years_of_education', 'marital_status_source_code']
S += [f'pxfi{i}__source_category' for i in range(1, 6)]
S += ['pxhi1__source_category'] + [f'pxhic{i}__source_category' for i in [1,2,3,4,5,7]]
CATEGORICAL = set(D + S[1:])
ALL_FEATURES = C + W + D + S
FEATURE_SETS = {'W':W, 'WD':W+D, 'WDS':W+D+S, 'C':C, 'CD':C+D,
                'CW':C+W, 'CWD':C+W+D, 'CWDS':C+W+D+S}
COMPARISONS = [('WD','W','Diet added to wearables'), ('WDS','WD','SDH added to wearables and diet'),
 ('CD','C','Diet added to clinical measures'), ('CW','C','Wearables added to clinical measures'),
 ('CWD','CW','Diet added to clinical measures and wearables'),
 ('CWD','CD','Wearables added to clinical measures and diet'), ('CWDS','CWD','SDH added to full model')]
CLINICAL_CODES = {'bmi_vsorres':'bmi_source', 'height_vsorres':'height_source_cm',
 'weight_vsorres':'weight_source_kg', 'waist_vsorres':'waist_source_cm', 'hip_vsorres':'hip_source_cm',
 'whr_vsorres':'waist_hip_ratio_source', 'bp1_sysbp_vsorres':'bp1_systolic_source',
 'bp1_diabp_vsorres':'bp1_diastolic_source', 'bp2_sysbp_vsorres':'bp2_systolic_source',
 'bp2_diabp_vsorres':'bp2_diastolic_source'}

@dataclass(frozen=True)
class Config:
    dataset: str
    analysis_base: str
    history_days: int = 7
    start_search_days: int = 10
    min_observed_minutes_per_day: int = 600
    oxygen_min_observed_minutes_per_day: int = 60
    min_good_days: int = 5
    min_day_span: int = 5
    sleep_gap_minutes: int = 60
    sleep_min_episodes: int = 3
    sleep_min_span_days: float = 5.0
    sleep_max_recency_days: float = 2.0
    sleep_max_unresolved_fraction: float = 0.10
    expected_hba1c_operator: int = 4172703
    seed: int = 20260927
    outer_folds: int = 5
    inner_folds: int = 3
    bootstrap_draws: int = 2000
    # A day is elapsed UTC time, not a claim about local midnight or visit clock time.
    naive_timestamp_timezone: str | None = None
    analysis_status: str = 'post_test_revision_exploratory_internal_evaluation'
    software_test_only: bool = False


def log(x):
    print(x, flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, x):
    Path(path).write_text(json.dumps(x, indent=2, default=str, allow_nan=False), encoding='utf-8')


def read(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p, sep='\t' if p.suffix == '.tsv' else ',', dtype=str, keep_default_na=False)


def require(frame, cols, label):
    absent = set(cols)-set(frame)
    if absent:
        raise ValueError(f'{label}: missing columns {sorted(absent)}')


def ids(frame, label):
    require(frame, ['person_id'], label)
    frame = frame.copy()
    frame['person_id'] = frame.person_id.astype('string').str.strip()
    if frame.person_id.isna().any() or frame.person_id.eq('').any() or frame.person_id.duplicated().any():
        raise ValueError(f'{label}: missing or duplicate person_id; do not drop records arbitrarily.')
    return frame


def numeric(series, label):
    raw = series.astype('string').str.strip()
    missing = raw.isna() | raw.str.lower().isin(MISSING)
    result = pd.to_numeric(raw.mask(missing), errors='coerce')
    if ((~missing) & result.isna()).any():
        raise ValueError(f'{label}: unexpected nonnumeric representations; inspect locally.')
    result = pd.Series(result.to_numpy(dtype=float, na_value=np.nan), index=series.index)
    if np.isinf(result).any():
        raise ValueError(f'{label}: infinite values.')
    return result


def category(series):
    raw = series.astype('string').str.strip()
    raw = raw.mask(raw.isna() | raw.str.lower().isin(MISSING))
    # Normalize integral numeric categories; never impose an ordinal scale.
    raw = raw.str.replace(r'^(\d+)\.0+$', r'\1', regex=True)
    return raw.astype(object).where(raw.notna(), np.nan)


def calendar(series, label):
    raw = series.astype('string').str.strip()
    missing = raw.isna() | raw.str.lower().isin(MISSING)
    parsed = pd.to_datetime(raw.mask(missing), format='mixed', errors='coerce', utc=True)
    if ((~missing) & parsed.isna()).any():
        raise ValueError(f'{label}: unparseable dates.')
    return parsed.dt.normalize()


def timestamp(value, cfg):
    if value is None or str(value).strip().lower() in MISSING:
        return pd.NaT
    try:
        t = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise ValueError(f'Malformed wearable timestamp: {value!r}')
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        if cfg.naive_timestamp_timezone is None:
            raise ValueError('Timezone-naive wearable timestamp. Set naive_timestamp_timezone only from documented source timezone.')
        t = t.tz_localize(cfg.naive_timestamp_timezone, ambiguous='raise', nonexistent='raise')
    return t.tz_convert('UTC')


def new_run(cfg):
    root = Path(cfg.analysis_base) / 'hba1c_no_cgm_v1'
    out = root / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'protocol.json', {**asdict(cfg), 'CGM_used':False,
        'window_rule':'First usable wearable timestamp in [visit calendar day UTC, visit + start_search_days); seven elapsed 24-hour blocks.',
        'outcome':'Recorded unit-confirmed laboratory HbA1c on the study visit calendar day.',
        'common_cohort':'Visit-day HbA1c and at least one quality-passing wearable source; missing diet/SDH never excludes.',
        'education_note':'Stored numeric education is retained to isolate the CGM change. Coding/units require source-codebook verification before interpretation.',
        'test_note':'The original test split has already informed this project. This revision has no untouched test split.',
        'sensitivity_note':'Any new sensitivity is additional post-test analysis, not retrospectively prespecified.'})
    versions = {p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib']}
    write_json(out/'environment.json', {'python':platform.python_version(), **versions})
    return out


def source_paths(cfg):
    base, ds = Path(cfg.analysis_base), Path(cfg.dataset)
    return {'roster':ds/'participants.tsv', 'manifest':ds/'wearable_activity_monitor/manifest.tsv',
      'measurement':ds/'clinical_data/measurement.csv', 'observation':ds/'clinical_data/observation.csv',
      'clinical_mapping':base/'expanded_predictor_inventory/TRAIN_all_clinical_variables.csv',
      'hba1c':base/'staging/cleaning/hba1c_sources_UNIT_CHECKED.csv',
      'sdh':base/'staging/sdh/sdh_predictor_candidates.csv'}


def load_roster(cfg):
    r = ids(read(source_paths(cfg)['roster']), 'full participant roster')
    require(r, ['recommended_split','study_visit_date','age'], 'roster')
    r['recommended_split'] = r.recommended_split.str.strip().str.lower().replace({'training':'train','val':'validation','valid':'validation','testing':'test'})
    if not r.recommended_split.isin(['train','validation','test']).all():
        raise ValueError('Unrecognized split labels.')
    r['visit_day_utc'] = calendar(r.study_visit_date, 'study_visit_date')
    r['age_years'] = numeric(r.age, 'age').where(lambda s:s>0)
    return r


def require_full_roster(table, roster, label):
    absent = set(roster.person_id)-set(table.person_id)
    if absent:
        raise ValueError(f'{label}: {len(absent)} roster IDs absent. Use the all-participant cleaned source, not an old selected cohort.')


def resolve_path(raw, dataset):
    if pd.isna(raw) or str(raw).strip().lower() in MISSING:
        return None
    value = str(raw).strip().strip('"').replace('\\','/')
    ds = Path(dataset)
    paths = [Path(value), ds/value.lstrip('/'), ds/'wearable_activity_monitor'/value.lstrip('/')]
    marker = 'wearable_activity_monitor/'
    if marker in value:
        paths.append(ds/marker/value.split(marker,1)[1])
    found = list(dict.fromkeys(p.resolve() for p in paths if p.is_file()))
    if len(found)>1:
        raise ValueError(f'Ambiguous wearable path: {raw}')
    if not found:
        # A referenced but absent raw file is a configuration/data-transfer failure.
        raise FileNotFoundError(f'Referenced Garmin file unavailable: {raw}. Restore it before running; do not treat a moved file as missing biology.')
    return found[0]


def read_scalar(path, signal, cfg):
    audit = {'raw_records':0, 'invalid_value_or_missing_time':0, 'duplicate_records':0, 'conflicting_timestamp_rows':0}
    if path is None:
        return pd.DataFrame(columns=['time','value']), audit
    _, list_key, value_key, expected_unit = SIGNALS[signal]
    doc = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(doc.get('body',{}).get(list_key),list):
        raise ValueError(f'{path}: unexpected JSON body for {signal}')
    rows=[]
    for rec in doc['body'][list_key]:
        audit['raw_records']+=1
        payload=rec.get(value_key)
        if not isinstance(payload,dict):
            raise ValueError(f'{path}: missing {value_key} payload.')
        unit=str(payload.get('unit','')).strip().lower()
        if unit != expected_unit:
            raise ValueError(f'{path}: {signal} unexpected unit {unit!r}; expected {expected_unit!r}')
        raw_value = payload.get('value')
        if raw_value is None or str(raw_value).strip().lower() in MISSING:
            v = np.nan
        else:
            try: v = float(raw_value)
            except (TypeError, ValueError): raise ValueError(f'{path}: unexpected nonnumeric {signal} value.')
            if np.isinf(v): raise ValueError(f'{path}: infinite {signal} value.')
        t = timestamp(rec.get('effective_time_frame',{}).get('date_time'), cfg)
        valid = np.isfinite(v) and not pd.isna(t)
        valid = valid and ((0<=v<=100) if signal=='stress' else (0<v<=100) if signal=='oxygen_saturation' else v>0)
        if not valid:
            audit['invalid_value_or_missing_time']+=1
            continue
        rows.append((t,float(v)))
    x=pd.DataFrame(rows,columns=['time','value'])
    before=len(x); x=x.drop_duplicates(['time','value']); audit['duplicate_records']=before-len(x)
    conflict=x.duplicated('time',keep=False); audit['conflicting_timestamp_rows']=int(conflict.sum())
    return x.loc[~conflict].sort_values('time').reset_index(drop=True), audit


def read_sleep(path,cfg):
    columns=['start','end','sleep_hours','elapsed_hours','unresolved_fraction']
    audit={'raw_records':0,'invalid_intervals':0,'duplicate_records':0,'episodes':0}
    if path is None:
        return pd.DataFrame(columns=columns), [], audit
    doc=json.loads(path.read_text(encoding='utf-8-sig'))
    records=doc.get('body',{}).get('sleep')
    if not isinstance(records,list):
        raise ValueError(f'{path}: unexpected sleep schema.')
    intervals=[]
    for rec in records:
        audit['raw_records']+=1
        frame=rec.get('effective_time_frame') or rec.get('sleep_stage_time_frame') or {}
        ti=frame.get('time_interval',{})
        a=timestamp(ti.get('start_date_time'),cfg); b=timestamp(ti.get('end_date_time'),cfg)
        if pd.isna(a) or pd.isna(b) or b<=a:
            audit['invalid_intervals']+=1; continue
        stage=str(rec.get('sleep_stage_state','')).strip().lower()
        if stage not in {'light','deep','rem','awake'}: stage='unknown'
        intervals.append((a,b,stage))
    unique=set(intervals); audit['duplicate_records']=len(intervals)-len(unique)
    events=defaultdict(Counter)
    for a,b,s in unique:
        events[a][s]+=1; events[b][s]-=1
    times=sorted(events); active=Counter(); segments=[]
    for i,t in enumerate(times[:-1]):
        active.update(events[t]); labels=[s for s,n in active.items() if n>0]
        if labels:
            segments.append((t,times[i+1],labels[0] if len(labels)==1 else 'conflict'))
    groups=[]
    for seg in segments:
        if not groups or (seg[0]-groups[-1][-1][1]).total_seconds()>cfg.sleep_gap_minutes*60:
            groups.append([])
        groups[-1].append(seg)
    rows=[]
    for group in groups:
        a,b=group[0][0],group[-1][1]; elapsed=(b-a).total_seconds()/3600
        durations=defaultdict(float)
        for st,en,s in group: durations[s]+=(en-st).total_seconds()/3600
        recorded=sum(durations.values()); asleep=sum(durations[s] for s in ['light','deep','rem'])
        unresolved=max(0,elapsed-recorded)+durations['unknown']+durations['conflict']
        rows.append((a,b,asleep,elapsed,unresolved/elapsed))
    audit['episodes']=len(rows)
    # Only unambiguous recognized segments may anchor a history.
    anchor_times=[a for a,b,s in segments if s in {'light','deep','rem','awake'}]
    return pd.DataFrame(rows,columns=columns), anchor_times, audit


def wearable_features(cfg, roster, out):
    manifest=ids(read(source_paths(cfg)['manifest']),'Garmin manifest')
    require(manifest,[x[0] for x in SIGNALS.values()]+['sleep_filepath'],'manifest')
    if not set(manifest.person_id).issubset(set(roster.person_id)):
        raise ValueError('Manifest includes IDs outside the supplied full roster.')
    manifest=manifest.set_index('person_id'); rows=[]; status=[]; daily=[]; episodes=[]; source_hashes=[]
    for ix,person in enumerate(roster.itertuples(index=False),1):
        pid=person.person_id
        row={'person_id':pid, **{f:np.nan for f in W}, 'window_start_utc':pd.NaT,'window_end_utc':pd.NaT,
             'wearable_sources_passing':0,'has_selected_window':False, 'window_status':'no_valid_near_visit_wearable'}
        scalar={}; local_audits={}; anchors=[]
        if pid not in manifest.index:
            row['window_status']='absent_from_wearable_manifest'
        for signal in SCALARS+['sleep']:
            col=SIGNALS[signal][0] if signal!='sleep' else 'sleep_filepath'
            raw=manifest.loc[pid,col] if pid in manifest.index else ''
            path=resolve_path(raw,cfg.dataset)
            if path:
                source_hashes.append({'person_id':pid,'signal':signal,'path':str(path),'sha256':digest(path)})
            if signal=='sleep':
                ep, stimes, audit=read_sleep(path,cfg); anchors.extend(stimes)
            else:
                scalar[signal],audit=read_scalar(path,signal,cfg)
                anchors.extend(scalar[signal].time.tolist())
            local_audits[signal]={**audit,'file_available':path is not None,
                'status':'missing_manifest_path' if path is None else 'no_near_visit_window', 'quality_pass':False}
        visit=person.visit_day_utc
        if not pd.isna(visit):
            candidates=[t for t in anchors if visit<=t<visit+pd.Timedelta(days=cfg.start_search_days)]
        else:
            candidates=[]; row['window_status']='missing_visit_date'
        if candidates:
            start=min(candidates); end=start+pd.Timedelta(days=cfg.history_days)
            row.update(window_start_utc=start,window_end_utc=end,has_selected_window=True,window_status='selected')
            for s,x in scalar.items():
                x=x.loc[(x.time>=start)&(x.time<end)].copy()
                if len(x):
                    x['day']=((x.time-start).dt.total_seconds()/86400).astype(int)
                    x['minute']=((x.time-start).dt.total_seconds()/60).astype(int)
                gooddays=[]; vals=[]
                for d in range(cfg.history_days):
                    z=x.loc[x.day.eq(d)] if len(x) else x
                    nmin=z.minute.nunique() if len(z) else 0
                    mean=float(z.value.mean()) if len(z) else np.nan
                    good=nmin >= (cfg.oxygen_min_observed_minutes_per_day if s=='oxygen_saturation' else cfg.min_observed_minutes_per_day)
                    daily.append({'person_id':pid,'signal':s,'day':d,'observed_minutes':nmin,
                                  'mean':mean,'quality_pass':good})
                    if good: gooddays.append(d); vals.append(mean)
                passed=len(gooddays)>=cfg.min_good_days and (max(gooddays)-min(gooddays)>=cfg.min_day_span)
                local_audits[s].update(good_days=len(gooddays),quality_pass=bool(passed),
                    status='quality_pass' if passed else 'insufficient_wearable_coverage')
                if passed:
                    row['wearable_sources_passing']+=1
                    row[f'{s}__history_mean']=float(np.mean(vals))
                    row[f'{s}__sd_daily_means']=float(np.std(vals,ddof=1))
                    row[f'{s}__slope_per_day']=float(np.polyfit(gooddays,vals,1)[0])
                    if gooddays[-1]==cfg.history_days-1:
                        row[f'{s}__last_day_minus_prior_mean']=float(vals[-1]-np.mean(vals[:-1]))
            if len(ep):
                selected=(ep.start>=start)&(ep.end<=end)
                quality=(ep.elapsed_hours<=24)&(ep.sleep_hours>0)&(ep.unresolved_fraction<=cfg.sleep_max_unresolved_fraction)
                ep=ep.assign(person_id=pid,fully_contained=selected,episode_quality_pass=quality)
                episodes.extend(ep.to_dict('records'))
                z=ep.loc[selected&quality].sort_values('end')
            else: z=ep
            ends=np.array([(e-start).total_seconds()/86400 for e in z.end])
            passed=(len(z)>=cfg.sleep_min_episodes and np.ptp(ends)>=cfg.sleep_min_span_days and
                    cfg.history_days-ends[-1]<=cfg.sleep_max_recency_days)
            local_audits['sleep'].update(accepted_episodes=len(z),quality_pass=bool(passed),
                status='quality_pass' if passed else 'insufficient_sleep_coverage')
            if passed:
                v=z.sleep_hours.to_numpy(float); row['wearable_sources_passing']+=1
                row['sleep__mean_hours_per_episode']=float(v.mean())
                row['sleep__sd_hours_per_episode']=float(v.std(ddof=1))
                row['sleep__slope_hours_per_day']=float(np.polyfit(ends,v,1)[0])
                # Require a unique latest end time for a defensible latest episode contrast.
                if len(ends)==1 or ends[-1]>ends[-2]:
                    row['sleep__latest_minus_previous_mean']=float(v[-1]-v[:-1].mean())
        rows.append(row)
        for s,audit in local_audits.items(): status.append({'person_id':pid,'signal':s,**audit})
        if ix==1 or ix%100==0 or ix==len(roster): log(f'Wearable preparation {ix:,}/{len(roster):,}')
    for name,records in [('wearable_source_status_LOCAL_ONLY',status),('daily_wearable_audit_LOCAL_ONLY',daily),
                          ('sleep_episode_audit_LOCAL_ONLY',episodes),('wearable_file_hashes_LOCAL_ONLY',source_hashes)]:
        pd.DataFrame(records).to_csv(out/f'{name}.csv',index=False)
    result=pd.DataFrame(rows)
    result.to_csv(out/'all_roster_wearable_features_LOCAL_ONLY.csv',index=False)
    return result


def static_features(cfg,roster,out):
    paths=source_paths(cfg); inventory=read(paths['clinical_mapping'])
    keys=['concept_id','source_label','unit_concept_id','unit_source_value']
    require(inventory,['source']+keys,'clinical mapping inventory')
    inventory['source_code']=inventory.source_label.str.split(',',n=1).str[0].str.strip().str.lower()
    lookup=inventory.loc[inventory.source_code.isin(CLINICAL_CODES),['source','source_code']+keys].drop_duplicates()
    if set(CLINICAL_CODES)-set(lookup.source_code):
        raise ValueError('Clinical mapping inventory lacks expected source codes.')
    if lookup.duplicated(['source']+keys).any(): raise ValueError('Ambiguous clinical mapping.')
    lookup.to_csv(out/'clinical_mapping_used.csv',index=False)
    parts=[]; diets=[]; timing=[]
    for source in ['measurement','observation']:
        file=paths[source]; headers=pd.read_csv(file,nrows=0).columns
        label=f'{source}_source_value'; concept=f'{source}_concept_id'
        needed=['person_id',label,concept,'value_as_number']
        require(pd.DataFrame(columns=headers),needed,str(file))
        optional=['unit_concept_id','unit_source_value','value_source_value',f'{source}_date']
        for chunk in pd.read_csv(file,dtype=str,keep_default_na=False,chunksize=100000,
                                 usecols=needed+[c for c in optional if c in headers]):
            chunk=chunk.loc[chunk.person_id.isin(set(roster.person_id))].copy()
            chunk['source_code']=chunk[label].str.split(',',n=1).str[0].str.strip().str.lower()
            selected=chunk.loc[chunk.source_code.isin(CLINICAL_CODES)].copy()
            for c in optional:
                if c not in selected: selected[c]=''
            selected=selected.rename(columns={label:'source_label',concept:'concept_id'})
            if len(selected):
                selected=selected.merge(lookup.loc[lookup.source.eq(source),keys],on=keys,how='left',indicator=True,validate='many_to_one')
                if selected._merge.ne('both').any():
                    selected.loc[selected._merge.ne('both')].to_csv(out/'UNRECOGNIZED_CLINICAL_MAPPING_LOCAL_ONLY.csv',index=False)
                    raise ValueError('New clinical mapping/units found. Review saved audit; do not silently discard these records.')
                selected['source']=source; parts.append(selected)
            if source=='observation':
                z=chunk.loc[chunk.source_code.isin([f'diet{i}' for i in range(1,10)])].copy()
                if 'value_source_value' not in z: z['value_source_value']=''
                diets.append(z)
    if not parts: raise ValueError('No clinical measurements matched.')
    raw=pd.concat(parts,ignore_index=True)
    raw['value']=numeric(raw.value_as_number,'clinical values')
    # Distinct repeated values are ambiguous; dates are retained in an audit.
    for code,group in raw.groupby('source_code'):
        if (group.groupby('person_id').value.nunique(dropna=True)>1).any():
            raw.to_csv(out/'CONFLICTING_CLINICAL_RECORDS_LOCAL_ONLY.csv',index=False)
            raise ValueError(f'Conflicting clinical values for {code}; resolve source visit mapping before modeling.')
    raw['value']=raw.value.where(raw.value>0)
    raw.to_csv(out/'clinical_extraction_audit_LOCAL_ONLY.csv',index=False)
    values=raw.groupby(['person_id','source_code']).value.first().unstack().rename(columns=CLINICAL_CODES)
    result=roster[['person_id','age_years']].merge(values,left_on='person_id',right_index=True,how='left',validate='one_to_one')
    if not diets or not sum(map(len,diets)): raise ValueError('No diet item records found in the original observations.')
    diet=pd.concat(diets,ignore_index=True)
    a=numeric(diet.value_as_number,'diet numeric'); b=pd.to_numeric(diet.value_source_value,errors='coerce')
    conflict=a.notna()&b.notna()&a.ne(b)
    if conflict.any(): raise ValueError('Diet numeric/source-value disagreement.')
    value=a.fillna(b).mask(lambda s:s.eq(777))
    if (value.notna()&~value.isin([0,1,2])).any(): raise ValueError('Unexpected diet codes; expected 0, 1, 2, missing, or 777.')
    fallback=diet.value_source_value.astype('string').str.strip()
    bad=a.isna()&b.isna()&~fallback.str.lower().isin(MISSING)
    if bad.any(): raise ValueError('Unrecognized diet text without numeric representation.')
    diet['category']=value.map(lambda v:str(int(v)) if pd.notna(v) else np.nan)
    if (diet.groupby(['person_id','source_code']).category.nunique(dropna=True)>1).any():
        diet.to_csv(out/'CONFLICTING_DIET_RECORDS_LOCAL_ONLY.csv',index=False)
        raise ValueError('Conflicting diet responses per participant/item; review source dates.')
    diet.to_csv(out/'diet_extraction_audit_LOCAL_ONLY.csv',index=False)
    wide=diet.groupby(['person_id','source_code']).category.first().unstack()
    wide=wide.rename(columns={f'diet{i}':f'diet{i}__source_category' for i in range(1,10)})
    result=result.merge(wide,left_on='person_id',right_index=True,how='left',validate='one_to_one')
    for f in D:
        if f not in result: result[f]=np.nan
    sdh=ids(read(paths['sdh']),'all-roster cleaned SDH'); require(sdh,S,'SDH')
    require_full_roster(sdh,roster,'SDH')
    for f in S: sdh[f]=numeric(sdh[f],f) if f=='years_of_education' else category(sdh[f])
    result=result.merge(sdh[['person_id']+S],on='person_id',how='left',validate='one_to_one')
    # Audit collection dates relative to visit; no new date-driven selection is imposed.
    for label,df in [('clinical',raw),('diet',diet)]:
        for col in [c for c in df if c.endswith('_date')]:
            z=df[['person_id','source_code',col]].merge(roster[['person_id','visit_day_utc']],on='person_id',validate='many_to_one')
            z['source_date']=calendar(z[col],col)
            z['days_from_visit']=(z.source_date-z.visit_day_utc).dt.days
            z['domain']=label; timing.append(z[['person_id','source_code','source_date','days_from_visit','domain']])
    if timing: pd.concat(timing,ignore_index=True).to_csv(out/'static_predictor_timing_LOCAL_ONLY.csv',index=False)
    return result


def prepare(cfg,out):
    out=Path(out); paths=source_paths(cfg)
    # No fallback to a prior selected participant table is allowed.
    for p in paths.values():
        if not p.is_file(): raise FileNotFoundError(p)
    write_json(out/'input_hashes.json',{k:{'path':str(p),'sha256':digest(p)} for k,p in paths.items()})
    roster=load_roster(cfg)
    log(f'Full roster: {len(roster):,}; no CGM-based filtering.')
    static=static_features(cfg,roster,out)
    wear=wearable_features(cfg,roster,out)
    hb=ids(read(paths['hba1c']),'unit-checked all-roster HbA1c')
    cols=['hba1c_percent_unit_confirmed','hba1c_days_from_visit','hba1c_operator_concept_id']
    require(hb,cols,'HbA1c'); require_full_roster(hb,roster,'HbA1c')
    for c in cols: hb[c]=numeric(hb[c],c)
    observed=hb.hba1c_percent_unit_confirmed.notna()
    if observed.any() and hb.loc[observed,'hba1c_percent_unit_confirmed'].isin([0,1]).all():
        raise ValueError('HbA1c column looks like a flag rather than percent-valued measurements.')
    if (observed&hb.hba1c_operator_concept_id.ne(cfg.expected_hba1c_operator)).any():
        raise ValueError('HbA1c operator differs from verified original pipeline; inspect source representation first.')
    hb=hb[['person_id']+cols].rename(columns={'hba1c_percent_unit_confirmed':'target_hba1c_percent'})
    joined=roster.drop(columns=['age_years']).merge(static,on='person_id',how='left',validate='one_to_one')
    joined=joined.merge(wear,on='person_id',how='left',validate='one_to_one').merge(hb,on='person_id',how='left',validate='one_to_one')
    joined['laboratory_hba1c_available']=joined.target_hba1c_percent.gt(0)
    joined['visit_day_hba1c_eligible']=joined.laboratory_hba1c_available & joined.hba1c_days_from_visit.eq(0) & joined.visit_day_utc.notna()
    joined['any_quality_wearable']=joined.wearable_sources_passing.gt(0)
    joined['eligible_common']=joined.visit_day_hba1c_eligible & joined.any_quality_wearable
    joined['exclusion_reason']=np.select([
        ~joined.laboratory_hba1c_available,joined.visit_day_utc.isna(),~joined.hba1c_days_from_visit.eq(0),
        ~joined.has_selected_window,~joined.any_quality_wearable],
        ['no_positive_unit_checked_hba1c','missing_visit_date','hba1c_not_visit_day','no_near_visit_wearable_window','no_quality_passing_wearable'],default='included')
    for f in ALL_FEATURES:
        joined[f]=category(joined[f]) if f in CATEGORICAL else numeric(joined[f],f)
    if len(joined)!=len(roster): raise AssertionError('Roster rows changed during preparation.')
    joined.to_csv(out/'all_roster_analysis_status_LOCAL_ONLY.csv',index=False)
    flow=[]
    for split,z in joined.groupby('recommended_split',sort=False):
        for step,mask in [('Roster',np.ones(len(z),bool)),('Positive unit-checked laboratory HbA1c',z.laboratory_hba1c_available),
          ('Visit-day HbA1c and known visit date',z.visit_day_hba1c_eligible),
          ('Plus near-visit wearable window',z.visit_day_hba1c_eligible&z.has_selected_window),
          ('Plus any quality-passing wearable: common cohort',z.eligible_common)]:
            flow.append({'split':split,'step':step,'n':int(np.sum(mask))})
        z.loc[z.eligible_common].to_csv(out/f'{split}_analysis_cohort_LOCAL_ONLY.csv',index=False)
    pd.DataFrame(flow).to_csv(out/'cohort_flow.csv',index=False)
    joined.groupby(['recommended_split','exclusion_reason']).size().rename('n').to_csv(out/'exclusion_reasons.csv')
    log(pd.DataFrame(flow).to_string(index=False))
    log('CGM-independent preparation complete. Outcomes were not imputed; models have not been fitted.')
    write_json(out/'preparation_complete.json',{'rows_preserved':len(joined),'CGM_used':False,
               'table_sha256':digest(out/'all_roster_analysis_status_LOCAL_ONLY.csv')})
    return joined
