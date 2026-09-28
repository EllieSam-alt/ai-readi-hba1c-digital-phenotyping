"""Synthetic tests only; no AI-READI participant data are distributed."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
import json
import numpy as np
import pandas as pd
from aireadi_no_cgm import *
from aireadi_no_cgm_models import (train_revision,evaluate_revision,selection_reports,
    explain_revision,permutation_shapley,verify_lock,load_prepared,additional_quality_sensitivity)


def fixture(root,n=48):
    ds=root/'dataset'; base=root/'analysis'; (ds/'clinical_data').mkdir(parents=True)
    (ds/'wearable_activity_monitor').mkdir(); (base/'expanded_predictor_inventory').mkdir(parents=True)
    (base/'staging/cleaning').mkdir(parents=True); (base/'staging/sdh').mkdir()
    rng=np.random.default_rng(45); roster=[]; clinical=[]; observation=[]; manifest=[]; hb=[]; sdh=[]
    inventory=[]
    for k,(code,name) in enumerate(CLINICAL_CODES.items()):
        inventory.append(dict(source='measurement',source_code=code,concept_id=str(100+k),source_label=code+', Synthetic source',unit_concept_id='1',unit_source_value='synthetic_verified_unit'))
    for i in range(n):
        pid=f'S{i:03d}'; visit=pd.Timestamp('2024-01-10',tz='UTC')
        split='train' if i<28 else 'validation' if i<38 else 'test'
        age=25+i%45; bmi=20+rng.random()*15
        roster.append(dict(person_id=pid,recommended_split=split,study_visit_date='2024-01-10',age=age))
        vals=[bmi,165,70,80+i%20,100,0.8,110+i%30,70+i%15,112+i%30,72+i%15]
        for m,v in zip(inventory,vals):
            clinical.append(dict(person_id=pid,measurement_concept_id=m['concept_id'],measurement_source_value=m['source_label'],
                value_as_number=v,unit_concept_id='1',unit_source_value='synthetic_verified_unit',measurement_date='2024-01-10'))
        for k in range(1,10):
            v=777 if i==3 and k==2 else (i+k)%3
            observation.append(dict(person_id=pid,observation_concept_id='900',observation_source_value=f'diet{k}, Synthetic question',
                value_as_number=v,value_source_value=str(v),observation_date='2024-01-10'))
        record={'person_id':pid,**{SIGNALS[s][0]:'' for s in SCALARS},'sleep_filepath':''}
        if i%16!=15:
            for s in ['heart_rate','oxygen_saturation']:
                _,listkey,valkey,unit=SIGNALS[s]; records=[]
                for d in range(7):
                    for minute in range(600 if s=='heart_rate' else 60):
                        t=visit+pd.Timedelta(days=d,hours=6,minutes=minute)
                        v=(60+age*.2+d*.4+rng.normal(0,.2)) if s=='heart_rate' else (94+i%5)
                        records.append({valkey:{'value':v,'unit':unit},'effective_time_frame':{'date_time':t.isoformat()}})
                p=ds/'wearable_activity_monitor'/f'{pid}_{s}.json'
                p.write_text(json.dumps({'body':{listkey:records}}))
                record[SIGNALS[s][0]]=str(p.relative_to(ds))
            sleep=[]
            for d in range(7):
                start=visit+pd.Timedelta(days=d)
                end=start+pd.Timedelta(hours=5+i%3+d*.03)
                sleep.append({'sleep_stage_state':'light','effective_time_frame':{'time_interval':{'start_date_time':start.isoformat(),'end_date_time':end.isoformat()}}})
            p=ds/'wearable_activity_monitor'/f'{pid}_sleep.json'; p.write_text(json.dumps({'body':{'sleep':sleep}}))
            record['sleep_filepath']=str(p.relative_to(ds))
        manifest.append(record)
        hb.append(dict(person_id=pid,hba1c_percent_unit_confirmed='' if i==1 else 4+.05*bmi+.003*age+rng.normal(0,.2),
            hba1c_days_from_visit=4 if i==2 else 0,hba1c_operator_concept_id=4172703))
        sdh.append({'person_id':pid,'years_of_education':12+i%7,'marital_status_source_code':['C51773','C51774'][i%2],
                    **{f:str(i%2) for f in S if f not in ['years_of_education','marital_status_source_code']}})
    pd.DataFrame(roster).to_csv(ds/'participants.tsv',sep='\t',index=False)
    pd.DataFrame(manifest).to_csv(ds/'wearable_activity_monitor/manifest.tsv',sep='\t',index=False)
    pd.DataFrame(clinical).to_csv(ds/'clinical_data/measurement.csv',index=False)
    pd.DataFrame(observation).to_csv(ds/'clinical_data/observation.csv',index=False)
    pd.DataFrame(inventory).to_csv(base/'expanded_predictor_inventory/TRAIN_all_clinical_variables.csv',index=False)
    pd.DataFrame(hb).to_csv(base/'staging/cleaning/hba1c_sources_UNIT_CHECKED.csv',index=False)
    pd.DataFrame(sdh).to_csv(base/'staging/sdh/sdh_predictor_candidates.csv',index=False)
    return Config(str(ds),str(base),software_test_only=True,outer_folds=2,inner_folds=2,bootstrap_draws=30)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(); cls.root=Path(cls.tmp.name)
        cls.cfg=fixture(cls.root); cls.out=new_run(cls.cfg); cls.table=prepare(cls.cfg,cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_no_cgm_files_or_dependency(self):
        self.assertFalse(any('cgm' in p.name.lower() for p in Path(self.cfg.dataset).rglob('*')))
        self.assertEqual(len(self.table),48)
        self.assertTrue(self.table.loc[self.table.person_id.eq('S000'),'eligible_common'].iloc[0])
        self.assertEqual(set(self.table.person_id),set(load_roster(self.cfg).person_id))
        self.assertEqual(int(self.table.eligible_common.sum()),43)
        self.assertTrue(self.table['oxygen_saturation__history_mean'].notna().any())
        self.assertTrue(self.table['sleep__slope_hours_per_day'].notna().any())
        self.assertTrue(self.table.loc[self.table.person_id.eq('S003'),'diet2__source_category'].isna().all())

    def test_02_hba1c_independent_windows(self):
        original=read(source_paths(self.cfg)['hba1c'])
        changed=original.copy(); changed['hba1c_percent_unit_confirmed']=12
        changed.to_csv(source_paths(self.cfg)['hba1c'],index=False)
        dest=self.root/'wearable_again'; dest.mkdir()
        other=wearable_features(self.cfg,load_roster(self.cfg),dest)
        saved=pd.read_csv(self.out/'all_roster_wearable_features_LOCAL_ONLY.csv',dtype={'person_id':str})
        pd.testing.assert_frame_equal(saved[['person_id']+W],other[['person_id']+W],check_dtype=False)
        self.assertEqual(saved.window_start_utc.fillna('').tolist(),other.window_start_utc.astype(str).replace('NaT','').tolist())
        original.to_csv(source_paths(self.cfg)['hba1c'],index=False)

    def test_03_actual_missing_and_invalid_text(self):
        a=numeric(pd.Series([None,pd.NA,'nan','',1.5]),'test')
        self.assertEqual(a.isna().sum(),4)
        with self.assertRaises(ValueError): numeric(pd.Series(['unexpected']),'test')
        with self.assertRaises(ValueError): ids(pd.DataFrame({'person_id':['A','A']}),'test')
        with self.assertRaises(ValueError): timestamp('2024-01-01 12:00:00',self.cfg)
        with self.assertRaises(FileNotFoundError): resolve_path('wearable_activity_monitor/moved.json',self.cfg.dataset)

    def test_04_conflicts_and_sleep_union(self):
        path=self.root/'conflict.json'
        path.write_text(json.dumps({'body':{'heart_rate':[
          {'heart_rate':{'value':x,'unit':'beats/min'},'effective_time_frame':{'date_time':t}}
          for x,t in [(70,'2024-01-01T00:00Z'),(70,'2024-01-01T00:00Z'),(80,'2024-01-01T00:00Z'),(65,'2024-01-01T00:01Z')]]}}))
        x,a=read_scalar(path,'heart_rate',self.cfg)
        self.assertEqual(len(x),1); self.assertEqual(a['duplicate_records'],1); self.assertEqual(a['conflicting_timestamp_rows'],2)
        path=self.root/'sleep.json'
        r=lambda stage,a,b:{'sleep_stage_state':stage,'effective_time_frame':{'time_interval':{'start_date_time':a,'end_date_time':b}}}
        rec=r('light','2024-01-01T00:00Z','2024-01-01T06:00Z')
        path.write_text(json.dumps({'body':{'sleep':[rec,rec,r('awake','2024-01-01T02:00Z','2024-01-01T03:00Z')]}}))
        ep,_,audit=read_sleep(path,self.cfg)
        self.assertEqual(len(ep),1); self.assertAlmostEqual(ep.sleep_hours.iloc[0],5)
        self.assertAlmostEqual(ep.unresolved_fraction.iloc[0],1/6)

    def test_05_shapley_additive_exactness(self):
        bg=pd.DataFrame({'a':[0.,2.],'b':[1.,3.]}); x=pd.DataFrame({'a':[3.,4.],'b':[5.,2.]})
        fn=lambda z:1+2*z.a.to_numpy()-3*z.b.to_numpy()
        values,base,pred=permutation_shapley(fn,bg,x,n_permutations=1,progress=False)
        np.testing.assert_allclose(values,(x-bg.mean()).to_numpy()*[2,-3])
        np.testing.assert_allclose(base+values.sum(axis=1),pred)

    def test_06_end_to_end_train_freeze_evaluate_explain(self):
        selection_reports(self.table,self.out)
        predictions=train_revision(self.cfg,self.table,self.out)
        self.assertEqual(predictions.person_id.nunique(),len(predictions))
        before={p.name:digest(p) for p in (self.out/'revised_frozen_models').glob('*.joblib')}
        val=evaluate_revision(self.cfg,self.table,self.out,'validation')
        test=evaluate_revision(self.cfg,self.table,self.out,'test')
        self.assertFalse(set(predictions.person_id)&set(test.person_id))
        self.assertFalse(set(predictions.person_id)&set(val.person_id))
        importance=explain_revision(self.cfg,self.table,self.out,'validation','CWDS',background_n=2,n_permutations=1)
        self.assertEqual(len(importance),48)
        self.assertEqual(before,{p.name:digest(p) for p in (self.out/'revised_frozen_models').glob('*.joblib')})
        verify_lock(self.out)
        altered=self.table.copy(); altered.loc[altered.recommended_split.eq('test'),'age_years']+=1
        with self.assertRaises(AssertionError): evaluate_revision(self.cfg,altered,self.out,'test')
        resumed=load_prepared(self.out)
        self.assertEqual(int(resumed.eligible_common.sum()),43)
        # Preserve ONLY synthetic diagnostic figures for package QA outside the temp fixture.
        qa=os.environ.get('AI_READI_SYNTHETIC_QA_DIR')
        if qa:
            import shutil
            d=Path(qa); d.mkdir(exist_ok=True,parents=True)
            for p in self.out.rglob('*.png'):
                if p.name in ['cohort_flow.png','SHAP_beeswarm_and_importance.png']:
                    shutil.copy2(p,d/p.name)

    def test_07_additional_sensitivity_entrypoint(self):
        out=additional_quality_sensitivity(self.cfg,self.out)
        for split in ['validation','test']:
            comparison=pd.read_csv(out/f'{split}_matched_quality_comparisons.csv')
            self.assertEqual(set(comparison.rule),{'main','sensitivity'})
            self.assertTrue(comparison.common_n.gt(0).all())
        self.assertTrue((out/'additional_analysis.json').is_file())
        main=pd.read_csv(self.out/'all_roster_wearable_features_LOCAL_ONLY.csv',dtype={'person_id':str})
        alt=pd.read_csv(out/'all_roster_wearable_features_LOCAL_ONLY.csv',dtype={'person_id':str})
        pd.testing.assert_series_equal(main.window_start_utc,alt.window_start_utc)
        pd.testing.assert_series_equal(main.window_end_utc,alt.window_end_utc)

if __name__=='__main__': unittest.main(verbosity=2)
