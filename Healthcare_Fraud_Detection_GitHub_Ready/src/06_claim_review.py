"""Retrospective provider-label proxy; claim-specific review index is not a fraud probability."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

def run(data,out):
    out.mkdir(parents=True,exist_ok=True)
    dx=[f'ClmDiagnosisCode_{i}' for i in range(1,11)]
    px=[f'ClmProcedureCode_{i}' for i in range(1,7)]
    cols=['Provider','ClaimID','ClaimStartDt']+dx+px
    c=pd.concat([pd.read_csv(next(data.glob('Train_'+k+'*.csv')),usecols=cols,low_memory=False).assign(ip=int(k=='Inpatientdata')) for k in ['Inpatientdata','Outpatientdata']],ignore_index=True)
    c['date']=pd.to_datetime(c.ClaimStartDt); c['dx_count']=c[dx].notna().sum(axis=1); c['px_count']=c[px].notna().sum(axis=1); c=c.drop(columns=dx+px+['ClaimStartDt'])
    assert c.ClaimID.is_unique
    y=pd.read_csv(next(data.glob('Train-*.csv'))).set_index('Provider').PotentialFraud.map({'Yes':1,'No':0})
    tr,h=train_test_split(y.index,test_size=.4,random_state=42,stratify=y); va,te=train_test_split(h,test_size=.5,random_state=42,stratify=y.loc[h]); assert not(set(tr)&set(te) or set(va)&set(te) or set(tr)&set(va))
    d=c.groupby(['Provider','date']).agg(n=('ClaimID','size'),ip=('ip','sum')).reset_index().sort_values(['Provider','date'])
    for f in ['n','ip']: d['prior_'+f]=d.groupby('Provider')[f].cumsum()-d[f]
    d['prior_days']=d.groupby('Provider').cumcount(); d['prior_ip_share']=d.prior_ip/d.prior_n.replace(0,np.nan)
    fs=['prior_n','prior_ip','prior_days','prior_ip_share']
    def snap(ids,date):
        z=d[d.Provider.isin(ids)&(d.date<pd.Timestamp(date))].groupby('Provider').tail(1).set_index('Provider').copy(); z.prior_n+=z.n; z.prior_ip+=z.ip; z.prior_days+=1; z.prior_ip_share=z.prior_ip/z.prior_n
        x=z[fs].reindex(ids); x[fs[:3]]=x[fs[:3]].fillna(0); return x
    m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,random_state=42).fit(snap(tr,'2009-07-01'),y.loc[tr]); rows=[]
    for date in ['2009-09-01','2009-10-01','2009-11-01','2009-12-01']:
        p=m.predict_proba(snap(te,date))[:,1]; yy=y.loc[te]; rows.append(dict(cutoff=date,n=len(yy),positives=int(yy.sum()),roc_auc=roc_auc_score(yy,p),average_precision=average_precision_score(yy,p)))
    pd.DataFrame(rows).to_csv(out/'chronological_sensitivity.csv',index=False)
    # Fixed reference distribution from training providers before July; claim complexity is a review cue, not evidence of fraud.
    ref=c[c.Provider.isin(tr)&(c.date<pd.Timestamp('2009-07-01'))]; q=ref.groupby('ip')[['dx_count','px_count']].quantile(.95)
    z=c[c.Provider.isin(te)&(c.date>=pd.Timestamp('2009-11-01'))].merge(d[['Provider','date']+fs],on=['Provider','date'],validate='many_to_one'); z['provider_score']=m.predict_proba(z[fs])[:,1]
    z['dx_high']=z.dx_count>z.ip.map(q.dx_count); z['px_high']=z.px_count>z.ip.map(q.px_count)
    z['review_index']=.8*z.provider_score+.1*z.dx_high+.1*z.px_high
    # Local model sensitivity: replace one feature with training median, not causal attribution.
    med=snap(tr,'2009-07-01').median(); effects=[]
    for f in fs:
        x=z[fs].copy(); x[f]=med[f]; effects.append(z.provider_score.to_numpy()-m.predict_proba(x)[:,1])
    a=np.array(effects).T; idx=np.argmax(np.abs(a),axis=1); z['largest_sensitivity_feature']=np.array(fs)[idx]; z['score_change_vs_median']=a[np.arange(len(z)),idx]
    z['review_action']='Verify coding and supporting documentation; no automatic denial'; z['status']='Pending human review'; z['daily_rank']=z.sort_values(['date','review_index','ClaimID'],ascending=[True,False,True]).groupby('date').cumcount()+1
    # rank assignment aligns by index; retain 10 per day
    queue=z[z.daily_rank<=10].sort_values(['date','daily_rank']); assert queue.groupby('date').size().max()<=10
    queue.to_csv(out/'claim_specific_review_queue.csv',index=False)
    same=z.groupby(['Provider','date']).review_index.nunique(); summary=pd.DataFrame([dict(claims=len(z),days=z.date.nunique(),queued=len(queue),provider_day_groups=len(same),groups_with_distinct_scores=int((same>1).sum()),dx_threshold_outpatient=q.loc[0,'dx_count'],dx_threshold_inpatient=q.loc[1,'dx_count'],px_threshold_outpatient=q.loc[0,'px_count'],px_threshold_inpatient=q.loc[1,'px_count'])]); summary.to_csv(out/'claim_review_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(summary.to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,default=Path('outputs')); a=p.parse_args(); run(a.data,a.out)
