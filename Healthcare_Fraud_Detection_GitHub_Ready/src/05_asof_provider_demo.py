"""Service-date proxy experiment; not verified submission-time or claim fraud validation."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

def main(data, out):
    out.mkdir(parents=True,exist_ok=True)
    cols=['Provider','ClaimID','ClaimStartDt','BeneID']
    claims=pd.concat([pd.read_csv(next(data.glob('Train_'+kind+'*.csv')),usecols=cols).assign(inpatient=int(kind=='Inpatientdata')) for kind in ['Inpatientdata','Outpatientdata']],ignore_index=True)
    claims['date']=pd.to_datetime(claims.ClaimStartDt,errors='raise')
    assert claims.ClaimID.is_unique
    labels=pd.read_csv(next(data.glob('Train-*.csv'))).set_index('Provider').PotentialFraud.map({'Yes':1,'No':0})
    assert labels.notna().all()
    daily=claims.groupby(['Provider','date']).agg(n=('ClaimID','size'),ip=('inpatient','sum')).reset_index().sort_values(['Provider','date'])
    for c in ['n','ip']:
        daily['prior_'+c]=daily.groupby('Provider')[c].cumsum()-daily[c]
    daily['prior_days']=daily.groupby('Provider').cumcount()
    daily['prior_ip_share']=daily.prior_ip/daily.prior_n.replace(0,np.nan)
    features=['prior_n','prior_ip','prior_days','prior_ip_share']
    # Independent direct-count checks for sampled provider dates.
    for r in daily.sample(100,random_state=42).itertuples():
        past=claims[(claims.Provider==r.Provider)&(claims.date<r.date)]
        assert len(past)==r.prior_n and past.inpatient.sum()==r.prior_ip
    # Provider-disjoint, date-ordered snapshots; labels are static retrospective provider labels.
    train_ids,hold_ids=train_test_split(labels.index,test_size=.4,random_state=42,stratify=labels)
    val_ids,test_ids=train_test_split(hold_ids,test_size=.5,random_state=42,stratify=labels.loc[hold_ids])
    def snapshot(ids, cutoff):
        d=daily[daily.Provider.isin(ids)&(daily.date<pd.Timestamp(cutoff))].groupby('Provider').tail(1).set_index('Provider')
        # Include all observed claims strictly before snapshot cutoff, including last observed day.
        d['prior_n']+=d.n; d['prior_ip']+=d.ip; d['prior_days']+=1
        d['prior_ip_share']=d.prior_ip/d.prior_n
        x=d[features].reindex(ids); x[['prior_n','prior_ip','prior_days']]=x[['prior_n','prior_ip','prior_days']].fillna(0)
        return x,labels.loc[ids]
    X,y=snapshot(train_ids,'2009-07-01'); V,v=snapshot(val_ids,'2009-09-01'); T,t=snapshot(test_ids,'2009-11-01')
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,random_state=42).fit(X,y)
    pv=model.predict_proba(V)[:,1]; pt=model.predict_proba(T)[:,1]
    rows=[]
    for name,yy,p in [('validation',v,pv),('test',t,pt)]:
        rows.append(dict(split=name,providers=len(yy),positive=int(yy.sum()),roc_auc=roc_auc_score(yy,p),average_precision=average_precision_score(yy,p)))
    pd.DataFrame(rows).to_csv(out/'asof_snapshot_metrics.csv',index=False)
    rank=pd.DataFrame({'label':t.values,'score':pt},index=t.index).sort_values('score',ascending=False,kind='stable')
    pd.DataFrame([dict(k=k,captured=int(rank.head(k).label.sum()),precision=rank.head(k).label.mean(),recall=rank.head(k).label.sum()/t.sum()) for k in [25,50,100,200]]).to_csv(out/'asof_review_capacity.csv',index=False)
    # Claim-level illustration: inherited provider risk, not claim-specific fraud probability.
    demo=claims[claims.Provider.isin(test_ids)&(claims.date>=pd.Timestamp('2009-11-01'))].merge(daily[['Provider','date']+features],on=['Provider','date'],validate='many_to_one')
    demo['provider_risk_score']=model.predict_proba(demo[features])[:,1]
    demo.sort_values(['date','provider_risk_score','ClaimID'],ascending=[True,False,True]).groupby('date').head(10).drop(columns=['BeneID','ClaimStartDt']).to_csv(out/'asof_claim_queue_demo.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print('claims',len(claims),'sample checks passed 100; demonstration claims',len(demo))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,default=Path('outputs')); a=p.parse_args(); main(a.data,a.out)
