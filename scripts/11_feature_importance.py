"""Faculty points 1, 2, 5: feature importance four ways (impurity, permutation, SHAP, mutual information) on the unseen_app and old exams. Writes results/feature_importance.csv."""
import pandas as pd, numpy as np, devmap, features as F, shap, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
cols=['comp','rational','mem','localmem','coalesced','transfer','wgsize']
res=[]
for m in ['amd','nvidia']:
    df=devmap.load(m).set_index('row_id',drop=False)
    X=pd.concat([df[cols].astype(float),F.ratios(df)],axis=1); y=df.label
    mi=pd.Series(mutual_info_classif(X,y,random_state=1,discrete_features=False),index=X.columns)
    for exam in ['unseen_app','old']:
        rounds=devmap.build_exam(df,exam,seed=1)
        imp=[];perm=[];sh=[];acc=[]
        for r in rounds:
            tr,ex=r['study'],r['exam']
            rf=RandomForestClassifier(300,random_state=1,n_jobs=-1).fit(X.loc[tr],y.loc[tr])
            acc.append((rf.predict(X.loc[ex])==y.loc[ex]).mean()*len(ex))
            imp.append(pd.Series(rf.feature_importances_,index=X.columns)*len(ex))
            p=permutation_importance(rf,X.loc[ex],y.loc[ex],n_repeats=10,random_state=1,scoring='accuracy',n_jobs=-1)
            perm.append(pd.Series(p.importances_mean,index=X.columns)*len(ex))
            if exam=='unseen_app':
                sv=shap.TreeExplainer(rf).shap_values(X.loc[ex])
                sv=sv[1] if isinstance(sv,list) else sv[...,1]
                sh.append(pd.Series(np.abs(sv).mean(0),index=X.columns)*len(ex))
        n=len(df)
        out=pd.DataFrame({'impurity':sum(imp)/n,'permutation (acc drop)':sum(perm)/n*100})
        if sh: out['SHAP']=sum(sh)/n
        out['MI']=mi
        out['machine']=m;out['exam']=exam;out['rf acc']=round(sum(acc)/n*100,1)
        res.append(out)
R=pd.concat(res); R.index.name='feature'
R.to_csv('../results/feature_importance.csv')
pd.set_option('display.width',200)
for (m,e),g in R.groupby(['machine','exam']):
    print('\n',m,e,'RF acc',g['rf acc'].iloc[0]); print(g.drop(columns=['machine','exam','rf acc']).sort_values('permutation (acc drop)',ascending=False).round(3).to_string())
