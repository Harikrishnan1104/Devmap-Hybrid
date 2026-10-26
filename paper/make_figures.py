import sys; sys.path.insert(0,'.')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os; O=os.path.join(os.path.dirname(os.path.abspath(__file__)),'figures')+'/'
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman'],'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#2a6fdb'; GREY='#9a9a9a'; DARK='#333333'
W=3.45
# Fig 1 architecture
fig,ax=plt.subplots(figsize=(W,2.7)); ax.axis('off'); ax.set_xlim(-0.2,10.45); ax.set_ylim(0,7.4)
def box(x,y,w,h,t,fc='#f2f2f2',ec=DARK,fs=5.6,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.15',fc=fc,ec=ec,lw=0.7))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal')
def arr(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=0.7,color=DARK))
box(0.0,5.9,3.2,1.2,'OpenCL kernel\nsource code')
box(3.5,5.9,3.0,1.2,'Static code counts\n(comp, mem, ...)')
box(6.8,5.9,3.2,1.2,'Job parameters\n(transfer, wgsize)')
box(0.0,3.9,3.2,1.3,'Frozen GraphCodeBERT\n(mean-pooled, 768-d)',fc='#e3ecfb',ec=BLUE)
box(0.0,2.2,3.2,1.1,'Scaler + PCA to 16-d\n(fit on training only)',fc='#e3ecfb',ec=BLUE)
box(5.45,3.9,4.75,1.3,'5 scale-free ratios\n(data per op, coalesced share,\nlocal share, comp/mem, ops/item)')
box(2.2,0.15,5.6,1.35,'Random Forest (300 trees)\n→ CPU or GPU, with confidence',fc='#dde8f8',ec=BLUE,bold=True)
arr(1.6,5.9,1.6,5.2); arr(1.6,3.9,1.6,3.3); arr(1.6,2.2,3.4,1.5)
arr(5.2,5.9,6.6,5.2); arr(8.4,5.9,8.2,5.2); arr(7.5,3.9,6.6,1.5); arr(5.0,5.9,5.0,1.5)
ax.text(8.9,2.5,'concatenate\n16 + 6 + 5 = 27\nfeatures',ha='center',fontsize=5.4,style='italic')
fig.savefig(O+'fig1_arch.png',dpi=300,bbox_inches='tight'); plt.close()
# Fig 2 leakage drop
models=['CB','GCB','RF','Hybrid']
old_a=[92.9,93.7,92.1,87.1]; old_n=[88.7,87.1,88.8,87.2]; ua_a=[62.5,72.5,76.3,75.8]; ua_n=[64.3,64.5,69.7,72.1]
fig,axs=plt.subplots(1,2,figsize=(W,2.0),sharey=True)
x=np.arange(4)
for a,(old,ua,t) in zip(axs,[(old_a,ua_a,'AMD'),(old_n,ua_n,'NVIDIA')]):
    a.bar(x-0.19,old,0.36,color=GREY,label='Random 5-fold (paper)')
    a.bar(x+0.19,ua,0.36,color=BLUE,label='Unseen applications')
    for i in range(4):
        a.text(x[i]+0.19,ua[i]+0.8,f'{ua[i]:.0f}',ha='center',fontsize=5.6)
        a.text(x[i]-0.19,old[i]+0.8,f'{old[i]:.0f}',ha='center',fontsize=5.6,color='#555')
    a.set_xticks(x); a.set_xticklabels(models,fontsize=6); a.set_title(t,fontsize=7.5); a.set_ylim(50,100)
    a.axhline(58.8 if t=='AMD' else 56.9,ls='--',lw=0.6,color='#c0392b')
axs[0].set_ylabel('Accuracy (%)')
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,fontsize=5.6,frameon=False,loc='lower center',ncol=2,bbox_to_anchor=(0.5,-0.08))
fig.text(0.5,-0.13,'CB = CodeBERTa, GCB = GraphCodeBERT (both fine-tuned), RF = Random Forest; dashed = majority guess',ha='center',fontsize=5.2,color='#555')
fig.savefig(O+'fig2_leakage.png',dpi=300,bbox_inches='tight'); plt.close()
# Fig 3 GPU share by data size
bands=['0–6 MB','6–58 MB','58–378 MB','0.4–3.7 GB']
fig,ax=plt.subplots(figsize=(W,1.7))
ax.plot(range(4),[13,33,46,72],'-o',color=BLUE,ms=3.5,lw=1.4,label='AMD')
ax.plot(range(4),[43,69,55,60],'-o',color=GREY,ms=3.5,lw=1.4,label='NVIDIA')
for i,(a,b) in enumerate(zip([13,33,46,72],[43,69,55,60])):
    ax.text(i,a-7 if a<b else a+3,f'{a}%',ha='center',fontsize=5.8,color=BLUE)
    ax.text(i,b+3 if b>a else b-7,f'{b}%',ha='center',fontsize=5.8,color='#555')
ax.axhline(50,ls=':',lw=0.6,color='#777'); ax.set_xticks(range(4)); ax.set_xticklabels(bands,fontsize=6)
ax.set_ylabel('GPU faster (%)'); ax.set_ylim(0,85); ax.legend(frameon=False,fontsize=6,loc='lower right'); ax.set_xlabel('Bytes copied between CPU and GPU (quartiles, 170 runs each)',fontsize=6.2)
fig.savefig(O+'fig3_datasize.png',dpi=300,bbox_inches='tight'); plt.close()
# Fig 4 permutation importance
feat=['Data size','Work-group size','Coalesced share','Comparisons','Work per item','Coalesced acc.','Data per op','Local share','Memory acc.','Arith./mem. acc.','Local-mem acc.','Arith. ops']
amd=[13.6,3.2,2.5,0.4,0.2,2.3,1.2,0.2,0.3,0.6,0.3,0.4]; nv=[-0.4,3.4,0.7,1.7,1.5,-1.0,0.0,-0.2,-0.5,-0.9,-0.7,-1.8]
fig,axs=plt.subplots(1,2,figsize=(W,2.3),sharey=True)
y=np.arange(len(feat))[::-1]
for a,v,t in zip(axs,[amd,nv],['AMD','NVIDIA']):
    c=[BLUE if (val==max(v)) else (GREY if val>=0 else '#d0d0d0') for val in v]
    a.barh(y,v,color=c,height=0.65); a.axvline(0,color=DARK,lw=0.6); a.set_title(t,fontsize=7.5); a.set_xlim(-2.5,15)
    for yi,val in zip(y,v): a.text(val+0.3 if val>=0 else 0.3,yi,f'{val:.1f}',va='center',fontsize=5.2)
axs[0].set_yticks(y); axs[0].set_yticklabels(feat,fontsize=5.8)
fig.text(0.55,-0.01,'Accuracy points lost when the feature is shuffled (unseen applications)',ha='center',fontsize=6)
fig.savefig(O+'fig4_importance.png',dpi=300,bbox_inches='tight'); plt.close()
# Fig 5 NPB vs rest
groups=['NPB\n(527 rows)','Other suites\n(153 rows)']
fig,axs=plt.subplots(1,2,figsize=(W,1.85),sharey=True)
data={'AMD':{'GCB fine-tuned':[73.2,70.2],'Hybrid':[78.2,61.4],'Random Forest':[79.3,61.4]},'NVIDIA':{'GCB fine-tuned':[62.7,70.4],'Hybrid':[76.3,64.7],'Random Forest':[71.7,64.1]}}
cols={'GCB fine-tuned':BLUE,'Hybrid':'#1b3f7a','Random Forest':GREY}
for a,(t,d) in zip(axs,data.items()):
    for k,(name,v) in enumerate(d.items()):
        a.bar(np.arange(2)+(k-1)*0.26,v,0.25,color=cols[name],label=name)
        for i in range(2): a.text(i+(k-1)*0.26,v[i]+0.8,f'{v[i]:.0f}',ha='center',fontsize=5.2)
    a.set_xticks(range(2)); a.set_xticklabels(groups,fontsize=5.6); a.set_title(t,fontsize=7.5); a.set_ylim(50,85)
axs[0].set_ylabel('Accuracy (%)'); axs[1].legend(frameon=False,fontsize=5.2,loc='upper right')
fig.savefig(O+'fig5_npb.png',dpi=300,bbox_inches='tight'); plt.close()
print('ok')
