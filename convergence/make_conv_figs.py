#!/usr/bin/env python3
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FG=os.path.join(os.path.dirname(os.path.abspath(__file__)),"figs")+os.sep
import os; os.makedirs(FG,exist_ok=True)

# ---- real data ----
Ng=np.array([12,24,48,96]); dHij=np.array([3.835e-4,3.427e-5,8.496e-9,8.017e-11])
dT4=np.array([3.698e-2,4.024e-3,4.332e-5,3.506e-7])
elems=np.array([24,54,96]); nbias=np.array([14.15,7.12,4.21]); c44b=np.array([2.82,1.10,0.48])
cats=['far\n(η>0.5)','mid\n(0.1–0.5)','near\n(η<0.05)']
med=np.array([1.49e-4,2.04e-4,2.64e-4]); p95=np.array([6.22e-4,1.48e-3,2.75e-3]); mx=np.array([1.31e-3,1.01e-2,1.29e-2])
idn=['trace(ψ,ij)\n−2φ','∇²(ψ,ijkl)\n−2φ,ij','ψ,ijkl\nsymmetry']
imed=np.array([5.42e-15,1.17e-13,1.07e-13]); i95=np.array([4.00e-14,2.99e-11,3.90e-11]); imx=np.array([1.02e-13,6.41e-10,7.65e-10])

# ===== Figure 4a: convergence =====
fig,ax=plt.subplots(1,2,figsize=(9.2,3.7))
ax[0].loglog(Ng,dHij,'o-',color='#1f4e79',label='ψ,ij')
ax[0].loglog(Ng,dT4,'s-',color='#c0392b',label='ψ,ijkl')
ax[0].axvline(96,ls=':',color='gray'); ax[0].text(99,3e-2,'retained\nN$_g$=96',fontsize=8,color='gray')
ax[0].set_xlabel('edge Gauss–Legendre points  N$_g$'); ax[0].set_ylabel('max relative change vs N$_g$=192')
ax[0].set_title('(a$_1$) Edge-quadrature convergence',fontsize=10); ax[0].legend(fontsize=9); ax[0].grid(True,which='both',alpha=.3)
ax[1].loglog(elems,nbias,'o-',color='#1f4e79',label='normal C$_{ii}$ bias')
ax[1].loglog(elems,c44b,'s-',color='#27ae60',label='shear C$_{44}$ bias')
for e,b in zip(elems,nbias): ax[1].annotate(f'{b:.1f}%',(e,b),textcoords='offset points',xytext=(4,4),fontsize=8)
ax[1].set_xlabel('boundary elements (unit-contrast cube)'); ax[1].set_ylabel('departure from exact tensor (%)')
ax[1].set_title('(a$_2$) BEM discretization convergence',fontsize=10); ax[1].legend(fontsize=9); ax[1].grid(True,which='both',alpha=.3)
plt.tight_layout(); plt.savefig(FG+"fig_conv_quad.png",dpi=200); plt.close()

# ===== Figure 4b: error distributions =====
fig,ax=plt.subplots(1,2,figsize=(9.2,3.7))
x=np.arange(3); w=0.25
ax[0].bar(x-w,med,w,label='median',color='#5b9bd5'); ax[0].bar(x,p95,w,label='95th pct',color='#2e75b6'); ax[0].bar(x+w,mx,w,label='max',color='#1f4e79')
ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels(cats,fontsize=9)
ax[0].set_ylabel('relative error  |φ$_c$−φ$_{ref}$|/max(|φ$_{ref}$|,q$_{scale}$)')
ax[0].set_title('(b$_1$) φ vs reference quadrature\n45 tetrahedra, 1799 exterior points',fontsize=9.5); ax[0].legend(fontsize=8); ax[0].grid(True,axis='y',alpha=.3)
x2=np.arange(3)
ax[1].bar(x2-w,imed,w,label='median',color='#f0b27a'); ax[1].bar(x2,i95,w,label='95th pct',color='#e67e22'); ax[1].bar(x2+w,imx,w,label='max',color='#ba4a00')
ax[1].set_yscale('log'); ax[1].set_xticks(x2); ax[1].set_xticklabels(idn,fontsize=8.5)
ax[1].axhline(2.2e-16,ls=':',color='gray'); ax[1].text(2.1,3e-16,'machine ε',fontsize=8,color='gray')
ax[1].set_ylabel('normalized residual'); ax[1].set_ylim(1e-16,1e-8)
ax[1].set_title('(b$_2$) Differential-identity & symmetry residuals\n25 tetrahedra, 99 points',fontsize=9.5); ax[1].legend(fontsize=8); ax[1].grid(True,axis='y',alpha=.3)
plt.tight_layout(); plt.savefig(FG+"fig_conv_dist.png",dpi=200); plt.close()
print("wrote fig_conv_quad.png and fig_conv_dist.png")
