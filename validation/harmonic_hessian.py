#!/usr/bin/env python3
"""phi,ij closed form via chain rule on w_edge(a,b,lm,lp); analytic w-derivatives
from sympy; validated against finite differences of the (validated) closed gradient
and against a tight 2D-quadrature check of W_face."""
import numpy as np, sympy as sp
import potentials_reference as R
import harmonic_closed as H

# ---- analytic partials of w_edge via sympy, lambdified ----
a_,b_,lm_,lp_=sp.symbols('a b lm lp',real=True)
rp=sp.sqrt(a_**2+b_**2+lp_**2); rm=sp.sqrt(a_**2+b_**2+lm_**2); d2=a_**2+b_**2
w=b_*(sp.log(rp+lp_)-sp.log(rm+lm_)) - sp.Abs(a_)*(sp.atan(b_*lp_/(d2+sp.Abs(a_)*rp))-sp.atan(b_*lm_/(d2+sp.Abs(a_)*rm)))
w_a=sp.diff(w,a_); w_b=sp.diff(w,b_); w_lm=sp.diff(w,lm_); w_lp=sp.diff(w,lp_)
fa=sp.lambdify((a_,b_,lm_,lp_),w_a,'numpy'); fb=sp.lambdify((a_,b_,lm_,lp_),w_b,'numpy')
flm=sp.lambdify((a_,b_,lm_,lp_),w_lm,'numpy'); flp=sp.lambdify((a_,b_,lm_,lp_),w_lp,'numpy')

def hess_phi_closed(x,V):
    Hm=np.zeros((3,3))
    for (n,verts,fc) in H.faces_of_tetra(V):
        m=len(verts)
        for e in range(m):
            va=verts[e]; vb=verts[(e+1)%m]
            a,b,lm,lp,Vn,Vd=H.edge_frame(x,n,va,vb,fc)
            wa=fa(a,b,lm,lp); wb=fb(a,b,lm,lp); wlm=flm(a,b,lm,lp); wlp=flp(a,b,lm,lp)
            # (W_face),j = sum_edges [ -n_j wa - Vn_j wb - Vd_j (wlm+wlp) ]
            dWj = -n*wa - Vn*wb - Vd*(wlm+wlp)
            # phi,ij = -sum_faces n_i (W),j
            Hm += -np.outer(n,dWj)
    return Hm

# tight check of W_face vs 2D triangle quadrature
def W_face_num(x,verts,ntri=200):
    v0,v1,v2=verts
    A=0.5*np.linalg.norm(np.cross(v1-v0,v2-v0))
    s=0.0; cnt=0
    for i in range(ntri):
        for j in range(ntri-i):
            u=(i+1/3)/ntri; v=(j+1/3)/ntri
            p=v0+u*(v1-v0)+v*(v2-v0); s+=1.0/np.linalg.norm(p-x); cnt+=1
    return s*A/cnt

if __name__=="__main__":
    V=[np.array(v,float) for v in [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]]
    x=np.array([0.7,0.6,0.9])
    # 1) W_face closed vs 2D quadrature
    print("W_face closed vs 2D-quadrature:")
    for (n,verts,fc) in H.faces_of_tetra(V):
        Wc=H.W_face(x,n,verts,fc); Wn=W_face_num(x,verts)
        print(f"   closed={Wc:.6f} quad={Wn:.6f} diff={abs(Wc-Wn):.2e}")
    # 2) phi,ij closed vs FD of the validated closed gradient
    def gc(xx): return H.gradphi_closed(xx,V)
    Hfd=np.zeros((3,3)); h=1e-5
    for j in range(3):
        e=np.zeros(3); e[j]=h
        Hfd[:,j]=(gc(x+e)-gc(x-e))/(2*h)
    Hc=hess_phi_closed(x,V)
    print("\nphi,ij closed (analytic) vs FD-of-closed-gradient:")
    print("closed=\n",np.round(Hc,6)); print("FD=\n",np.round(Hfd,6))
    print("max abs diff=",np.abs(Hc-Hfd).max())
    print("symmetry max|H-H^T|=",np.abs(Hc-Hc.T).max(),"  trace(should~0 outside)=",np.trace(Hc))
