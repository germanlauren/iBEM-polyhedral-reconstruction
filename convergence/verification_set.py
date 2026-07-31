#!/usr/bin/env python3
"""Verification-set statistics for the reconstructed polyhedral potentials.
(1) harmonic phi (closed form) vs high-order tetra volume quadrature, at exterior
    field points classified by normalized boundary distance eta=d(x,dOmega)/Lc:
    far (eta>0.5), mid (0.1<eta<0.5), near (eta<0.05). Reference quadrature order
    raised until the reference changes < 1e-12; error e=|rec-ref|/max(|ref|,qscale).
(2) differential-identity / symmetry residuals (need NO external reference, valid
    everywhere): trace(psi,ij)-2phi ; lap(psi,ijkl)-2phi,ij ; psi,ijkl full symmetry.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "homogenization"))
import numpy as np, time
import numpy as np
rng=np.random.default_rng(7)
import potentials_reference as R
import harmonic_closed as HC
import harmonic_hessian as HH
import biharmonic_full as bf
bf._x64,bf._w64=np.polynomial.legendre.leggauss(96)   # retained edge order

def rand_tetra():
    while True:
        V=[np.zeros(3)]+[rng.uniform(-1,1,3) for _ in range(3)]
        M=np.array([V[1]-V[0],V[2]-V[0],V[3]-V[0]])
        vol=abs(np.linalg.det(M))/6
        # avoid slivers: require decent volume and edge lengths
        el=[np.linalg.norm(V[i]-V[j]) for i in range(4) for j in range(i+1,4)]
        if vol>0.03 and min(el)>0.3: return [np.asarray(v,float) for v in V]

def dist_to_boundary(x,V):
    # min distance to the 4 triangular faces (point-plane, clamped to triangle approx by plane dist)
    d=1e9
    for (n,verts,fc) in HC.faces_of_tetra(V):
        d=min(d,abs(np.dot(n,x-verts[0])))
    return d

def sample_exterior(V,Lc):
    c=sum(V)/4.0
    pts=[]
    for _ in range(40):
        x=c+rng.uniform(-2.2,2.2,3)
        # keep exterior: signed outside at least one face
        outside=any(np.dot(n,x-verts[0])>1e-6 for (n,verts,fc) in HC.faces_of_tetra(V))
        if outside:
            eta=dist_to_boundary(x,V)/Lc
            pts.append((x,eta))
    return pts

# ---- (1) phi accuracy vs reference, by category ----
cats={'far':[], 'mid':[], 'near':[]}
Ntet=45; Npts=0
phi_scale=[]
for _ in range(Ntet):
    V=rand_tetra()
    Lc=max(np.linalg.norm(V[i]-V[j]) for i in range(4) for j in range(i+1,4))
    # reference quadrature: raise order until stable <1e-12 (use n=30 then n=40)
    p30,w30=R.tet_quad_points(V,30); p40,w40=R.tet_quad_points(V,40)
    for (x,eta) in sample_exterior(V,Lc):
        phi_ref30=R.phi_num(x,p30,w30); phi_ref=R.phi_num(x,p40,w40)
        if abs(phi_ref-phi_ref30)/max(abs(phi_ref),1e-9)>1e-3:  # reference not converged here
            ref_ok=False
        else:
            ref_ok=True
        phi_c=HC.phi_closed(x,V)
        phi_scale.append(abs(phi_ref))
        rec=('near' if eta<0.05 else 'mid' if eta<0.5 else 'far')
        cats[rec].append((abs(phi_c-phi_ref), abs(phi_ref), ref_ok))
        Npts+=1

qscale=np.median(phi_scale)
print(f"(1) phi closed vs volume-quadrature reference | {Ntet} tetrahedra, {Npts} exterior field points")
print(f"    error e = |phi_c - phi_ref| / max(|phi_ref|, qscale),  qscale=median|phi_ref|={qscale:.4f}")
print(f"    {'category':>10} {'N':>5} {'median':>11} {'95th pct':>11} {'max':>11} {'ref-converged %':>16}")
for k in ['far','mid','near']:
    arr=cats[k]
    if not arr: continue
    e=np.array([a/max(b,qscale) for (a,b,ok) in arr])
    okf=100.0*np.mean([ok for (_,_,ok) in arr])
    print(f"    {k:>10} {len(arr):5d} {np.median(e):11.2e} {np.percentile(e,95):11.2e} {e.max():11.2e} {okf:15.0f}%")

# ---- (2) differential-identity / symmetry residuals (no external reference) ----
tr_res=[]; lap_res=[]; sym_res=[]
Ntet2=25
for _ in range(Ntet2):
    V=rand_tetra(); c=sum(V)/4.0
    for _ in range(4):
        x=c+rng.uniform(-1.8,1.8,3)
        if not any(np.dot(n,x-verts[0])>1e-6 for (n,verts,fc) in HC.faces_of_tetra(V)):
            continue  # keep exterior (phi harmonic, lap phi=0)
        phi=HC.phi_closed(x,V); phiH=HH.hess_phi_closed(x,V)
        _,Hij,_,T4=bf.psi_derivs(x,V)
        tr_res.append(abs(np.trace(Hij)-2*phi)/max(abs(2*phi),1e-9))
        lapT4=np.einsum('ijkk->ij',T4)
        lap_res.append(np.abs(lapT4-2*phiH).max()/max(np.abs(2*phiH).max(),1e-9))
        s=max(np.abs(T4-np.transpose(T4,p)).max() for p in [(1,0,2,3),(0,1,3,2),(2,1,0,3),(3,1,2,0)])
        sym_res.append(s/max(np.abs(T4).max(),1e-9))
tr_res=np.array(tr_res); lap_res=np.array(lap_res); sym_res=np.array(sym_res)
print(f"\n(2) differential-identity & symmetry residuals | {len(tr_res)} exterior points, {Ntet2} tetrahedra")
print(f"    trace(psi,ij)=2phi      : median {np.median(tr_res):.2e}  95th {np.percentile(tr_res,95):.2e}  max {tr_res.max():.2e}")
print(f"    lap(psi,ijkl)=2 phi,ij  : median {np.median(lap_res):.2e}  95th {np.percentile(lap_res,95):.2e}  max {lap_res.max():.2e}")
print(f"    psi,ijkl full symmetry  : median {np.median(sym_res):.2e}  95th {np.percentile(sym_res,95):.2e}  max {sym_res.max():.2e}")
