#!/usr/bin/env python3
"""Edge-quadrature convergence of the biharmonic polyhedral derivatives.
The HARMONIC potential phi and phi,ij are closed-form (log + atan2), independent
of the edge-quadrature order Ng. Only the BIHARMONIC edge kernel v (Eq. 18) is
evaluated by Gauss-Legendre; hence only psi,ij and psi,ijkl converge with Ng.
Reference = Ng=192. Reports max relative change of psi,ij and psi,ijkl vs ref,
plus wall time, over a fixed set of field points and tetrahedra."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "homogenization"))
import numpy as np, time
import biharmonic_full as bf
import harmonic_closed as H

def set_Ng(Ng):
    bf._x64, bf._w64 = np.polynomial.legendre.leggauss(Ng)

# fixed evaluation set: 3 tetrahedra x several exterior field points
tets = [
    [(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
    [(0,0,0),(1.3,0.2,0.1),(0.1,1.1,0.2),(0.2,0.3,1.4)],
    [(0,0,0),(2,0,0),(0.5,1.5,0),(0.4,0.4,1.0)],
]
tets = [[np.array(v,float) for v in T] for T in tets]
fpts = [np.array(p,float) for p in
        [(0.7,0.6,0.9),(1.5,0.2,0.3),(-0.4,0.5,0.5),(0.9,0.9,0.9),(1.1,1.1,0.2)]]

def eval_all(Ng):
    set_Ng(Ng)
    H2s=[]; T4s=[]
    for T in tets:
        for x in fpts:
            _,Hij,_,T4 = bf.psi_derivs(x,T)
            H2s.append(Hij.ravel()); T4s.append(T4.ravel())
    return np.concatenate(H2s), np.concatenate(T4s)

orders=[12,24,48,96,192]
res={}
for Ng in orders:
    t0=time.time()
    h,t=eval_all(Ng)
    res[Ng]=(h,t,time.time()-t0)

hR,tR,_=res[192]
def relmax(a,ref):
    scale=np.maximum(np.abs(ref).max(),1e-12)
    return np.abs(a-ref).max()/scale

print("Edge-quadrature convergence (reference Ng=192)")
print(f"{'Ng':>5} {'d(psi,ij)':>14} {'d(psi,ijkl)':>14} {'time[s]':>9}")
for Ng in orders:
    h,t,dt=res[Ng]
    print(f"{Ng:5d} {relmax(h,hR):14.3e} {relmax(t,tR):14.3e} {dt:9.3f}")
