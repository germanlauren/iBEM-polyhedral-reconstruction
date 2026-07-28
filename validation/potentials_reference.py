#!/usr/bin/env python3
"""Numerical GROUND-TRUTH for the uniform-polyhedron potentials used by iBEM's
Eshelby machinery, to validate the closed-form Uniform_Polyhedral reconstruction.

Harmonic potential   phi(x)  = \int_V 1/|x-y| dV
Biharmonic potential psi(x)  = \int_V |x-y|   dV
Derivatives phi,i phi,ij and psi,i..psi,ijkl are what the C++ methods must return
(summed over faces/edges). Here we compute them by direct tetra quadrature so any
closed form can be checked against these numbers.
"""
import numpy as np
from itertools import product

# ---- high-order tetra quadrature (subdivide + centroid rule for robustness) ----
def tet_quad_points(verts, n=24):
    # barycentric grid over the tetra; returns points and weights (sum=vol)
    v0,v1,v2,v3=[np.asarray(v,float) for v in verts]
    vol=abs(np.dot(np.cross(v1-v0,v2-v0),v3-v0))/6.0
    pts=[]; ws=[]
    # uniform barycentric sampling
    idx=[(i,j,k) for i in range(n) for j in range(n-i) for k in range(n-i-j)]
    for (i,j,k) in idx:
        l=n-1-i-j-k
        # cell centroid in barycentric (i+.. )/n ; simple midpoint sampling
        b=np.array([i+0.25,j+0.25,k+0.25,l+0.25]); b=b/b.sum()
        p=b[0]*v0+b[1]*v1+b[2]*v2+b[3]*v3
        pts.append(p); ws.append(1.0)
    pts=np.array(pts); ws=np.array(ws); ws*=vol/ws.sum()
    return pts,ws

def phi_num(x, pts, ws):
    r=np.linalg.norm(pts-x,axis=1); return np.sum(ws/r)
def psi_num(x, pts, ws):
    r=np.linalg.norm(pts-x,axis=1); return np.sum(ws*r)

def grad_num(f, x, h=1e-4):
    g=np.zeros(3)
    for i in range(3):
        e=np.zeros(3); e[i]=h
        g[i]=(f(x+e)-f(x-e))/(2*h)
    return g
def hess_num(f, x, h=2e-3):
    H=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            ei=np.zeros(3); ei[i]=h; ej=np.zeros(3); ej[j]=h
            H[i,j]=(f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej))/(4*h*h)
    return H

if __name__=="__main__":
    # test tetra + an EXTERIOR field point (avoid 1/r singularity)
    verts=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    x=np.array([0.7,0.6,0.9])
    for n in (16,24,32):
        pts,ws=tet_quad_points(verts,n)
        ph=lambda xx: phi_num(xx,pts,ws); ps=lambda xx: psi_num(xx,pts,ws)
        phi=ph(x); gphi=grad_num(ph,x); Hphi=hess_num(ph,x)
        lap=np.trace(Hphi)   # should ~0 outside (harmonic); -4pi inside
        print(f"n={n:2d} vol-pts={len(pts):5d} phi={phi:.6f} |grad phi|={np.linalg.norm(gphi):.6f} lap(phi)={lap:.3e}")
    # biharmonic check: lap(psi)=2*phi (since lap|r|=2/r)
    psi=ps(x); Hpsi=hess_num(ps,x); lap_psi=np.trace(Hpsi)
    print(f"psi={psi:.6f}  lap(psi)={lap_psi:.6f}  2*phi={2*phi:.6f}  (should match)")
