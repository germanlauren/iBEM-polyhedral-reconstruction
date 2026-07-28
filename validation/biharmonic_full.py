#!/usr/bin/env python3
"""All biharmonic derivatives psi,i psi,ij psi,ijk psi,ijkl via generic chain rule.
Params (a,b,lm,lp) are LINEAR in x with constant gradients (-n,-Vnorm,-Vdir,-Vdir),
so x-derivatives of v_edge = contractions of param-partials with these gradients."""
import numpy as np, sympy as sp
import potentials_reference as R, harmonic_closed as H, harmonic_hessian as HH

_a,_b,_l=sp.symbols('a b l',real=True)
_g=_b*((_a**2+_b**2+_l**2)**sp.Rational(3,2)-(_a**2)**sp.Rational(3,2))/(3*(_b**2+_l**2))
# cache lambdified g partials d^(na+nb+nl) g / da^na db^nb dl^nl
_cache={}
def gpart(na,nb,nl):
    key=(na,nb,nl)
    if key not in _cache:
        e=sp.diff(_g,_a,na,_b,nb,_l,nl)
        _cache[key]=sp.lambdify((_a,_b,_l),e,modules=[{'DiracDelta':lambda *a:0.0},'numpy'])
    return _cache[key]

_x64,_w64=np.polynomial.legendre.leggauss(64)
def _quad(f,lm,lp):
    if abs(lp-lm)<1e-15: return 0.0
    mid=.5*(lp+lm); half=.5*(lp-lm)
    return half*np.sum(_w64*f(mid+half*_x64))

def v_partial(na,nb,nlm,nlp,a,b,lm,lp):
    if nlm>0 and nlp>0: return 0.0
    if nlp>0:  return  gpart(na,nb,nlp-1)(a,b,lp)
    if nlm>0:  return -gpart(na,nb,nlm-1)(a,b,lm)
    return _quad(lambda l: gpart(na,nb,0)(a,b,l), lm,lp)

# param index: 0=a,1=b,2=lm,3=lp
def counts(pl):
    c=[0,0,0,0]
    for p in pl: c[p]+=1
    return c
def vmix(pl,a,b,lm,lp):
    c=counts(pl); return v_partial(c[0],c[1],c[2],c[3],a,b,lm,lp)

def edge_tensors(a,b,lm,lp,n,Vn,Vd):
    G=[-n,-Vn,-Vd,-Vd]     # grad of each param wrt x
    v0=vmix([],a,b,lm,lp)
    T1=np.zeros(3); T2=np.zeros((3,3)); T3=np.zeros((3,3,3))
    for p in range(4):
        vp=vmix([p],a,b,lm,lp)
        T1+=vp*G[p]
        for q in range(4):
            vpq=vmix([p,q],a,b,lm,lp)
            T2+=vpq*np.multiply.outer(G[p],G[q])
            for r in range(4):
                vpqr=vmix([p,q,r],a,b,lm,lp)
                T3+=vpqr*np.multiply.outer(np.multiply.outer(G[p],G[q]),G[r])
    return v0,T1,T2,T3

def psi_derivs(x,V):
    grad=np.zeros(3); Hij=np.zeros((3,3)); T3=np.zeros((3,3,3)); T4=np.zeros((3,3,3,3))
    for (n,verts,fc) in H.faces_of_tetra(V):
        m=len(verts)
        for e in range(m):
            va=verts[e]; vb=verts[(e+1)%m]
            a,b,lm,lp,Vn,Vd=H.edge_frame(x,n,va,vb,fc)
            v0,t1,t2,t3=edge_tensors(a,b,lm,lp,n,Vn,Vd)
            grad += -n*v0
            Hij  += -np.multiply.outer(n,t1)
            T3   += -np.multiply.outer(n,t2)
            T4   += -np.multiply.outer(n,t3)
    return grad,Hij,T3,T4

if __name__=="__main__":
    V=[np.array(v,float) for v in [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]]
    x=np.array([0.7,0.6,0.9])
    grad,H2,T3,T4=psi_derivs(x,V)
    # references
    pts,ws=R.tet_quad_points(V,44); ps=lambda xx:R.psi_num(xx,pts,ws)
    gnum=R.grad_num(ps,x)
    phi_H=HH.hess_phi_closed(x,V)     # phi,ij (validated)
    print("psi,i   closed vs num:", np.round(grad,5), np.round(gnum,5), "max|d|=%.1e"%np.abs(grad-gnum).max())
    # symmetry checks
    print("psi,ij  symmetric:", np.abs(H2-H2.T).max())
    print("psi,ijk symmetric (ijk perm):", np.abs(T3-np.transpose(T3,(1,0,2))).max(), np.abs(T3-np.transpose(T3,(0,2,1))).max())
    print("psi,ijkl symmetric:", np.abs(T4-np.transpose(T4,(1,0,2,3))).max(), np.abs(T4-np.transpose(T4,(0,1,3,2))).max())
    # identity lap(psi)=2phi  -> trace(psi,ij)=2phi
    print("trace(psi,ij)=%.6f  2*phi=%.6f"%(np.trace(H2), 2*R.phi_num(x,pts,ws)))
    # identity lap(psi,ij)=2 phi,ij  -> contract last two of psi,ijkl
    lapT4=np.einsum('ijkk->ij',T4)
    print("max| lap(psi,ij) - 2 phi,ij | =", np.abs(lapT4-2*phi_H).max())
    # cross-check psi,ijk vs FD of psi,ij, psi,ijkl vs FD of psi,ijk
    def H2f(xx): return psi_derivs(xx,V)[1]
    T3fd=np.zeros((3,3,3)); h=1e-4
    for kk in range(3):
        e=np.zeros(3);e[kk]=h
        T3fd[:,:,kk]=(H2f(x+e)-H2f(x-e))/(2*h)
    print("psi,ijk closed vs FD(psi,ij): max|d|=%.2e"%np.abs(T3-T3fd).max())
