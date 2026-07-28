#!/usr/bin/env python3
"""Closed-form UNIFORM-polyhedron harmonic potential phi and derivatives phi,i phi,ij
as edge/face contributions, validated against potentials_reference.py (numerical)."""
import numpy as np
import potentials_reference as R

def faces_of_tetra(V):
    V=[np.asarray(v,float) for v in V]
    c=sum(V)/4.0
    tris=[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    faces=[]
    for t in tris:
        a,b,cc=V[t[0]],V[t[1]],V[t[2]]
        n=np.cross(b-a,cc-a); n=n/np.linalg.norm(n)
        fc=(a+b+cc)/3.0
        if np.dot(n,fc-c)<0: n=-n   # outward
        faces.append((n,[V[t[0]],V[t[1]],V[t[2]]],fc))
    return faces

def edge_frame(x, n, va, vb, face_centroid):
    Vdir=vb-va; Vdir=Vdir/np.linalg.norm(Vdir)
    Vnorm=np.cross(Vdir,n); Vnorm=Vnorm/np.linalg.norm(Vnorm)
    # outward within plane: point away from face interior
    if np.dot(Vnorm, face_centroid-va)>0: Vnorm=-Vnorm
    a=np.dot(n,(va-x))
    b=np.dot(va-x,Vnorm)
    lm=np.dot(va-x,Vdir); lp=np.dot(vb-x,Vdir)
    return a,b,lm,lp,Vnorm,Vdir

def w_edge(a,b,lm,lp):
    rp=np.sqrt(a*a+b*b+lp*lp); rm=np.sqrt(a*a+b*b+lm*lm)
    d2=a*a+b*b
    # log (line) term
    log_term=0.0
    if abs(b)>1e-14:
        log_term=b*(np.log(rp+lp)-np.log(rm+lm))
    # solid-angle term
    sa=0.0
    if abs(a)>1e-14 and abs(b)>1e-14:
        sa=-abs(a)*(np.arctan(b*lp/(d2+abs(a)*rp))-np.arctan(b*lm/(d2+abs(a)*rm)))
    return log_term+sa

def W_face(x,n,verts,fc):
    W=0.0
    m=len(verts)
    for e in range(m):
        va=verts[e]; vb=verts[(e+1)%m]
        a,b,lm,lp,Vn,Vd=edge_frame(x,n,va,vb,fc)
        W+=w_edge(a,b,lm,lp)
    return W

def phi_closed(x,V):
    return 0.5*sum(np.dot(n,verts[0]-x)*W_face(x,n,verts,fc) for (n,verts,fc) in faces_of_tetra(V))
def gradphi_closed(x,V):
    g=np.zeros(3)
    for (n,verts,fc) in faces_of_tetra(V):
        g+= -n*W_face(x,n,verts,fc)
    return g

if __name__=="__main__":
    V=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    pts,ws=R.tet_quad_points(V,40)
    ph=lambda xx: R.phi_num(xx,pts,ws)
    for x in [np.array([0.7,0.6,0.9]),np.array([1.5,0.2,0.3]),np.array([-0.4,0.5,0.5])]:
        phi_n=ph(x); phi_c=phi_closed(x,V)
        g_n=R.grad_num(ph,x); g_c=gradphi_closed(x,V)
        print(f"x={x}")
        print(f"   phi:   num={phi_n:.6f} closed={phi_c:.6f}  diff={abs(phi_n-phi_c):.2e}")
        print(f"   grad:  num={np.round(g_n,5)} closed={np.round(g_c,5)}  max|d|={np.abs(g_n-g_c).max():.2e}")
