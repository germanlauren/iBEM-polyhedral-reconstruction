#!/usr/bin/env python3
"""Resumable dilute Eshelby/Mori-Tanaka benchmark in the EIM-valid regime.
Stiff spherical inclusion (contrast 10) in an isotropic matrix, phi in {0.05,0.10},
fine n=4 boundary mesh, exact symmetrized boundary-traction average (Eq. 19).
Checkpoints to esh.npz; call until DONE, then compares to the Mori-Tanaka sphere."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "homogenization"))
import numpy as np, os, time, shutil, tempfile
import ibem_homog as H

E0,nu0=1000.0,0.30; E1,nu1=10000.0,0.30; L=1.0; n=4
PHIS=[0.05,0.10]
def r_of(phi): return (phi*3/(4*np.pi))**(1/3.)
ckpt="esh.npz"; cases=H.macro_cases()
if os.path.exists(ckpt):
    d=np.load(ckpt,allow_pickle=True); C=d["C"]; done=set(map(tuple,d["done"]))
else:
    C=np.zeros((len(PHIS),6,6)); done=set()
tmp=tempfile.mkdtemp(); b=os.path.join(tmp,"iBEM_run"); shutil.copy(H.BIN,b); os.chmod(b,0o755)
nodes,quads=H.cube_mesh(L,n); pts=H.interior_grid(L,6)
t0=time.time(); BUDGET=30.0
try:
    for pi,phi in enumerate(PHIS):
        sph=[(0.5,0.5,0.5,r_of(phi))]
        for col in range(6):
            if (pi,col) in done: continue
            if time.time()-t0>BUDGET: raise TimeoutError
            sig=H.run_case_boundary(b,tmp,E0,nu0,E1,nu1,nodes,quads,sph,pts,cases[col],L)
            C[pi,:,col]=sig; done.add((pi,col))
            np.savez(ckpt,C=C,done=np.array(sorted(done)))
            print(f"phi={phi:.2f} case {col} done")
except TimeoutError:
    print("BUDGET hit -> call again")
finally:
    shutil.rmtree(tmp,ignore_errors=True)

def mt_sphere(phi,Km,mum,Ki,mui):
    K=Km+phi*(Ki-Km)*(3*Km+4*mum)/(3*Km+4*mum+3*(1-phi)*(Ki-Km))
    fm=mum*(9*Km+8*mum)/(6*(Km+2*mum))
    mu=mum+phi*(mui-mum)/(1+(1-phi)*(mui-mum)/(mum+fm))
    E=9*K*mu/(3*K+mu); return K,mu,E

if len(done)==len(PHIS)*6:
    Km=E0/(3*(1-2*nu0)); mum=E0/(2*(1+nu0))
    Ki=E1/(3*(1-2*nu1)); mui=E1/(2*(1+nu1))
    print("\n=== DONE: dilute Eshelby/MT benchmark (stiff sphere, contrast 10, n=4) ===")
    print(f"{'phi':>6} {'iBEM K':>9} {'MT K':>9} {'iBEM mu':>9} {'MT mu':>9} {'iBEM E':>9} {'MT E':>9} {'E rel':>7}")
    for pi,phi in enumerate(PHIS):
        Cs=0.5*(C[pi]+C[pi].T); S=np.linalg.inv(Cs)
        E_ib=1/S[0,0]
        # effective K, mu from tensor (isotropic average)
        K_ib=(Cs[0,0]+2*Cs[0,1])/3.0
        mu_ib=Cs[3,3]
        K_mt,mu_mt,E_mt=mt_sphere(phi,Km,mum,Ki,mui)
        print(f"{phi:6.2f} {K_ib:9.1f} {K_mt:9.1f} {mu_ib:9.1f} {mu_mt:9.1f} {E_ib:9.1f} {E_mt:9.1f} {abs(E_ib-E_mt)/E_mt*100:6.1f}%")
