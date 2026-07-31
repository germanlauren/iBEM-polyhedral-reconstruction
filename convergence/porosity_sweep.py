#!/usr/bin/env python3
"""Resumable building-block RVE porosity sweep. Solid cuticle matrix (E_s=12 GPa,
nu=0.35) with a compliant near-void spherical inclusion (contrast 1e-3) at pore
fractions phi in {0.05,0.10,0.20,0.30}. Reports the effective through-thickness
Young modulus E3 to show the 8.2 GPa building-block value is not cherry-picked.
Checkpoints each (phi,case) to poro.npz; call until DONE."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "homogenization"))
import numpy as np, os, time, shutil, tempfile
import ibem_homog as H

Es=12000.0; nu=0.35; contrast=1e-3; L=1.0; n=3
PHIS=[0.05,0.10,0.20,0.30]
def r_of(phi): return (phi*3/(4*np.pi))**(1/3.)
ckpt="poro.npz"; cases=H.macro_cases()
if os.path.exists(ckpt):
    d=np.load(ckpt,allow_pickle=True); C=d["C"]; done=set(map(tuple,d["done"]))
else:
    C=np.zeros((len(PHIS),6,6)); done=set()
tmp=tempfile.mkdtemp(); b=os.path.join(tmp,"iBEM_run"); shutil.copy(H.BIN,b); os.chmod(b,0o755)
nodes,quads=H.cube_mesh(L,n); pts=H.interior_grid(L,6)
t0=time.time(); BUDGET=30.0
try:
    for pi,phi in enumerate(PHIS):
        r=r_of(phi); sph=[(0.5,0.5,0.5,r)]
        for col in range(6):
            if (pi,col) in done: continue
            if time.time()-t0>BUDGET: raise TimeoutError
            sig=H.run_case_boundary(b,tmp,Es,nu,Es*contrast,nu,nodes,quads,sph,pts,cases[col],L)
            C[pi,:,col]=sig; done.add((pi,col))
            np.savez(ckpt,C=C,done=np.array(sorted(done)))
            print(f"phi={phi:.2f} case {col} done")
except TimeoutError:
    print("BUDGET hit -> call again")
finally:
    shutil.rmtree(tmp,ignore_errors=True)
if len(done)==len(PHIS)*6:
    print("\n=== DONE: building-block RVE porosity sweep (E_s=12 GPa) ===")
    print(f"{'phi_RVE':>8} {'r':>6} {'E1(GPa)':>8} {'E2':>8} {'E3(GPa)':>8}")
    for pi,phi in enumerate(PHIS):
        Cs=0.5*(C[pi]+C[pi].T); S=np.linalg.inv(Cs)
        E1=1/S[0,0]/1000; E2=1/S[1,1]/1000; E3=1/S[2,2]/1000
        print(f"{phi:8.2f} {r_of(phi):6.3f} {E1:8.2f} {E2:8.2f} {E3:8.2f}")
