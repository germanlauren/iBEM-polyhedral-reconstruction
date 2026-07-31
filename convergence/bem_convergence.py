#!/usr/bin/env python3
"""Resumable BEM-discretization convergence. Unit-contrast cube (E1=E0): the exact
homogenized tensor is the isotropic matrix tensor, so the numerical departure is
the pure boundary-mesh + quadrature bias. Runs meshes n in {2,3,4}, 6 unit-strain
cases each, using the exact boundary-traction stress average (Eq. 19). Checkpoints
each (mesh,case) to bem_conv.npz; call repeatedly until DONE."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "homogenization"))
import numpy as np, os, sys, time, shutil, tempfile
import ibem_homog as H

E0,nu0=1000.0,0.30
MESHES=[2,3,4]
L=1.0
ckpt="bem_conv.npz"
cases=H.macro_cases()

# state: C[mesh_index] 6x6, done set of (mi,case)
if os.path.exists(ckpt):
    d=np.load(ckpt,allow_pickle=True)
    C=d["C"]; done=set(map(tuple,d["done"]))
else:
    C=np.zeros((len(MESHES),6,6)); done=set()

tmp=tempfile.mkdtemp(); b=os.path.join(tmp,"iBEM_run"); shutil.copy(H.BIN,b); os.chmod(b,0o755)
t0=time.time(); BUDGET=30.0
try:
    for mi,n in enumerate(MESHES):
        nodes,quads=H.cube_mesh(L,n); pts=H.interior_grid(L,6)
        for col in range(6):
            if (mi,col) in done: continue
            if time.time()-t0>BUDGET: raise TimeoutError
            sig=H.run_case_boundary(b,tmp,E0,nu0,E0,nu0,nodes,quads,
                                    [(0.5,0.5,0.5,1e-4)],pts,cases[col],L)
            C[mi,:,col]=sig; done.add((mi,col))
            np.savez(ckpt,C=C,done=np.array(sorted(done)))
            print(f"mesh n={n} case {col} done: {np.round(sig,3)}")
except TimeoutError:
    print("BUDGET hit -> call again")
finally:
    shutil.rmtree(tmp,ignore_errors=True)

if len(done)==len(MESHES)*6:
    C0=H.iso_C(E0,nu0)
    print("\n=== DONE: BEM discretization convergence (unit contrast) ===")
    print(f"{'n':>3} {'elems':>6} {'C11':>9} {'C33':>9} {'C44':>9} {'norm-bias%':>11} {'e_sym':>10}")
    for mi,n in enumerate(MESHES):
        Cs=0.5*(C[mi]+C[mi].T)
        nb=max(abs(Cs[0,0]-C0[0,0]),abs(Cs[1,1]-C0[1,1]),abs(Cs[2,2]-C0[2,2]))/C0[0,0]*100
        esym=np.linalg.norm(C[mi]-C[mi].T)/np.linalg.norm(C[mi])
        nel=6*n*n
        print(f"{n:3d} {nel:6d} {Cs[0,0]:9.2f} {Cs[2,2]:9.2f} {Cs[3,3]:9.2f} {nb:11.2f} {esym:10.2e}")
    print(f"exact matrix: C11={C0[0,0]:.2f} C33={C0[2,2]:.2f} C44={C0[3,3]:.2f}")
