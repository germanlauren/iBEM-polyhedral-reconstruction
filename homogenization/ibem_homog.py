#!/usr/bin/env python3
"""iBEM homogenization harness (3D ellipsoidal/sphere path).
Builds a cubic RVE (BEM surface mesh) with KUBC displacement BCs u=E.x,
places stiff spheres (soft/solid matrix), runs iBEM for 6 unit macro-strain
cases, samples interior stress on a grid and assembles the effective
stiffness tensor C_eff (Voigt [xx,yy,zz,yz,xz,xy], engineering shears)."""
import numpy as np, subprocess, os, shutil, tempfile

BIN = "/sessions/adoring-wizardly-faraday/mnt/outputs/iBEM_src/iBEM-1.0.0/build_linux/iBEM_run"

# ---------- cube surface mesh (structured quads, outward normals) ----------
def cube_mesh(L, n):
    # boundary nodes of an (n+1)^3 grid, deduplicated
    idx = {}
    coords = []
    def nid(i,j,k):
        key=(i,j,k)
        if key not in idx:
            idx[key]=len(coords)
            coords.append((i*L/n, j*L/n, k*L/n))
        return idx[key]
    quads=[]
    def face(fixed_axis, fixed_val, outward_sign):
        # loop over the two free axes
        ax=[a for a in range(3) if a!=fixed_axis]
        for a in range(n):
            for b in range(n):
                pts=[]
                for (da,db) in [(0,0),(1,0),(1,1),(0,1)]:
                    c=[0,0,0]; c[fixed_axis]=fixed_val
                    c[ax[0]]=a+da; c[ax[1]]=b+db
                    pts.append(nid(*c))
                # ensure outward normal via ordering
                p=[np.array(coords[q]) for q in pts]
                nrm=np.cross(p[1]-p[0],p[3]-p[0])
                e=np.zeros(3); e[fixed_axis]=outward_sign
                if np.dot(nrm,e)<0: pts=pts[::-1]
                quads.append(pts)
    face(0,0,-1); face(0,n,+1)
    face(1,0,-1); face(1,n,+1)
    face(2,0,-1); face(2,n,+1)
    return np.array(coords), np.array(quads,dtype=int)

# ---------- writers ----------
def write_bem(path, E0, nu0, nodes, quads, Emac, title="RVE"):
    NN=len(nodes); NE=len(quads)
    with open(path,"w") as f:
        f.write(f"{title}\n{E0}\n{nu0}\n{NN}\n{NE}\n")
        for x,y,z in nodes: f.write(f"{x:.10g} {y:.10g} {z:.10g}\n")
        for q in quads: f.write(" ".join(str(i+1) for i in q)+"\n")  # 1-based
        # BC: 3*NN rows 'flag value'; KUBC -> flag 0 (disp prescribed) u=E.x
        for a in range(NN):
            u=Emac@nodes[a]
            for c in range(3): f.write(f"0 {u[c]:.10g}\n")
        # traction BC: NE rows x 24; all tractions unknown (flag 0), value 0
        for _ in range(NE): f.write(" ".join(["0"]*12+["0"]*12)+"\n")

def write_position(path, spheres, E1, nu1):
    with open(path,"w") as f:
        f.write(f"{len(spheres)}\n{E1}\n{nu1}\n")
        for (x,y,z,r) in spheres: f.write(f"{x:.10g} {y:.10g} {z:.10g} {r:.10g}\n")

def write_post(path, pts):
    with open(path,"w") as f:
        f.write(f"{len(pts)}\n")
        for x,y,z in pts: f.write(f"{x:.10g} {y:.10g} {z:.10g}\n")

def interior_grid(L, m, pad=0.06):
    xs=np.linspace(pad*L,(1-pad)*L,m)
    P=np.array([(x,y,z) for x in xs for y in xs for z in xs])
    return P

# stress.txt order: [sxx, sxy, sxz, syy, syz, szz] -> Voigt [xx,yy,zz,yz,xz,xy]
def to_voigt(s):
    sxx,sxy,sxz,syy,syz,szz=s.T
    return np.stack([sxx,syy,szz,syz,sxz,sxy],axis=1)

# 6 unit macro-strain tensors (engineering shear = 1)
def macro_cases():
    E=[]
    for (i,j,val) in [(0,0,1),(1,1,1),(2,2,1)]:
        M=np.zeros((3,3)); M[i,j]=val; E.append(M)
    for (i,j) in [(1,2),(0,2),(0,1)]:
        M=np.zeros((3,3)); M[i,j]=0.5; M[j,i]=0.5; E.append(M)  # eps=0.5 -> gamma=1
    return E

def run_case(binpath, workdir, E0,nu0,E1,nu1, nodes,quads, spheres, pts, Emac):
    write_bem(os.path.join(workdir,"BEM.txt"),E0,nu0,nodes,quads,Emac)
    write_position(os.path.join(workdir,"position.txt"),spheres,E1,nu1)
    write_post(os.path.join(workdir,"postprocess.txt"),pts)
    with open(os.path.join(workdir,"calculationType.txt"),"w") as f: f.write("1\n")
    r=subprocess.run([binpath],cwd=workdir,capture_output=True,text=True,timeout=600)
    sf=os.path.join(workdir,"stress.txt")
    if not os.path.exists(sf): raise RuntimeError("no stress.txt\n"+r.stdout[-500:]+r.stderr[-500:])
    S=np.loadtxt(sf)
    if S.ndim==1: S=S[None,:]
    return S

def homogenize(E0,nu0,E1,nu1, L=1.0, n=4, spheres=None, msample=8, verbose=True):
    nodes,quads=cube_mesh(L,n)
    if spheres is None: spheres=[(L/2,L/2,L/2,1e-4*L)]  # negligible dummy
    pts=interior_grid(L,msample)
    C=np.zeros((6,6))
    tmp=tempfile.mkdtemp(prefix="ibem_")
    binlocal=os.path.join(tmp,"iBEM_run"); shutil.copy(BIN,binlocal); os.chmod(binlocal,0o755)
    try:
        for col,Emac in enumerate(macro_cases()):
            S=run_case(binlocal,tmp,E0,nu0,E1,nu1,nodes,quads,spheres,pts,Emac)
            savg=to_voigt(S).mean(axis=0)   # <sigma> Voigt
            C[:,col]=savg
            if verbose: print(f" case {col}: <sigma>={np.round(savg,4)}")
    finally:
        shutil.rmtree(tmp,ignore_errors=True)
    return 0.5*(C+C.T)

def iso_C(E,nu):
    lam=E*nu/((1+nu)*(1-2*nu)); mu=E/(2*(1+nu))
    C=np.zeros((6,6))
    for i in range(3):
        for j in range(3): C[i,j]=lam
        C[i,i]+=2*mu
    for i in range(3,6): C[i,i]=mu
    return C

if __name__=="__main__":
    print("=== MACHINERY TEST: unit contrast (E1=E0) should give C_eff = C_matrix ===")
    E0,nu0=1000.0,0.3
    C=homogenize(E0,nu0,E0,nu0, L=1.0,n=4, spheres=[(0.5,0.5,0.5,0.15)], msample=8)
    C0=iso_C(E0,nu0)
    print("C_eff=\n",np.round(C,2))
    print("C_matrix=\n",np.round(C0,2))
    err=np.abs(C-C0).max()/np.abs(C0).max()
    print(f"max relative error vs matrix: {err:.3e}")

# ---------- exact <sigma> from boundary tractions: (1/V) integral t_i x_j dS ----------
def sigma_avg_boundary(Tvec, nodes, quads, L):
    gp=1.0/np.sqrt(3.0)
    GP=[(-gp,-gp),(gp,-gp),(gp,gp),(-gp,gp)]
    def shp(xi,eta):
        N=np.array([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])/4.0
        dxi=np.array([-(1-eta),(1-eta),(1+eta),-(1+eta)])/4.0
        deta=np.array([-(1-xi),-(1+xi),(1+xi),(1-xi)])/4.0
        return N,dxi,deta
    sig=np.zeros((3,3)); V=L**3
    for e,q in enumerate(quads):
        X=nodes[q]
        tn=np.array([[Tvec[12*e+3*a+c] for c in range(3)] for a in range(4)])
        for (xi,eta) in GP:
            N,dxi,deta=shp(xi,eta)
            xg=N@X; tg=N@tn; xk=dxi@X; xe=deta@X
            J=np.linalg.norm(np.cross(xk,xe))
            sig+=J*np.outer(tg,xg)
    sig/=V
    sig=0.5*(sig+sig.T)
    return np.array([sig[0,0],sig[1,1],sig[2,2],sig[1,2],sig[0,2],sig[0,1]])

def run_case_boundary(binpath, workdir, E0,nu0,E1,nu1, nodes,quads, spheres, pts, Emac, L):
    run_case(binpath,workdir,E0,nu0,E1,nu1,nodes,quads,spheres,pts,Emac)
    T=np.loadtxt(os.path.join(workdir,"T.txt"))
    return sigma_avg_boundary(T,nodes,quads,L)
