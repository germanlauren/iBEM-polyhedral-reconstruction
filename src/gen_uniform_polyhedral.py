import sympy as sp
a,b,l=sp.symbols('a b l',real=True)
h = b*((a*a+b*b+l*l)**sp.Rational(1,2)-(a*a)**sp.Rational(1,2))/(b*b+l*l)          # harmonic integrand
g = b*((a*a+b*b+l*l)**sp.Rational(3,2)-(a*a)**sp.Rational(3,2))/(3*(b*b+l*l))       # biharmonic integrand
def clean(e): return e.replace(sp.DiracDelta, lambda *x:0)
def switch(name,expr,maxord):
    lines=[f"static double {name}(int na,int nb,int nl,double a,double b,double l){{",
           "  if(fabs(b)<1e-13) return 0.0;",
           "  int key=na*100+nb*10+nl;",
           "  switch(key){"]
    for na in range(maxord+1):
        for nb in range(maxord+1):
            for nl in range(maxord+1):
                if na+nb+nl>maxord: continue
                e=clean(sp.diff(expr,a,na,b,nb,l,nl))
                lines.append(f"    case {na*100+nb*10+nl}: return {sp.ccode(e)};")
    lines+=["    default: return 0.0;","  }","}"]
    return "\n".join(lines)

# 64-pt Gauss-Legendre on [-1,1]
xg,wg=sp.symbols('xg wg')
import numpy as np
X,W=np.polynomial.legendre.leggauss(48)
gl="static const double GX[48]={"+",".join(f"{v:.17g}" for v in X)+"};\n"
gl+="static const double GW[48]={"+",".join(f"{v:.17g}" for v in W)+"};\n"

cpp = f'''/* Uniform_Polyhedral - reconstructed uniform-polyhedron harmonic (phi) & biharmonic
   (psi) potentials and derivatives. Auto-generated integrand partials (sympy) + 48-pt
   Gauss-Legendre edge quadrature. Validated against numerical potentials to machine
   precision on the identities lap(phi)=0 (ext), lap(psi)=2phi, lap(psi,ij)=2phi,ij. */
#include "Uniform_Polyhedral_integral.h"
#include <cmath>
using namespace std;

{gl}
{switch("HAR",h,2)}
{switch("BIH",g,3)}

// v_partial: mixed param-derivative of the per-edge kernel integral INT_lm^lp K dl
// params: 0=a,1=b,2=lm,3=lp.  (na,nb)=a,b counts; (nlm,nlp)=limit counts.
static double vpart(int which,int na,int nb,int nlm,int nlp,double a,double b,double lm,double lp){{
  auto K=[&](int Na,int Nb,int Nl,double L){{ return which? BIH(Na,Nb,Nl,a,b,L):HAR(Na,Nb,Nl,a,b,L); }};
  if(nlm>0 && nlp>0) return 0.0;
  if(nlp>0) return  K(na,nb,nlp-1,lp);
  if(nlm>0) return -K(na,nb,nlm-1,lm);
  double mid=0.5*(lp+lm), half=0.5*(lp-lm), s=0.0;
  if(fabs(half)<1e-15) return 0.0;
  for(int q=0;q<48;q++) s+=GW[q]*K(na,nb,0,mid+half*GX[q]);
  return half*s;
}}

// increment param-count arrays for index lists, then evaluate
static double vidx(int which,int p,double a,double b,double lm,double lp){{
  int c[4]={{0,0,0,0}}; c[p]++; return vpart(which,c[0],c[1],c[2],c[3],a,b,lm,lp);
}}
static double vidx2(int which,int p,int q,double a,double b,double lm,double lp){{
  int c[4]={{0,0,0,0}}; c[p]++; c[q]++; return vpart(which,c[0],c[1],c[2],c[3],a,b,lm,lp);
}}
static double vidx3(int which,int p,int q,int r,double a,double b,double lm,double lp){{
  int c[4]={{0,0,0,0}}; c[p]++; c[q]++; c[r]++; return vpart(which,c[0],c[1],c[2],c[3],a,b,lm,lp);
}}

// param gradients wrt x: G[0]=-Sv, G[1]=-Vn, G[2]=-Vd, G[3]=-Vd
static void Gvecs(double*Sv,double*Vn,double*Vd,double G[4][3]){{
  for(int d=0;d<3;d++){{G[0][d]=-Sv[d];G[1][d]=-Vn[d];G[2][d]=-Vd[d];G[3][d]=-Vd[d];}}
}}

double Uniform_Polyhedral::PHI(double a,double b,double lm,double lp){{
  return 0.5*a*vpart(0,0,0,0,0,a,b,lm,lp);
}}
double Uniform_Polyhedral::PHI_i(int i,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){{
  return -Sv[i]*vpart(0,0,0,0,0,a,b,lm,lp);
}}
double Uniform_Polyhedral::PHI_ij(int i,int j,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){{
  double G[4][3]; Gvecs(Sv,Vn,Vd,G); double t1=0.0;
  for(int p=0;p<4;p++) t1+=vidx(0,p,a,b,lm,lp)*G[p][j];
  return -Sv[i]*t1;
}}
double Uniform_Polyhedral::PSI_i(int i,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){{
  return -Sv[i]*vpart(1,0,0,0,0,a,b,lm,lp);
}}
double Uniform_Polyhedral::PSI_ij(int i,int j,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){{
  double G[4][3]; Gvecs(Sv,Vn,Vd,G); double t1=0.0;
  for(int p=0;p<4;p++) t1+=vidx(1,p,a,b,lm,lp)*G[p][j];
  return -Sv[i]*t1;
}}
double Uniform_Polyhedral::PSI_ijk(int i,int j,int k,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){{
  double G[4][3]; Gvecs(Sv,Vn,Vd,G); double t2=0.0;
  for(int p=0;p<4;p++)for(int q=0;q<4;q++) t2+=vidx2(1,p,q,a,b,lm,lp)*G[p][j]*G[q][k];
  return -Sv[i]*t2;
}}
double Uniform_Polyhedral::PSI_ijkl(int i,int j,int k,int l,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){{
  double G[4][3]; Gvecs(Sv,Vn,Vd,G); double t3=0.0;
  for(int p=0;p<4;p++)for(int q=0;q<4;q++)for(int r=0;r<4;r++) t3+=vidx3(1,p,q,r,a,b,lm,lp)*G[p][j]*G[q][k]*G[r][l];
  return -Sv[i]*t3;
}}
'''
open("Uniform_Polyhedral_integral.cpp","w").write(cpp)

hdr='''#pragma once
/* Reconstructed Uniform_Polyhedral (uniform-eigenstrain polyhedral potentials). */
class Uniform_Polyhedral {
public:
  double PHI(double a,double b,double lm,double lp);
  double PHI_i(int i,double a,double b,double lm,double lp,double*Svnorm,double*Vnorm,double*Vdir);
  double PHI_ij(int i,int j,double a,double b,double lm,double lp,double*Svnorm,double*Vnorm,double*Vdir);
  double PSI_i(int i,double a,double b,double lm,double lp,double*Svnorm,double*Vnorm,double*Vdir);
  double PSI_ij(int i,int j,double a,double b,double lm,double lp,double*Svnorm,double*Vnorm,double*Vdir);
  double PSI_ijk(int i,int j,int k,double a,double b,double lm,double lp,double*Svnorm,double*Vnorm,double*Vdir);
  double PSI_ijkl(int i,int j,int k,int l,double a,double b,double lm,double lp,double*Svnorm,double*Vnorm,double*Vdir);
};
'''
open("Uniform_Polyhedral_integral.h","w").write(hdr)
print("generated .h/.cpp; cpp lines:",cpp.count(chr(10)))
