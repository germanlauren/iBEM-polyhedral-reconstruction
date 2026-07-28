#pragma once
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
