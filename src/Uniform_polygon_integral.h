/* Reconstructed header for the Uniform_Polygon class.
   The iBEM 1.0.0 release (and the upstream iBemResearch/iBEM repo) ship
   Uniform_polygon_integral.cpp but NOT its header. All declarations below are
   generated 1:1 from the method definitions in that .cpp, so this is exact. */
#pragma once
class Uniform_Polygon {
public:
    double PHI_o(double b, double lm, double lp);
    double PHI(double b, double lm, double lp);
    double Phi_b(double b, double lm, double lp);
    double Phi_lp(double b, double lp);
    double Phi_lm(double b, double lm);
    double PHI_j(int j, double b, double lm, double lp, double* Vnorm, double* Vdir);
    double Psi(double b, double lm, double lp);
    double Psi_b(double b, double lm, double lp);
    double Psi_lp(double b, double lp);
    double Psi_lm(double b, double lp);
    double Psi_2b(double b, double lm, double lp);
    double Psi_lp_b(double b, double lp);
    double Psi_lm_b(double b, double lm);
    double Psi_2lp(double b, double lp);
    double Psi_2lm(double b, double lm);
    double PSI_jk(int j, int k, double b, double lm, double lp, double* Vnorm, double* Vdir);
    double Psi_3b(double b, double lm, double lp);
    double Psi_3lp(double b, double lp);
    double Psi_3lm(double b, double lm);
    double Psi_2lp_b(double b, double lp);
    double Psi_2lm_b(double b, double lm);
    double Psi_lp_2b(double b, double lp);
    double Psi_lm_2b(double b, double lm);
    double PSI_j(int j, double b, double lm, double lp, double* Vnorm, double* Vdir);
    double PSI_jkl(int j, int k, int l, double b, double lm, double lp, double* Vnorm, double* Vdir);
    double PHI_1(int i, double b, double lm, double lp, double* Vnorm, double* Vdir);
    double PHI_2(int i, int j, double b, double lm, double lp, double* Vnorm, double* Vdir);
    double PSI_1(int i, double b, double lm, double lp, double* Vnorm, double* Vdir);
    double PSI_2(int i, int j, double b, double lm, double lp, double* Vnorm, double* Vdir);
    double PSI_3(int i, int j, int k, double b, double lm, double lp, double* Vnorm, double* Vdir);
    double PSI_4(int i, int j, int k, int l, double b, double lm, double lp, double* Vnorm, double* Vdir);
};
