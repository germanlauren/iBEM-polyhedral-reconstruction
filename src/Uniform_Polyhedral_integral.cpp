/* Uniform_Polyhedral - reconstructed uniform-polyhedron harmonic (phi) & biharmonic
   (psi) potentials and derivatives. Auto-generated integrand partials (sympy) + 48-pt
   Gauss-Legendre edge quadrature. Validated against numerical potentials to machine
   precision on the identities lap(phi)=0 (ext), lap(psi)=2phi, lap(psi,ij)=2phi,ij. */
#include "Uniform_Polyhedral_integral.h"
#include <cmath>
using namespace std;

static const double GX[48]={-0.99877100725242607,-0.99353017226635076,-0.98412458372282685,-0.97059159254624727,-0.95298770316043091,-0.93138669070655433,-0.90587913671556963,-0.87657202027424785,-0.84358826162439349,-0.80706620402944262,-0.76715903251574036,-0.72403413092381463,-0.67787237963266389,-0.6288673967765136,-0.57722472608397268,-0.523160974722233,-0.46690290475095841,-0.40868648199071672,-0.34875588629216075,-0.28736248735545555,-0.22476379039468905,-0.16122235606889171,-0.097004699209462697,-0.032380170962869367,0.032380170962869367,0.097004699209462697,0.16122235606889171,0.22476379039468905,0.28736248735545555,0.34875588629216075,0.40868648199071672,0.46690290475095841,0.523160974722233,0.57722472608397268,0.6288673967765136,0.67787237963266389,0.72403413092381463,0.76715903251574036,0.80706620402944262,0.84358826162439349,0.87657202027424785,0.90587913671556963,0.93138669070655433,0.95298770316043091,0.97059159254624727,0.98412458372282685,0.99353017226635076,0.99877100725242607};
static const double GW[48]={0.0031533460523091796,0.0073275539012764923,0.011477234579234974,0.015579315722942928,0.019616160457355297,0.023570760839324092,0.027426509708356882,0.031167227832798339,0.034777222564770657,0.038241351065830674,0.041545082943464554,0.0446745608566941,0.047616658492490284,0.050359035553854278,0.052890189485193487,0.055199503699984054,0.05727729210040293,0.059114839698395483,0.060704439165893583,0.062039423159892464,0.063114192286253784,0.063924238584647949,0.064466164435949838,0.064737696812683682,0.064737696812683682,0.064466164435949838,0.063924238584647949,0.063114192286253784,0.062039423159892464,0.060704439165893583,0.059114839698395483,0.05727729210040293,0.055199503699984054,0.052890189485193487,0.050359035553854278,0.047616658492490284,0.0446745608566941,0.041545082943464554,0.038241351065830674,0.034777222564770657,0.031167227832798339,0.027426509708356882,0.023570760839324092,0.019616160457355297,0.015579315722942928,0.011477234579234974,0.0073275539012764923,0.0031533460523091796};

static double HAR(int na,int nb,int nl,double a,double b,double l){
  if(fabs(b)<1e-13) return 0.0;
  int key=na*100+nb*10+nl;
  switch(key){
    case 0: return b*(sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - fabs(a))/(pow(b, 2) + pow(l, 2));
    case 1: return b*l/((pow(b, 2) + pow(l, 2))*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))) - 2*b*l*(sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - fabs(a))/pow(pow(b, 2) + pow(l, 2), 2);
    case 2: return b*(-4*pow(l, 2)/((pow(b, 2) + pow(l, 2))*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))) - (pow(l, 2)/(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 1)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*(4*pow(l, 2)/(pow(b, 2) + pow(l, 2)) - 1)*(sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - fabs(a))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 10: return pow(b, 2)/((pow(b, 2) + pow(l, 2))*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))) - 2*pow(b, 2)*(sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - fabs(a))/pow(pow(b, 2) + pow(l, 2), 2) + (sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - fabs(a))/(pow(b, 2) + pow(l, 2));
    case 11: return l*(-pow(b, 2)/pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0) - 4*pow(b, 2)/((pow(b, 2) + pow(l, 2))*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))) + 8*pow(b, 2)*(sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - fabs(a))/pow(pow(b, 2) + pow(l, 2), 2) + pow(pow(a, 2) + pow(b, 2) + pow(l, 2), -1.0/2.0) - 2*(sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - fabs(a))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 20: return b*(-4*pow(b, 2)/((pow(b, 2) + pow(l, 2))*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))) - (pow(b, 2)/(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 1)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*(4*pow(b, 2)/(pow(b, 2) + pow(l, 2)) - 1)*(sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - fabs(a))/(pow(b, 2) + pow(l, 2)) - 4*(sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - fabs(a))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 100: return b*(a/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - (((a) > 0) - ((a) < 0)))/(pow(b, 2) + pow(l, 2));
    case 101: return -b*l*(a/pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0) + 2*(a/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - (((a) > 0) - ((a) < 0)))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 110: return (-a*pow(b, 2)/pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0) + a/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 2*pow(b, 2)*(a/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - (((a) > 0) - ((a) < 0)))/(pow(b, 2) + pow(l, 2)) - (((a) > 0) - ((a) < 0)))/(pow(b, 2) + pow(l, 2));
    case 200: return -b*(pow(a, 2)/pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0) - 1/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    default: return 0.0;
  }
}
static double BIH(int na,int nb,int nl,double a,double b,double l){
  if(fabs(b)<1e-13) return 0.0;
  int key=na*100+nb*10+nl;
  switch(key){
    case 0: return b*(-pow(a, 2)*fabs(a) + pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/(3*pow(b, 2) + 3*pow(l, 2));
    case 1: return 3*b*l*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(3*pow(b, 2) + 3*pow(l, 2)) - 6*b*l*(-pow(a, 2)*fabs(a) + pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/pow(3*pow(b, 2) + 3*pow(l, 2), 2);
    case 2: return b*(pow(l, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 4*pow(l, 2)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(pow(b, 2) + pow(l, 2)) + sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 2.0/3.0*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))*(4*pow(l, 2)/(pow(b, 2) + pow(l, 2)) - 1)/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 3: return b*l*(-(pow(l, 2)/(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 3)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 6*(4*pow(l, 2)/(pow(b, 2) + pow(l, 2)) - 1)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(pow(b, 2) + pow(l, 2)) - 6*(pow(l, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2)) + 8*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))*(2*pow(l, 2)/(pow(b, 2) + pow(l, 2)) - 1)/pow(pow(b, 2) + pow(l, 2), 2))/(pow(b, 2) + pow(l, 2));
    case 10: return 3*pow(b, 2)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(3*pow(b, 2) + 3*pow(l, 2)) - 6*pow(b, 2)*(-pow(a, 2)*fabs(a) + pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/pow(3*pow(b, 2) + 3*pow(l, 2), 2) + (-pow(a, 2)*fabs(a) + pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/(3*pow(b, 2) + 3*pow(l, 2));
    case 11: return l*(pow(b, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 4*pow(b, 2)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(pow(b, 2) + pow(l, 2)) - 8.0/3.0*pow(b, 2)*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/pow(pow(b, 2) + pow(l, 2), 2) + sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + (2.0/3.0)*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 12: return (-pow(b, 2)*pow(l, 2)/pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0) - 6*pow(b, 2)*pow(l, 2)/((pow(b, 2) + pow(l, 2))*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))) + 24*pow(b, 2)*pow(l, 2)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/pow(pow(b, 2) + pow(l, 2), 2) + 16*pow(b, 2)*pow(l, 2)*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/pow(pow(b, 2) + pow(l, 2), 3) + pow(b, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 4*pow(b, 2)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(pow(b, 2) + pow(l, 2)) - 8.0/3.0*pow(b, 2)*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/pow(pow(b, 2) + pow(l, 2), 2) + pow(l, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 4*pow(l, 2)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(pow(b, 2) + pow(l, 2)) - 8.0/3.0*pow(l, 2)*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/pow(pow(b, 2) + pow(l, 2), 2) + sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + (2.0/3.0)*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 20: return b*(pow(b, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 4*pow(b, 2)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(pow(b, 2) + pow(l, 2)) + 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 2.0/3.0*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))*(4*pow(b, 2)/(pow(b, 2) + pow(l, 2)) - 1)/(pow(b, 2) + pow(l, 2)) + (4.0/3.0)*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 21: return b*l*(-4*pow(b, 2)/((pow(b, 2) + pow(l, 2))*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))) + 16*pow(b, 2)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/pow(pow(b, 2) + pow(l, 2), 2) + (16.0/3.0)*pow(b, 2)*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/pow(pow(b, 2) + pow(l, 2), 3) - (pow(b, 2)/(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 1)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*(4*pow(b, 2)/(pow(b, 2) + pow(l, 2)) - 1)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(pow(b, 2) + pow(l, 2)) - 2*(pow(b, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2)) - 8*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(pow(b, 2) + pow(l, 2)) + (8.0/3.0)*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))*(4*pow(b, 2)/(pow(b, 2) + pow(l, 2)) - 1)/pow(pow(b, 2) + pow(l, 2), 2) - 16.0/3.0*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))/pow(pow(b, 2) + pow(l, 2), 2))/(pow(b, 2) + pow(l, 2));
    case 30: return (-pow(b, 2)*(pow(b, 2)/(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 3)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 3*pow(b, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 6*pow(b, 2)*(4*pow(b, 2)/(pow(b, 2) + pow(l, 2)) - 1)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(pow(b, 2) + pow(l, 2)) - 6*pow(b, 2)*(pow(b, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2)) - 12*pow(b, 2)*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))/(pow(b, 2) + pow(l, 2)) + 8*pow(b, 2)*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))*(2*pow(b, 2)/(pow(b, 2) + pow(l, 2)) - 1)/pow(pow(b, 2) + pow(l, 2), 2) + 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 2*(pow(a, 2)*fabs(a) - pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0))*(4*pow(b, 2)/(pow(b, 2) + pow(l, 2)) - 1)/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 100: return b*(-pow(a, 2)*(((a) > 0) - ((a) < 0)) + 3*a*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 2*a*fabs(a))/(3*pow(b, 2) + 3*pow(l, 2));
    case 101: return a*b*l*(pow(pow(a, 2) + pow(b, 2) + pow(l, 2), -1.0/2.0) + (2.0/3.0)*(a*(((a) > 0) - ((a) < 0)) - 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*fabs(a))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 102: return -a*b*(4*pow(l, 2)/((pow(b, 2) + pow(l, 2))*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))) + (pow(l, 2)/(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 1)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + (2.0/3.0)*(4*pow(l, 2)/(pow(b, 2) + pow(l, 2)) - 1)*(a*(((a) > 0) - ((a) < 0)) - 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*fabs(a))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 110: return a*(-1.0/3.0*a*(((a) > 0) - ((a) < 0)) + pow(b, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + (2.0/3.0)*pow(b, 2)*(a*(((a) > 0) - ((a) < 0)) - 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*fabs(a))/(pow(b, 2) + pow(l, 2)) + sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 2.0/3.0*fabs(a))/(pow(b, 2) + pow(l, 2));
    case 111: return a*l*(-pow(b, 2)/pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0) - 4*pow(b, 2)/((pow(b, 2) + pow(l, 2))*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))) - 8.0/3.0*pow(b, 2)*(a*(((a) > 0) - ((a) < 0)) - 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*fabs(a))/pow(pow(b, 2) + pow(l, 2), 2) + pow(pow(a, 2) + pow(b, 2) + pow(l, 2), -1.0/2.0) + (2.0/3.0)*(a*(((a) > 0) - ((a) < 0)) - 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*fabs(a))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 120: return a*b*(-4*pow(b, 2)/((pow(b, 2) + pow(l, 2))*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2))) - (pow(b, 2)/(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 1)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 2.0/3.0*(4*pow(b, 2)/(pow(b, 2) + pow(l, 2)) - 1)*(a*(((a) > 0) - ((a) < 0)) - 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*fabs(a))/(pow(b, 2) + pow(l, 2)) + (4.0/3.0)*(a*(((a) > 0) - ((a) < 0)) - 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*fabs(a))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 200: return -1.0/3.0*b*(-3*pow(a, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 4*a*(((a) > 0) - ((a) < 0)) - 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*fabs(a))/(pow(b, 2) + pow(l, 2));
    case 201: return b*l*(-(pow(a, 2)/(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 1)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + (2.0/3.0)*(-3*pow(a, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 4*a*(((a) > 0) - ((a) < 0)) - 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*fabs(a))/(pow(b, 2) + pow(l, 2)))/(pow(b, 2) + pow(l, 2));
    case 210: return (pow(a, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 4.0/3.0*a*(((a) > 0) - ((a) < 0)) - pow(b, 2)*(pow(a, 2)/(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 1)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + (2.0/3.0)*pow(b, 2)*(-3*pow(a, 2)/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 4*a*(((a) > 0) - ((a) < 0)) - 3*sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 2*fabs(a))/(pow(b, 2) + pow(l, 2)) + sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) - 2.0/3.0*fabs(a))/(pow(b, 2) + pow(l, 2));
    case 300: return -1.0/3.0*b*(3*pow(a, 3)/pow(pow(a, 2) + pow(b, 2) + pow(l, 2), 3.0/2.0) - 9*a/sqrt(pow(a, 2) + pow(b, 2) + pow(l, 2)) + 6*(((a) > 0) - ((a) < 0)))/(pow(b, 2) + pow(l, 2));
    default: return 0.0;
  }
}

// v_partial: mixed param-derivative of the per-edge kernel integral INT_lm^lp K dl
// params: 0=a,1=b,2=lm,3=lp.  (na,nb)=a,b counts; (nlm,nlp)=limit counts.
static double vpart(int which,int na,int nb,int nlm,int nlp,double a,double b,double lm,double lp){
  auto K=[&](int Na,int Nb,int Nl,double L){ return which? BIH(Na,Nb,Nl,a,b,L):HAR(Na,Nb,Nl,a,b,L); };
  if(nlm>0 && nlp>0) return 0.0;
  if(nlp>0) return  K(na,nb,nlp-1,lp);
  if(nlm>0) return -K(na,nb,nlm-1,lm);
  double mid=0.5*(lp+lm), half=0.5*(lp-lm), s=0.0;
  if(fabs(half)<1e-15) return 0.0;
  for(int q=0;q<48;q++) s+=GW[q]*K(na,nb,0,mid+half*GX[q]);
  return half*s;
}

// increment param-count arrays for index lists, then evaluate
static double vidx(int which,int p,double a,double b,double lm,double lp){
  int c[4]={0,0,0,0}; c[p]++; return vpart(which,c[0],c[1],c[2],c[3],a,b,lm,lp);
}
static double vidx2(int which,int p,int q,double a,double b,double lm,double lp){
  int c[4]={0,0,0,0}; c[p]++; c[q]++; return vpart(which,c[0],c[1],c[2],c[3],a,b,lm,lp);
}
static double vidx3(int which,int p,int q,int r,double a,double b,double lm,double lp){
  int c[4]={0,0,0,0}; c[p]++; c[q]++; c[r]++; return vpart(which,c[0],c[1],c[2],c[3],a,b,lm,lp);
}

// param gradients wrt x: G[0]=-Sv, G[1]=-Vn, G[2]=-Vd, G[3]=-Vd
static void Gvecs(double*Sv,double*Vn,double*Vd,double G[4][3]){
  for(int d=0;d<3;d++){G[0][d]=-Sv[d];G[1][d]=-Vn[d];G[2][d]=-Vd[d];G[3][d]=-Vd[d];}
}

double Uniform_Polyhedral::PHI(double a,double b,double lm,double lp){
  return 0.5*a*vpart(0,0,0,0,0,a,b,lm,lp);
}
double Uniform_Polyhedral::PHI_i(int i,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){
  return -Sv[i]*vpart(0,0,0,0,0,a,b,lm,lp);
}
double Uniform_Polyhedral::PHI_ij(int i,int j,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){
  double G[4][3]; Gvecs(Sv,Vn,Vd,G); double t1=0.0;
  for(int p=0;p<4;p++) t1+=vidx(0,p,a,b,lm,lp)*G[p][j];
  return -Sv[i]*t1;
}
double Uniform_Polyhedral::PSI_i(int i,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){
  return -Sv[i]*vpart(1,0,0,0,0,a,b,lm,lp);
}
double Uniform_Polyhedral::PSI_ij(int i,int j,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){
  double G[4][3]; Gvecs(Sv,Vn,Vd,G); double t1=0.0;
  for(int p=0;p<4;p++) t1+=vidx(1,p,a,b,lm,lp)*G[p][j];
  return -Sv[i]*t1;
}
double Uniform_Polyhedral::PSI_ijk(int i,int j,int k,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){
  double G[4][3]; Gvecs(Sv,Vn,Vd,G); double t2=0.0;
  for(int p=0;p<4;p++)for(int q=0;q<4;q++) t2+=vidx2(1,p,q,a,b,lm,lp)*G[p][j]*G[q][k];
  return -Sv[i]*t2;
}
double Uniform_Polyhedral::PSI_ijkl(int i,int j,int k,int l,double a,double b,double lm,double lp,double*Sv,double*Vn,double*Vd){
  double G[4][3]; Gvecs(Sv,Vn,Vd,G); double t3=0.0;
  for(int p=0;p<4;p++)for(int q=0;q<4;q++)for(int r=0;r<4;r++) t3+=vidx3(1,p,q,r,a,b,lm,lp)*G[p][j]*G[q][k]*G[r][l];
  return -Sv[i]*t3;
}
