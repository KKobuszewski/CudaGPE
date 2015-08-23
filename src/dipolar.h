#include "global.h"
#include <stdlib.h>
#include "gsl/gsl_sf_expint.h"

/*
 * Function x*exp(x)*expint_E1(x)
 * For underflow in gsl_sf_expint_E1 it is replaced with Pade approximant (formula 5.1.56 from Abramowitz & Stegun page 231, http://people.math.sfu.ca/~cbm/aands/page_231.htm)
 */

// coefficients to Pade approximant above 1
const long double a[] = {8.5733287401, 18.0590169730, 8.6347608925, 0.2677737343};
const long double b[] = {9.5733223454, 25.6329561486, 21.0996530827, 3.9584969228};

inline double func(long double k) {
    return ( k < 700.) ? 
           ((double) ( ((long double) k) * expl((long double) k) * ((long double) gsl_sf_expint_E1( ((double) k) )) )) : 
           ((double) ( powl(k,4) + a[0]*powl(k,3) + a[1]*powl(k,2) + a[2]*k + a[3] )/( powl(k,4) + b[0]*powl(k,3) + b[1]*powl(k,2) + b[2]*k + b[3] ) );
}

inline long double funcl(long double k) {
    return ( k < 700.) ? 
           ( ((long double) k) * expl((long double) k) * ((long double) gsl_sf_expint_E1( ((double) k) )) ) : 
           (( powl(k,4) + a[0]*powl(k,3) + a[1]*powl(k,2) + a[2]*k + a[3] )/( powl(k,4) + b[0]*powl(k,3) + b[1]*powl(k,2) + b[2]*k + b[3] )) ;
}


/*
 * Expression for fourier transform of dipolar potential.
 * func fails when k == 0, because of domain of expint_E1
 */
static inline double Vdd(long double k, long double a_dd, long double as) {
    return ( k == 0 ) ? ((double) 3.*a_dd ) : ((double) 3.*a_dd*( 1. - funcl( as*k*k/2.0 ) ));
}