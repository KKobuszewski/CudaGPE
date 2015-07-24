#ifndef __CUDA_COMPLEX_EXT_H__
#define __CUDA_COMPLEX_EXT_H__

#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuComplex.h>
    
 
__device__ static __inline__ double cuCarg(cuDoubleComplex z)
{
    return atan2( cuCimag(z), cuCreal(z) );
}
 
 
__device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex x)
{
    double factor = exp(x.x);
    return make_cuDoubleComplex(factor * cos(x.y), factor * sin(x.y));
}

__device__ static inline double cuCSqAbs(cuDoubleComplex z)
{
    return cuCreal(z)*cuCreal(z) + cuCimag(z)*cuCimag(z);
}


__device__ static __inline__ double cuCnorm(cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c)
{
    double absA = cuCabs(a);
    double absB = cuCabs(b);
    double absC = cuCabs(c);
    double u, v, w, r, s;
    
    if (absA > absB){
    	if (absA > absC){
    		// a is largest
    		u = absA;
    		v = absB;
    		w = absC;
    
    	}
    	else{
    		// c is largest
    		u = absC;
    		v = absA;
    		w = absB;
    	}
    }
    else{
    	if (absB > absC){
    		// b is largest
    		u = absB;
    		v = absC;
    		w = absA;
    	}
    	else{
    		// c is largest
    		u = absC;
    		v = absA;
    		w = absB;
    	}
    }
    r = v / u;
    s = w / u;
    r = 1.0 + r*r + s*s;
    r = u * sqrt(r);
    
    if ((u == 0.0) ||
    	(u > 1.79769313486231570e+308) || (v > 1.79769313486231570e+308) || (w > 1.79769313486231570e+308)) {
    	r = u + v + w;
    }
    
    return r;
}

__device__ static __inline__ cuDoubleComplex cuCsqrt(cuDoubleComplex x)
{
    double radius = cuCabs(x);
    double cosA = x.x / radius;
    cuDoubleComplex out;
    out.x = sqrt(radius * (cosA + 1.0) / 2.0);
    out.y = sqrt(radius * (1.0 - cosA) / 2.0);
    // signbit should be false if x.y is negative
    if (signbit(x.y))
    	out.y *= -1.0;
    
    return out;
}


__device__ static __inline__ cuDoubleComplex cuCadd(cuDoubleComplex x, double y)
{
    return make_cuDoubleComplex(cuCreal(x) + y, cuCimag(x));
}

__device__ static __inline__ cuDoubleComplex cuCdiv(cuDoubleComplex x, double y)
{
    return make_cuDoubleComplex(cuCreal(x) / y, cuCimag(x) / y);
}

__device__ static __inline__ cuDoubleComplex cuCmul(cuDoubleComplex x, double y)
{
    return make_cuDoubleComplex(cuCreal(x) * y, cuCimag(x) * y);
}

__device__ static __inline__ cuDoubleComplex cuCsub(cuDoubleComplex x, double y)
{
    return make_cuDoubleComplex(cuCreal(x) - y, cuCimag(x));
}

#endif