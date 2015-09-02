#ifndef __SIMULATION_H__
#define __SIMULATION_H__


#include "dipolar.h" // declaration of Vdd

void* simulation_thread(void* passing_ptr);
void* helper_thread(void* passing_ptr);


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							FUNC DEFINITIONS								 *
 * 																	 *
 * ************************************************************************************************************************************* */

inline void alloc_device();
inline void alloc_host();
inline void free_device();
inline void free_host();

inline void init_wavefunction();
inline void create_propagators();

inline void init_cufft();
inline void init_cublas();

inline void cpy_data_to_host();
inline void save_stats_host(uint64_t step_index);

inline void save_stats_dev(uint64_t step_index);
inline void save_simulation_params();


/* ************************************************************************************************************************************* *
 * 																	 *
 * 							COMPUTIONAL FUNCTIONS								 *
 * 																	 *
 * ************************************************************************************************************************************* */

/*
 * This function counts chemical potential in time-independent Gross-Pitaevskii equation from relative norm change during ITE step.
 * ( Wavefunction is being normed every step, so the relative change is the value of norm in current step! )
 * 
 */
static inline long double chemical_potential_ite(long double norm) {
    return -logl(norm)/( (long double) DT);
}
/*
 * Less accurate but faster version of function above.
 */
static inline double chemical_potential_ite(double norm) {
    return -log(norm)/DT;
}

// interaction potential
static inline double interaction_potential(uint32_t ii) {
    return Vdd(kx(ii),1.,1.) + G_CONTACT; // dipolar + contact interactions
}


#endif