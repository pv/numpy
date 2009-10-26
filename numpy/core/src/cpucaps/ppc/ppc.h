#ifndef _NPY_CPUCAPS_PPC_H_
#define _NPY_CPUCAPS_PPC_H_

typedef struct {
	int has_altivec;
} ppc_cpu_caps;

int ppc_get_caps(ppc_cpu_caps *cpu);

#endif
