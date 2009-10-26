#include <numpy/npy_compiler.h>
#include <numpy/npy_cpu.h>

#include "ppc.h"

/*
 * We enclose compiler specific assembly code in those ppc_* files:
 *  - can_altivec: return 1 if can run the cpuid instruction, 0 if it cannot,
 *                 -1 if the function is not implemented for the
 *                 platform/compiler combo
 */
#include "ppc_unknown.c"

int ppc_get_caps(ppc_cpu_caps *cpu)
{
    int st;

    cpu->has_altivec = 0;

    st = altivec_detect();
    if (st == -1) {
        cpu->has_altivec = -1;
    }

    return 0;
}
