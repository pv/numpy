#include <numpy/npy_compiler.h>
#include <numpy/npy_cpu.h>

#include "ppc.h"

/*
 * We enclose compiler specific assembly code in those ppc_* files:
 *  - altivec_detect: return 1 if can run the cpuid instruction, 0 if it
 *                    cannot, -1 if the function is not implemented for the
 *                    platform/compiler combo
 */
#if defined(NPY_OS_DARWIN) && defined(NPY_COMPILER_GCC)
#include "ppc_gcc_darwin.c"
#else
#include "ppc_unknown.c"
#endif

int ppc_get_caps(ppc_cpu_caps *cpu)
{
    int st;

    cpu->has_altivec = 0;

    st = altivec_detect();
    if (st == -1) {
        /* Altivec detection not implemented case */
        cpu->has_altivec = -1;
    } else if (st == 1) {
        cpu->has_altivec = 1;
    }

    return 0;
}
