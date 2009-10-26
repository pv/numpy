#include <numpy/npy_compiler.h>
#include <numpy/npy_cpu.h>

#include "cpuid.h"

typedef npy_uint32 npy_reg32_t;

/*
 * struct reprensenting the cpuid flags as put in the register
 */
typedef struct {
    npy_reg32_t eax;
    npy_reg32_t ebx;
    npy_reg32_t ecx;
    npy_reg32_t edx;
} cpuid_t;

/*
 * We enclose compiler specific assembly code in those cpuid_* files:
 *  - can_cpuid: return 1 if can run the cpuid instruction, 0 if it cannot, -1
 *               if the function is not implemented for the platform/compiler
 *               combo
 */
#ifdef NPY_COMPILER_GCC
#include "cpuid_gcc.c"
#else
#include "cpuid_unknown.c"
#endif

int cpuid_get_caps(i386_cpu_caps *cpu)
{
    cpuid_t cpuid;
    int max;

    cpu->can_cpuid = can_cpuid();

    if (cpu->can_cpuid != 1) {
        return 0;
    }

    return 0;
}
