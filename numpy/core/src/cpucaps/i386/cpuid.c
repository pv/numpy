#include <numpy/npy_compiler.h>
#include <numpy/npy_cpu.h>

#include "cpuid.h"

/*
 * SIMD: SSE 1, 2 and 3, MMX
 */
#define CPUID_FLAG_MMX  1 << 23 /* in edx */
#define CPUID_FLAG_SSE  1 << 25 /* in edx */
#define CPUID_FLAG_SSE2 1 << 26 /* in edx */
#define CPUID_FLAG_SSE3 1 << 0  /* in ecx */
#define CPUID_FLAG_SSSE3 0x0200  /* in ecx */
#define CPUID_FLAG_SSE4  0x080000  /* in ecx */
#define CPUID_FLAG_SSE42 0x100000  /* in ecx */

/*
 * long mode (AMD64 instruction set)
 */
#define CPUID_FLAGS_LONG_MODE   1 << 29 /* in edx */

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
 *  - read_cpuid: return 0 if can read the cpuid instruction, -1 if the
 *                function is not implemented for the platform/compiler combo.
 */
#ifdef NPY_COMPILER_GCC
#include "cpuid_gcc.c"
#else
#include "cpuid_unknown.c"
#endif

int cpuid_get_caps(i386_cpu_caps *cpu)
{
    cpuid_t cpuid;
    int st;

    cpu->can_cpuid = can_cpuid();

    cpu->has_mmx = 0;
    cpu->has_sse = 0;
    cpu->has_sse2 = 0;
    cpu->has_sse3 = 0;
    cpu->has_ssse3 = 0;
    cpu->has_sse4 = 0;
    cpu->has_sse42 = 0;

    if (cpu->can_cpuid != 1) {
        return 0;
    }

    /* determine which CPUID level we can use */
    st = read_cpuid(0, &cpuid);
    if (st == -1) {
        return 0;
    }
    if (cpuid.eax < 0x00000001) {
        return 0;
    }

    /* Get MMX, SSE 1, 2, and 3 capabilities */
    st = read_cpuid(0x00000001, &cpuid);
    if (st == -1) {
        return 0;
    }

    if (cpuid.edx & CPUID_FLAG_MMX) {
        cpu->has_mmx = 1;
    }
    if (cpuid.edx & CPUID_FLAG_SSE) {
        cpu->has_sse = 1;
    }
    if (cpuid.edx & CPUID_FLAG_SSE2) {
        cpu->has_sse2 = 1;
    }
    if (cpuid.ecx & CPUID_FLAG_SSE3) {
        cpu->has_sse3 = 1;
    }
    if (cpuid.ecx & CPUID_FLAG_SSSE3) {
        cpu->has_ssse3 = 1;
    }
    if (cpuid.ecx & CPUID_FLAG_SSE4) {
        cpu->has_sse4 = 1;
    }
    if (cpuid.ecx & CPUID_FLAG_SSE42) {
        cpu->has_sse42 = 1;
    }

    return 0;
}
