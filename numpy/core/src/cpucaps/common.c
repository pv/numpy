#include <numpy/npy_cpu.h>

#include "common.h"
#if defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64)
    #include "i386/cpuid.h"
#elif defined(NPY_CPU_PPC) || defined(NPY_CPU_PPC64)
    #include "ppc/ppc.h"
#endif

int get_cpu_caps(cpu_caps_t *const cpu)
{
#if defined(NPY_CPU_X86)
    cpu->arch = NPY_CPUTYPE_X86;
#elif defined(NPY_CPU_AMD64)
    cpu->arch = NPY_CPUTYPE_AMD64;
#elif defined(NPY_CPU_SPARC)
    cpu->arch = NPY_CPUTYPE_SPARC;
#elif defined(NPY_CPU_SPARC64)
    cpu->arch = NPY_CPUTYPE_SPARC64;
#elif defined(NPY_CPU_PPC)
    cpu->arch = NPY_CPUTYPE_PPC;
#elif defined(NPY_CPU_PPC64)
    cpu->arch = NPY_CPUTYPE_PPC64;
#elif defined(NPY_CPU_S390)
    cpu->arch = NPY_CPUTYPE_S390;
#elif defined(NPY_CPU_IA64)
    cpu->arch = NPY_CPUTYPE_IA64;
#elif defined(NPY_CPU_HPPA)
    cpu->arch = NPY_CPUTYPE_HPPA;
#elif defined(NPY_CPU_ALPHA)
    cpu->arch = NPY_CPUTYPE_ALPHA;
#elif (defined(NPY_CPU_ARMEL) || defined(NPY_CPU_ARMEB))
    cpu->arch = NPY_CPUTYPE_ARM;
#elif (defined(NPY_CPU_MIPSEL) || defined(NPY_CPU_MIPSEB))
    cpu->arch = NPY_CPUTYPE_MIPS;
#elif (defined(NPY_CPU_SH_EL) || defined(NPY_CPU_SH_EB))
    cpu->arch = NPY_CPUTYPE_SH;
#else
    cpu->arch = NPY_CPUTYPE_UNKNOWN;
#endif

#if defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64)
    {
        i386_cpu_caps intern;

        cpuid_get_caps(&intern);

        if (intern.can_cpuid == -1) {
            cpu->simd = NPY_SIMD_UNIMPLEMENTED;
        } else {
            cpu->simd = 0;

            if(intern.has_mmx) {
                cpu->simd |= NPY_SIMD_MMX;
            }
            if(intern.has_sse) {
                cpu->simd |= NPY_SIMD_SSE;
            }
            if(intern.has_sse2) {
                cpu->simd |= NPY_SIMD_SSE2;
            }
            if(intern.has_sse3) {
                cpu->simd |= NPY_SIMD_SSE3;
            }
        }
    }
#elif defined(NPY_CPU_PPC) || defined(NPY_CPU_PPC64)
    {
        ppc_cpu_caps intern;

        ppc_get_caps(&intern);

        cpu->simd = 0;

        if (intern.has_altivec == -1) {
            cpu->simd |= NPY_SIMD_UNIMPLEMENTED;
        } else if (intern.has_altivec == 1) {
            cpu->simd |= NPY_SIMD_ALTIVEC;
        }
    }
#else
    cpu->simd = NPY_SIMD_UNKNOWN;
#endif
    return 0;
}
