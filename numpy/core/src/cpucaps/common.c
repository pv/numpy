#include <numpy/npy_cpu.h>

#include "common.h"

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

    return 0;
}
