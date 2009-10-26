/* Small test file to test for cpucap library */
#include <stdio.h>
#include <stdlib.h>

#include <numpy/npy_cpu.h>
#include "common.h"

int main()
{
    int flag, simd_flag;
    cpu_caps_t cpuinfo;

    get_cpu_caps(&cpuinfo);

    flag = cpuinfo.arch;
    switch(flag) {
        case NPY_CPUTYPE_X86:
            printf("X86 detected\n");
            break;
        case NPY_CPUTYPE_AMD64:
            printf("AMD64 detected\n");
            break;
        case NPY_CPUTYPE_PPC:
            printf("PPC 32 detected\n");
            break;
        case NPY_CPUTYPE_PPC64:
            printf("PPC 64 detected\n");
            break;
        case NPY_CPUTYPE_SPARC:
            printf("SPARC 32 detected\n");
            break;
        case NPY_CPUTYPE_SPARC64:
            printf("SPARC 64 detected\n");
            break;
        default:
            printf("Unknown CPU\n");
            exit(EXIT_FAILURE);
    }

    simd_flag = cpuinfo.simd;
    if (simd_flag & NPY_SIMD_UNIMPLEMENTED) {
        printf("SIMD detection not implemented\n");
    }
    if (simd_flag & NPY_SIMD_MMX) {
        printf("Has MMX\n");
    }
    if (simd_flag & NPY_SIMD_SSE) {
        printf("Has SSE\n");
    }
    if (simd_flag & NPY_SIMD_SSE2) {
        printf("Has SSE2\n");
    }
    if (simd_flag & NPY_SIMD_SSE3) {
        printf("Has SSE3\n");
    }
    if (simd_flag & NPY_SIMD_SSSE3) {
        printf("Has SSSE3\n");
    }
    if (simd_flag & NPY_SIMD_SSE4) {
        printf("Has SSE4\n");
    }
    if (simd_flag & NPY_SIMD_SSE42) {
        printf("Has SSE4.2\n");
    }

    if (simd_flag & NPY_SIMD_ALTIVEC) {
        printf("Has ALTIVEC\n");
    }

    return 0;
}
