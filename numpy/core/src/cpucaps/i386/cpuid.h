#ifndef _NPY_CPUCAPS_CPUID_H_
#define _NPY_CPUCAPS_CPUID_H_

#define CPUID_VENDOR_STRING_LEN  12

typedef struct {
	int can_cpuid;
	int has_mmx;
	int has_sse;
	int has_sse2;
	int has_sse3;
	char vendor[CPUID_VENDOR_STRING_LEN+1];
} i386_cpu_caps;

int cpuid_get_caps(i386_cpu_caps *cpu);

#endif
