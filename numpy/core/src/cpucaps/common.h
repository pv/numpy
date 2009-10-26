#ifndef _NPY_CPUCAPS_COMMON_H
#define _NPY_CPUCAPS_COMMON_H

typedef	struct cpu_caps_t_tag	cpu_caps_t;

struct cpu_caps_t_tag {
    int arch;
    int simd;
};

int get_cpu_caps(cpu_caps_t *const cpu);

#endif
