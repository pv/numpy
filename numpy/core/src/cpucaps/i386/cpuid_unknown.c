#ifdef NPY_CPU_X86
static int can_cpuid(void)
{
    return -1;
}
static int read_cpuid(npy_reg32_t func, cpuid_t *cpuid)
{
    return -1;
}
#elif NPY_CPU_AMD64
static int can_cpuid(void)
{
    return 1;
}
static int read_cpuid(npy_reg32_t func, cpuid_t *cpuid)
{
    return -1;
}
#else
#error CPUID for unknown ARCH ? This is a bug, please report it to numpy \
    developers
#endif
