#ifdef NPY_CPU_X86
static int can_cpuid(void)
{
    int has_cpuid = 0 ;

    /*
     * See intel doc on cpuid (pdf)
     */
    asm volatile (
            "pushfl			\n\t"
            "popl %%eax		\n\t"
            "movl %%eax, %%ecx	\n\t"
            "xorl $0x200000, %%eax	\n\t"
            "pushl %%eax		\n\t"
            "popfl			\n\t"
            "pushfl			\n\t"
            "popl %%eax		\n\t"
            "xorl %%ecx, %%eax	\n\t"
            "andl $0x200000, %%eax	\n\t"
            "movl %%eax,%0		\n\t"
            :"=m" (has_cpuid)
            : /*no input*/
            : "eax","ecx","cc");

    if (has_cpuid != 0) {
        return 1;
    }

    return 0;
}
#elif defined(NPY_CPU_AMD64)
static int can_cpuid(void)
{
    return 1;
}
#else
#error CPUID for unknown ARCH ? This is a bug, please report it to numpy \
    developers
#endif
