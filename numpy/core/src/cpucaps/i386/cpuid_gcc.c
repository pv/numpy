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

static int read_cpuid(npy_reg32_t func, cpuid_t *cpuid)
{
    /* we save ebx because it is used when compiled by -fPIC */
    asm volatile(
            "pushl %%ebx      \n\t" /* save %ebx */
            "cpuid            \n\t"
            "movl %%ebx, %1   \n\t" /* save what cpuid just put in %ebx */
            "popl %%ebx       \n\t" /* restore the old %ebx */
            : "=a"(cpuid->eax), "=r"(cpuid->ebx), 
              "=c"(cpuid->ecx), "=d"(cpuid->edx)
            : "a"(func)
            : "cc"); 

    return 0;
}

#elif defined(NPY_CPU_AMD64)
static int can_cpuid(void)
{
    return 1;
}

static int read_cpuid(npy_reg32_t func, cpuid_t *cpuid)
{
    asm volatile (
            "  pushq %%rbx\n"
            "  cpuid\n"
            "  mov %%ebx, %%esi\n"
            "  popq %%rbx\n"
            : "=a" (cpuid->eax), "=r" (cpuid->ebx), 
              "=c" (cpuid->ecx), "=d" (cpuid->edx)
            : "a"(func)
            : "cc"); 

    return 0;
}
#else
#error CPUID for unknown ARCH ? This is a bug, please report it to numpy \
    developers
#endif
