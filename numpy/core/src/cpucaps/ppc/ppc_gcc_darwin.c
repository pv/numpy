#include <sys/sysctl.h>

/* See e.g. x264 or liboil cpu detection */

int altivec_detect( void )
{
    int res = 0;
    int selectors[2] = {CTL_HW, HW_VECTORUNIT};

    int has_altivec = 0;
    size_t length = sizeof(has_altivec);

    int error = sysctl(selectors, 2, &has_altivec, &length, NULL, 0);

    if( error == 0 && has_altivec != 0 ) {
        return 1;
    }

    return 0;
}
