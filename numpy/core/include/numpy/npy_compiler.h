#ifndef _NPY_COMPILER_H_
#define _NPY_COMPILER_H_

#if defined(__GCCXML__)
    #define NPY_COMPILER_GCCXML
#elif defined(__GNUC__)
    #define NPY_COMPILER_GCC
#elif defined(_MSC_VER)
    #define NPY_COMPILER_VISUALC
#else
    #define NPY_COMPILER_UNKNOWN
#endif

#endif
