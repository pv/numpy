#ifndef _NPY_UMATH_UFUNC_OBJECT_H_
#define _NPY_UMATH_UFUNC_OBJECT_H_

NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT PyObject *
ufunc_get_lazy_hook();

#define UFUNC_LAZY_BYPASS(self, calltype, args, kwds)                   \
    do {                                                                \
        PyObject *lazy_hook, *ret, *kw;                                 \
        lazy_hook = ufunc_get_lazy_hook();                              \
        if (lazy_hook) {                                                \
            kw = kwds;                                                  \
            if (!kw) {                                                  \
                kw = PyDict_New();                                      \
            }                                                           \
            ret = PyObject_CallFunction(lazy_hook, "OsOO",              \
                                        self, calltype, args, kw);      \
            if (ret == Py_True) {                                       \
                Py_DECREF(ret);                                         \
            }                                                           \
            else {                                                      \
                return ret; /* bypass */                                \
            }                                                           \
        }                                                               \
    } while (0)

#endif
