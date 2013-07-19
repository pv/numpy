#ifndef __UFUNC_OVERRIDE_H
#define __UFUNC_OVERRIDE_H
#include <npy_config.h>
#include "numpy/arrayobject.h"
#include "multiarray/common.h"


static PyObject *
PyUFunc_CheckOverride(PyUFuncObject *ufunc, PyObject *ufunc_method,
                      PyObject *args, PyObject *kwds)
{
    int i;
    int nargs = PyTuple_GET_SIZE(args);

    PyObject *obj;
    PyObject *override_args;

    PyObject *override = NULL;
    PyObject *res = NULL;

    for (i = 0; i < nargs; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_CheckExact(obj) || PyArray_IsAnyScalar(obj)) {
            continue;
        }

        if (PyObject_HasAttrString(obj, "__numpy_ufunc__")) {
            override = PyObject_GetAttrString(obj, "__numpy_ufunc__");
            override_args = Py_BuildValue("OOiO", ufunc, ufunc_method, i, args);
            res = PyObject_Call(override, override_args, kwds);

            Py_DECREF(override);
            Py_DECREF(override_args);

            if (res == Py_NotImplemented) {
                continue;
            }
            else {
                return res;
            }
        }
    }
    if (res) {
        PyErr_SetString(PyExc_TypeError, 
                "Not implemented for this type.");
        return NULL;
    }
    else {
        return NULL;
    }
}

#endif
