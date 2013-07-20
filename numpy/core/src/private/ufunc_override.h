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

    PyObject *override_obj = NULL;
    PyObject *override_meth = NULL;
    PyObject *override_res = NULL;
    PyObject *res = NULL;

    for (i = 0; i < nargs; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_CheckExact(obj) || PyArray_IsAnyScalar(obj)) {
            continue;
        }

        if (PyObject_HasAttrString(obj, "__numpy_ufunc__")) {
            override_meth = PyObject_GetAttrString(obj, "__numpy_ufunc__");
            override_args = Py_BuildValue("OOiO", ufunc, ufunc_method, i, args);
            res = PyObject_Call(override_meth, override_args, kwds);
            Py_DECREF(override_args);

            if (res != Py_NotImplemented) {
                if (!override_obj) {
                    override_obj = obj;
                    override_res = res;
                    Py_DECREF(obj);
                }

                else if (PyObject_IsSubclass(obj->ob_type, 
                                             override_obj->ob_type)) {
                    override_obj = obj;
                    override_res = res;
                    Py_DECREF(obj);
                }
            }
        }
    }
    /* 
     * Once override_meth is set it never is NULL again, so we use it to check
     * if any of the args had __numpy_ufunc__.
     */
    if (override_meth && !override_res) {
        PyErr_SetString(PyExc_TypeError, 
                "Not implemented for this type.");
        Py_DECREF(override_meth);
        return NULL;
    }
    if (override_res) {
        Py_DECREF(override_meth);
        return override_res;
    }
    else {
        return NULL;
    }
}

#endif
