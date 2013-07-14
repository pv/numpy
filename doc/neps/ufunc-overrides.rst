=================================
A Mechanism for Overriding Ufuncs
=================================

:Author: Blake Griffith
:Contact: blake.g@utexa.edu 
:Date: 2013-07-10


Executive summary
=================

NumPy's universal functions (ufuncs) currently have some limited
functionality for operating on user defined subclasses of ndarray using
``__array_prepare__`` and ``__array_wrap__`` [1]_, and there is little
to no support for arbitrary objects. e.g. SciPy's sparse matrices [2]_
[3]_.

Here we propose adding a mechanism to override ufuncs based on the ufunc
checking each of it's arguments for a ``__ufunc_override__`` attribute.
On discovery of ``__ufunc_override__`` the ufunc will hand off the
operation to a function contained the attribute. 

This covers some of the same ground as Travis Oliphant's proposal to
retro-fit NumPy with multi-methods [4]_, which would solve the same
problems. But the mechanism proposed here is much less intrusive.

.. [1] http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
.. [2] https://github.com/scipy/scipy/issues/2123
.. [3] https://github.com/scipy/scipy/issues/1569
.. [4] http://technicaldiscovery.blogspot.com/2013/07/thoughts-after-scipy-2013-and-specific.html


Motivation
==========

The current machinery for dispatching ufuncs is generally agreed to be
at a dead end. There have been lengthy discussions and other proposed
solutions [5]_. 

Using ufuncs with subclasses of ndarray is limited to
``__array_prepare__`` and ``__array_wrap__`` but these don't even allow
you to change the shape or the data of the arguments. Ufuncing things
that don't subclass ndarray is even more hopeless. Take this example of
ufuncs interoperability with sparse matrices.::

    In [1]: import numpy as np
    import scipy.sparse as sp

    a = np.random.randint(5, size=(3,3))
    b = np.random.randint(5, size=(3,3))

    asp = sp.csr_matrix(a)
    bsp = sp.csr_matrix(b)

    In [2]: a, b
    Out[2]:(array([[0, 4, 4],
                   [1, 3, 2],
                   [1, 3, 1]]),
            array([[0, 1, 0],
                   [0, 0, 1],
                   [4, 0, 1]]))

    In [3]: np.multiply(a, b) # The right answer
    Out[3]: array([[0, 4, 0],
                   [0, 0, 2],
                   [4, 0, 1]])

    In [4]: np.multiply(asp, bsp).todense() # calls __mul__ which does matrix multi
    Out[4]: matrix([[16,  0,  8],
                    [ 8,  1,  5],
                    [ 4,  1,  4]], dtype=int64)
                    
    In [5]: np.multiply(a, bsp) # Returns NotImplemented to user, bad!
    Out[5]: NotImplemed

Returning ``NotImplemented`` to user should not happen. I'm not sure if
the blame for this lies in scipy.sparse or numpy, but it should be
fixed.::

    In [6]: np.multiply(asp, b)
    Out[6]: array([[ <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>],
                       [ <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>],
                       [ <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>,
                        <3x3 sparse matrix of type '<class 'numpy.int64'>'
                    with 8 stored elements in Compressed Sparse Row format>]], dtype=object)

I'm not sure what happened here either, but I think raising
``TypeError`` would be preferable. Adding the ``__ufunc_override__``
functionality fixes this.

.. [5] http://mail.scipy.org/pipermail/numpy-discussion/2011-June/056945.html

Implementation
==============

Classes that should override ufuncs should contain a
``__array_priority__`` and ``__ufunc_override__`` attribute.
``__ufunc_override__`` is a dictionary keyed with the name
(``ufunc.__name__``) of the ufunc to be overridden, and valued with the
callable function that should override the ufunc. 

Every time a ufunc is executed it checks it arguments for a
``__ufunc_override__`` attribute. Then checks if the attribute contains
an override for the ufunc being called. A list of the eligible
overrides are made, then their corresponding ``__array_priority__`` is
compared to find the override with the highest priority. Once an
override is found, it is called with the ``args`` and ``kwds`` that were
given to the original ufunc.

Handing ``args`` and ``kwds``  as-is to the replacement function has one
drawback; if the replacement *function* is a *method* of the overriding argument, then
it expects this argument (``self``) to come first. In general this is
not the case unless the arguments are reordered for the overriding
argument to come first. This is probably a bad idea since ufuncs are not
necessarily associative. So the replacement funtions should be able to
handle the arguments in the same order as passed to the ufunc.

Demo
====

A pull request[6]_ has been made including the changes proposed in this NEP.
Here is a demo highlighting the effectiveness. Using the same variables
as above, except sparse matrices have a ufunc override attribute for
multiply.::

    In [1]: asp.__ufunc_override__
    Out[1]: {'multiply': <function scipy.sparse.base.multiply>}


    In [2]: np.multiply(asp, b)
    Out[2]: matrix([[0, 4, 0],
                    [0, 0, 2],
                    [4, 0, 1]])

We can define a simple class that will override the ufuncs like this.::

    In [3]: class TestClass(object):
                def foo(*args, **kwds):
                    return 42  # The answer.
                __array_priority__ = 13  # Just > matrix priority.
                __ufunc_override__ = {'add':foo}  # override add w/ foo

    In [4]: bar = TestClass()
    In [5]: np.add(bar, a)
    Out[5]: 42

.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:

