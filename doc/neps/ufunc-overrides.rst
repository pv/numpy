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
checking each of it's arguments for a ``__numpy_ufunc__`` method.
On discovery of ``__numpy_ufunc__`` the ufunc will hand off the
operation to the method. 

This covers some of the same ground as Travis Oliphant's proposal to
retro-fit NumPy with multi-methods [4]_, which would solve the same
problem. The mechanism here follows more closely the way Python enables
classes to override ``__mul__`` and other binary operations.

.. [1] http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
.. [2] https://github.com/scipy/scipy/issues/2123
.. [3] https://github.com/scipy/scipy/issues/1569
.. [4] http://technicaldiscovery.blogspot.com/2013/07/thoughts-after-scipy-2013-and-specific.html


Motivation
==========

The current machinery for dispatching Ufuncs is generally agreed to be
insufficient. There have been lengthy discussions and other proposed
solutions [5]_.

Using ufuncs with subclasses of ndarray is limited to
``__array_prepare__`` and ``__array_wrap__`` to prepare the arguments,
but these don't allow you to for example change the shape or the data of
the arguments. Ufuncing things that don't subclass ndarray is even more
hopeless, as the input arguments tend to be cast to object arrays, which
ends up producing surprising results.

Take this example of ufuncs interoperability with sparse matrices.::

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
    Out[5]: NotImplemted

Returning ``NotImplemented`` to user should not happen. Moreover::

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

Here, it appears that the sparse matrix was converted to a object array
scalar, which was then multiplied with all elements of the ``b`` array.
However, this behavior is more confusing than useful, and having a
``TypeError`` would be preferable.

Adding the ``__numpy_ufunc__`` functionality fixes this and would
deprecate the other ufunc modifying functions.

.. [5] http://mail.scipy.org/pipermail/numpy-discussion/2011-June/056945.html


Proposed interface
==================

Objects that want to override Ufuncs can define a ``__numpy_ufunc__`` method.
The method signature is::

    def __numpy_ufunc__(self, ufunc, method, i, inputs, kwargs)

Here:

- *ufunc* is the ufunc object that was called. 
- *method* is a string indicating which Ufunc method was called
  (one of ``"__call__"``, ``"reduce"``, ``"reduceat"``,
  ``"accumulate"``, ``"outer"``, ``"inner"``). 
- *i* is the index of *self* in *inputs*.
- *inputs* is a tuple of the input arguments to the ``ufunc``
- *kwargs* is a dictionary containing the optional input arguments
  of the ufunc. The ``out`` argument is always contained in
  *kwargs*, if given.

The ufunc's arguments are first normalized into a tuple of input data
(``inputs``), and dict of keyword arguments. The output argument ``out``
is always put into the keyword argument dictionary.

The function dispatch proceeds as follows:

- If one of the input arguments implements ``__numpy_ufunc__`` it is
  executed instead of the Ufunc.

- If more than one of the input arguments implements ``__numpy_ufunc__``,
  they are tried in the following order: subclasses before superclasses,
  otherwise left to right.  The first ``__numpy_ufunc__`` method returning
  something else than ``NotImplemented`` determines the return value of
  the Ufunc.

- If all ``__numpy_ufunc__`` methods of the input arguments return
  ``NotImplemented``, a ``TypeError`` is raised.

- If a ``__numpy_ufunc__`` method raises an error, the error is propagated
  immediately.

If none of the input arguments has a ``__numpy_ufunc__`` method, the
execution falls back on the default ufunc behaviour.


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

