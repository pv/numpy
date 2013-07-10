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
operation to a function contianed the attribute. 

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

Using subclasses of ndarray is limitted to ``__array_prepare__`` and
``__array_wrap__`` but these don't even allow you to change the shape or
the data of the arguments. Ufuncing things that don't subclass ndarray
is even more hopeless. Take this example of ufuncs interopability with
sparse matrices.::

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
    Out[3]:array([[0, 4, 0],
                  [0, 0, 2],
                  [4, 0, 1]])


To start with, it is virtually impossible to come up with a single
date/time type that fills the needs of every case of use.  So, after
pondering about different possibilities, we have stuck with *two*
different types, namely ``datetime64`` and ``timedelta64`` (these names
are preliminary and can be changed), that can have different time units
so as to cover different needs.

.. [5] http://mail.scipy.org/pipermail/numpy-discussion/2011-June/056945.html

Implementation
==============

Compatability
-------------

Forward
~~~~~~~

Backward
~~~~~~~~

.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:

