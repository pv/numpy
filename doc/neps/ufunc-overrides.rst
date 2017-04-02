=================================
A Mechanism for Overriding Ufuncs
=================================

.. currentmodule:: numpy

:Author: Blake Griffith
:Contact: blake.g@utexas.edu 
:Date: 2013-07-10

:Author: Pauli Virtanen

:Author: Nathaniel Smith

:Author: Marten van Kerkwijk
:Date: 2017-03-31

Executive summary
=================

NumPy's universal functions (ufuncs) currently have some limited
functionality for operating on user defined subclasses of
:class:`ndarray` using ``__array_prepare__`` and ``__array_wrap__``
[1]_, and there is little to no support for arbitrary
objects. e.g. SciPy's sparse matrices [2]_ [3]_.

Here we propose adding a mechanism to override ufuncs based on the ufunc
checking each of it's arguments for a ``__array_ufunc__`` method.
On discovery of ``__array_ufunc__`` the ufunc will hand off the
operation to the method. 

This covers some of the same ground as Travis Oliphant's proposal to
retro-fit NumPy with multi-methods [4]_, which would solve the same
problem. The mechanism here follows more closely the way Python enables
classes to override ``__mul__`` and other binary operations. It also
specifically addresses how binary operators and ufuncs should interact.

.. note:: In earlier iterations, the override was called
          ``__numpy_ufunc__``. An implementation was made, but had not
          quite the right behaviour, hence the change in name.

.. [1] http://docs.python.org/doc/numpy/user/basics.subclassing.html
.. [2] https://github.com/scipy/scipy/issues/2123
.. [3] https://github.com/scipy/scipy/issues/1569
.. [4] http://technicaldiscovery.blogspot.com/2013/07/thoughts-after-scipy-2013-and-specific.html


Motivation
==========

The current machinery for dispatching Ufuncs is generally agreed to be
insufficient. There have been lengthy discussions and other proposed
solutions [5]_, [6]_.

Using ufuncs with subclasses of :class:`ndarray` is limited to
``__array_prepare__`` and ``__array_wrap__`` to prepare the arguments,
but these don't allow you to for example change the shape or the data of
the arguments. Trying to ufunc things that don't subclass
:class:`ndarray` is even more difficult, as the input arguments tend to
be cast to object arrays, which ends up producing surprising results.

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

Returning :obj:`NotImplemented` to user should not happen. Moreover::

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

Here, it appears that the sparse matrix was converted to an object array
scalar, which was then multiplied with all elements of the ``b`` array.
However, this behavior is more confusing than useful, and having a
:exc:`TypeError` would be preferable.

Adding the ``__array_ufunc__`` functionality fixes this and would
deprecate the other ufunc modifying functions.

.. [5] http://mail.python.org/pipermail/numpy-discussion/2011-June/056945.html

.. [6] https://github.com/numpy/numpy/issues/5844

Proposed interface
==================

The standard array class :class:`ndarray` gains an ``__array_ufunc__``
method and objects can override Ufuncs by overriding this method (if
they are :class:`ndarray` subclasses) or defining their own. The method
signature is::

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs)

Here:

- *ufunc* is the ufunc object that was called. 
- *method* is a string indicating which Ufunc method was called
  (one of ``"__call__"``, ``"reduce"``, ``"reduceat"``,
  ``"accumulate"``, ``"outer"``, ``"inner"``). 
- *inputs* is a tuple of the input arguments to the ``ufunc``
- *kwargs* are the keyword arguments passed to the function. The ``out``
  arguments are always contained as a tuple in *kwargs*.

Hence, the arguments are normalized: only the input data (``inputs``)
are passed on as positional arguments, all the others are passed on as a
dict of keyword arguments (``kwargs``). In particular, if there are
output arguments, positional are otherwise, they are passed on as a
tuple in the ``out`` keyword argument.

The function dispatch proceeds as follows:

- If an input argument has a ``__array_ufunc__`` attribute, but its
  value is ``ndarray.__array_ufunc__``, the attribute is considered to
  be absent in what follows.  This happens for instances of `ndarray`
  and those `ndarray` subclasses that did not override their inherited
  ``__array_ufunc__`` implementation.

- If one of the input arguments implements ``__array_ufunc__``, it is
  executed instead of the ufunc.

- If more than one of the input arguments implements ``__array_ufunc__``,
  they are tried in the following order: subclasses before superclasses,
  otherwise left to right.

- The first ``__array_ufunc__`` method returning something else than
  :obj:`NotImplemented` determines the return value of the Ufunc.

- If all ``__array_ufunc__`` methods of the input arguments return
  :obj:`NotImplemented`, a :exc:`TypeError` is raised.

- If a ``__array_ufunc__`` method raises an error, the error is
  propagated immediately.

- If none of the input arguments had an ``__array_ufunc__`` method, the
  execution falls back on the default ufunc behaviour.


Type casting hierarchy
----------------------

Similarly to the Python operator dispatch mechanism, writing ufunc
dispatch methods requires some discipline in order to achieve
predictable results.

In particular, it is useful to maintain a clear idea of what types can
be upcast to others, possibly indirectly (i.e. A->B->C is implemented
but direct A->C not). Moreover, one should make sure the implementations of
``__array_ufunc__``, which implicitly define the type casting hierarchy,
don't contradict this.

The following rules should be followed:

1. The ``__array_ufunc__`` for type A should either return
   `NotImplemented`, or return an output of type A (unless an
   ``out=`` argument was given, in which case ``out`` is returned).

2. For any two different types *A*, *B*, the relation "A can handle B" 
   defined as::

       a.__array_ufunc__(..., b, ...) is not NotImplemented

   for instances *a* and *b* of *A* and *B*, defines the
   edges B->A of a graph.

   This graph must be a directed acyclic graph.

Under these conditions, the transitive closure of the "can handle"
relation defines a strict partial ordering of the types -- that is, the
type casting hierarchy.

In other words, for any given class A, all other classes that define
``__array_ufunc__`` must belong to exactly one of the groups:

- *Above A*: their ``__array_ufunc__`` can handle class A or some
  member of the "above A" classes. In other words, these are the types
  that A can be (indirectly) upcast to in ufuncs.

- *Below A*: they can be handled by the ``__array_ufunc__`` of class A
  or the ``__array_ufunc__`` of some member of the "below A" classes. In
  other words, these are the types that can be (indirectly) upcast to A
  in ufuncs.

- *Incompatible*: neither above nor below A; types for which no
  (indirect) upcasting is possible.

This guarantees that expressions involving ufuncs either raise a
`TypeError`, or the result type is independent of what ufuncs were
called, what order they were called in, and what order their arguments
were in.  Moreover, which ``__array_ufunc__`` payload code runs at each
step is independent of the order of arguments of the ufuncs.

Note also that while converting inputs that don't have
``__array_ufunc__`` to `ndarray` via `np.asarray` is consistent with the
type casting hierarchy, also returning `NotImplemented` is
consistent. However, the numpy ufunc (legacy) behavior is to try to
convert unknown objects to ndarrays.


.. admonition:: Example

   Type casting hierarchy.

   .. graphviz::

      digraph array_ufuncs {
         rankdir=BT;
         A -> C;
         B -> C;
         D -> B;
         ndarray -> A;
         ndarray -> B;
      }

   The ``__array_ufunc__`` of type A can handle ndarrays, B can handle ndarray and D,
   and C can handle A and B but not ndarrays or D.  The resulting graph is a DAG,
   and defines a type casting hierarchy, with relations ``C > A >
   ndarray``, ``C > B > ndarray``, ``C > B > D``. The type B is incompatible
   relative to A and vice versa, and A and ndarray are incompatible relative to D.
   Ufunc expressions involving these classes produce results of the highest type
   involved or raise a TypeError.


.. admonition:: Example

   1-cycle in the ``__array_ufunc__`` graph.

   .. graphviz::

      digraph array_ufuncs {
         rankdir=BT;
         A -> B;
         B -> A;
      }


   In this case, the ``__array_ufunc__`` relations have a cycle of length 1,
   and a type casting hierarchy does not exist. Binary operations are not
   commutative: ``type(a + b) is A`` but ``type(b + a) is B``.

.. admonition:: Example

   Longer cycle in the ``__array_ufunc__`` graph.

   .. graphviz::

      digraph array_ufuncs {
         rankdir=BT;
         A -> B;
         B -> C;
         C -> A;
      }


   In this case, the ``__array_ufunc__`` relations have a longer cycle, and a type
   casting hierarchy does not exist. Binary operations are still commutative,
   but type transitivity is lost: ``type(a + (b + c)) is A`` but
   ``type((a + b) + c) is C``.


Subclass hierarchies
--------------------

Generally, it is desirable to mirror the class hierarchy in the ufunc
type casting hierarchy. The recommendation is that an
``__array_ufunc__`` implementation of a class should generally return
`NotImplemented` unless the inputs are instances of the same class or
superclasses.  This guarantees that in the type casting hierarchy,
superclasses are below, subclasses above, and other classes are
incompatible.  Exceptions to this need to check they respect the
implicit type casting hierarchy.

.. note::

   It would in principle be consistent to have ``__array_ufunc__``
   handle instances of subclasses. This would correspond to "upcasting"
   ndarray subclasses to plain ndarrays. However, this does not
   seem like a useful model: the subclass *is-a* relationship
   ``Animal > Dog > Labrador`` is different in nature to the
   type casting hierarchy ``complex > real > integer``.

Subclasses can be easily constructed if methods consistently use
:func:`super` to pass through the class hierarchy [7]_.  To support
this, :class:`ndarray` has its own ``__array_ufunc__`` method,
equivalent to::

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        #
        # Handle items of type(self), superclasses, and items
        # without __array_ufunc__.
        #
        # As a legacy measure, also handle classes that don't
        # have an overridden their __array_ufunc__ and have
        # lower __array_priority__.
        #
        for item in inputs:
            if not hasattr(item, '__array_ufunc__'):
                # Handle items without __array_ufunc__
                pass
            elif isinstance(self, type(item)):
                # Handle superclasses
                pass
            elif (getattr(item, '__array_ufunc__', ndarray.__array_ufunc__) is ndarray.__array_ufunc__
                      and self.__array_priority__ >= getattr(other, '__array_priority__', -inf)):
                # Legacy handling of __array_priority__
                pass
            else:
                return NotImplemented

        # Perform ufunc on the underlying ndarrays (no __array_ufunc__ dispatch)
        items = [np.asarray(item) if isinstance(item, np.ndarray) else item
                 for item in inputs]
        result = getattr(ufunc, method)(*items, **kwargs)

        # Cast output to type(self), unless `out` specified
        if kwargs['out']:
            return result

        if isinstance(result, tuple):
            return tuple(x.view(type(self)) for x in result)
        else:
            return result.view(type(self))

Note that, as a special case, the ufunc dispatch mechanism does not call
the `ndarray.__array_ufunc__` method, even for `ndarray` subclasses
if they have not overridden the default `ndarray` implementation. As a
consequence, calling `ndarray.__array_ufunc__` will not result to a
nested ufunc dispatch cycle.  Custom implementations of
`__array_ufunc__` should generally avoid nested dispatch cycles.

This should be particularly useful for subclasses of :class:`ndarray`,
which only add an attribute like a unit or mask to a regular
:class:`ndarray`. In their `__array_ufunc__` implementation, such
classes can do possible adjustment of the arguments relevant to their
own class, and pass on to superclass implementation using :func:`super`
until the ufunc is actually done, and then do possible adjustments of
the outputs.

Turning Ufuncs off
------------------

For some classes, Ufuncs make no sense, and, like for other special
methods [8]_, one can indicate Ufuncs are not available by setting
``__array_ufunc__`` to :obj:`None`.  Inside a Ufunc, this is
equivalent to unconditionally returning :obj:`NotImplemented`, and thus
will lead to a :exc:`TypeError` (unless another operand implements
``__array_ufunc__`` and specifically knows how to deal with the class).

In the type casting hierarchy, this makes the type incompatible relative
to `ndarray`.

.. [7] https://rhettinger.wordpress.com/2011/05/26/super-considered-super/

.. [8] https://docs.python.org/3/reference/datamodel.html#specialnames


Python binary operations
========================

XXX: work in progress --- didn't think this through completely. Also,
the PEP must write down the other alternatives discussed earlier --- this
way it will become clearer what tradeoffs they imply.


The ``__array_ufunc__`` mechanism is fully independent of Python's
standard operator override mechanism, and the two do not interact
directly.

Because NumPy's :class:`ndarray` type implements its binary operations
via Ufuncs, the binop dispatch however needs to be carefully considered.

Traditionally :class:`ndarray` tried to cast all input arguments to
ndarrays, and, if this fails raise a TypeError, a behavior that could
be overridden by specifying ``__array_priority__``.

The ``__array_ufunc__`` attribute in binary operations is considered to
be equivalent to the previous ``__array_priority__ = inf``. It is thus
defined to play a dual role --- it not only allows overriding the ufunc
dispatch, but also its presence *turns off* the greedy default behavior
of ndarray.

The :class:`ndarray` will implement its binary operations equivalently
to the following logic::

    class ndarray:
        __array_priority__ = 0.0

        def __mul__(self, other):
            return self.__array_ufunc__(self, np.multiply, '__call__', self, other)

        def __rmul__(self, other):
            return self.__array_ufunc__(self, np.multiply, '__call__', other, self)

        def __imul__(self, other):
            result = self.__array_ufunc__(self, np.multiply, '__call__', other, self,
                                          out=(self,))
            if result is NotImplemented:
                raise TypeError()

Here, the implementation of ``__array_ufunc__`` is as described above.

There are the following salient points:

- The binops handle the same types as ``ndarray.__array_ufunc__``
  in exactly the same way.

- The binop is implemented via a ufunc call, but does **not**
  result to a ``__array_ufunc__`` dispatch resolution cycle.

- The presence of an ``__array_ufunc__`` attribute *is* noted by
  `ndarray` binops, and is considered to imply that the *other*
  operand cannot be handle by the binop of `ndarray`.

Moreover, ndarray disallows Python to convert ``x += y -> x = x + y``.
The ``ndarray.__imul__`` method either performs the operation, or
if it cannot, :exc:`TypeError` is raised.


Type casting hierarchy for binops
---------------------------------

The Python binary operations can be discussed in a similar type casting
framework as ``__array_ufunc__`` above.

The "can handle" relation needs to be replaced with::

    a.__mul__(b) is not NotImplemented

and we need an additional constraint:

3. ``a.__rmul__(b) is not NotImplemented`` if and only if
   ``a.__mul__(b) is not NotImplemented``.

In this case, the relation defines the B->A edge of a graph, and the
discussion goes through as before.

In the above formulation, it's then clear that if the
``__array_ufunc__`` satisfies the requirements to describe a type
casting hierarchy, and the binary ops are defined as above, the Python
binary ops also correspond to a type casting hierarchy.




Implementing array-like classes
===============================

For most numerical classes, the easiest way to override binary
operations is to follow the above pattern: define ``__array_ufunc__`` to
override the corresponding ufunc, and then define Python binary
operations using the ``__array_ufunc__`` so defined.

It is not recommended to try to replicate ndarray's handling of
``__array_priority__`` --- in ndarray subclasses this is however
obtained if you use ``super().__array_ufunc__``.

The simplest implementation would be::

    class ArrayLike(object):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            # Check inputs can be dealt with
            for item in inputs:
                if not hasattr(item, '__array_ufunc__'):
                    # Handle items without __array_ufunc__ (similarly to ndarray)
                    pass
                elif isinstance(item, ArrayLike):
                    # Handle own type
                    pass
                elif type(item) is np.ndarray:
                    # Handle ndarrays (but not its subclasses, to be safe)
                    pass
                else:
                    return NotImplemented

            # Deal only with multiplication, for this example...
            out = kwargs.pop('out', ())
            if ufunc is not np.multiply or method != '__call__' or kwargs:
                return NotImplemented

            # Convert inputs
            inputs = [np.asarray(x) if not isinstance(x, ArrayLike) else x
                      for x in inputs]

            # Do computation
            result = ArrayLike(...)

            # Manage output argument if any
            if out:
                out[0][...] = result
                return out[0]
            else:
                return result

        def __mul__(self, other):
            return self.__array_ufunc__(np.multiply, '__call__', self, other)

        def __rmul__(self, other):
            return self.__array_ufunc__(np.multiply, '__call__', other, self)

        def __imul__(self, other):
            result = self.__array_ufunc__(np.multiply, '__call__', self, other, out=(self,))
            if result is NotImplemented:
                # If you don't want to allow "x += y" -> "x = x + y"
                raise TypeError()
            return result

A point that needs to be considered more carefully is the implementation
of in-place operations. The main question is whether Python is allowed to
do the fallback ``x += y  ->  x = x + y`` or not.

We also suggest using the following boilerplate mix-in code:

.. code::

    # Copyright 2017 Google Inc.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    
    import numpy as np
    
    
    def _binary_method(ufunc):
        def func(self, other):
            return self.__array_ufunc__(ufunc, '__call__', self, other)
        return func
    
    def _reflected_binary_method(ufunc):
        def func(self, other):    
            return self.__array_ufunc__(ufunc, '__call__', other, self)
        return func
    
    def _inplace_binary_method(ufunc):
        def func(self, other):
            return self.__array_ufunc__(ufunc, '__call__', self, other, out=(self,))
        return func
    
    def _numeric_methods(ufunc):
        return (_binary_method(ufunc),
                _reflected_binary_method(ufunc),
                _inplace_binary_method(ufunc))
    
    def _unary_method(ufunc):
        def func(self):
            return self.__array_ufunc__(ufunc, '__call__', self)
        return func
    
    
    class UFuncSpecialMethodMixin(object):
        """Implements all special methods using __array_ufunc__."""
    
        # comparisons
        __lt__ = _binary_method(np.less)
        __le__ = _binary_method(np.less_equal)
        __eq__ = _binary_method(np.equal)
        __ne__ = _binary_method(np.not_equal)
        __gt__ = _binary_method(np.greater)
        __ge__ = _binary_method(np.greater_equal)
    
        # numeric methods
        __add__, __radd__, __iadd__ = _numeric_methods(np.add)
        __sub__, __rsub__, __isub__ = _numeric_methods(np.subtract)
        __mul__, __rmul__, __imul__ = _numeric_methods(np.multiply)
        __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(np.matmul)
        __div__, __rdiv__, __idiv__ = _numeric_methods(np.divide)  # Python 2 only
        __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(np.true_divide)
        __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(np.floor_divide)
        __mod__, __rmod__, __imod__ = _numeric_methods(np.mod)
        # No ufunc for __divmod__!
        # TODO: handle the optional third argument for __pow__?
        __pow__, __rpow__, __ipow__ = _numeric_methods(np.power)
        __lshift__, __rlshift__, __ilshift__ = _numeric_methods(np.left_shift)
        __rshift__, __rrshift__, __irshift__ = _numeric_methods(np.right_shift)
        __and__, __rand__, __iand__ = _numeric_methods(np.logical_and)
        __xor__, __rxor__, __ixor__ = _numeric_methods(np.logical_xor)
        __or__, __ror__, __ior__ = _numeric_methods(np.logical_or)
    
        # unary methods
        __neg__ =_unary_method(np.negative)
        # No ufunc for __pos__!
        __abs__ = _unary_method(np.absolute)
        __invert__ = _unary_method(np.invert)

    
    class ArrayLike(UFuncSpecialMethodMixin):
        """An array-like class that wraps a generic duck-array.

        Try using it to wrap your favorite duck array! (e.g., from dask.array,
        xarray or pandas)
        Example usage:
            >>> x = ArrayLike(np.array([1, 2, 3]))
            >>> x - 1
            ArrayLike(array([0, 1, 2]))
            >>> 1 - x
            ArrayLike(array([ 0, -1, -2]))
            >>> np.arange(3) - x
            ArrayLike(array([-1, -1, -1]))
            >>> x - np.arange(3)
            ArrayLike(array([1, 1, 1]))
        """
    
        def __init__(self, value):
            self.value = value

        __array_priority__ = 1000  # backward-compatibility with old Numpy

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            # Check inputs can be dealt with
            for item in inputs:
                if not hasattr(item, '__array_ufunc__'):
                    # Handle items without __array_ufunc__ (similarly to ndarray)
                    pass
                elif isinstance(item, ArrayLike):
                    # Handle own type
                    pass
                elif type(item) is np.ndarray:
                    # Handle ndarrays (but not its subclasses, to be safe)
                    pass
                else:
                    return NotImplemented

            # ArrayLike implements arithmetic and ufuncs by deferring to the wrapped array
            inputs = tuple(x.value if isinstance(self, type(x)) else x
                           for x in inputs)
            if kwargs.get('out') is not None:
                kwargs['out'] = tuple(x.value if isinstance(self, type(x)) else x
                                      for x in kwargs['out'])
            result = getattr(ufunc, method)(*inputs, **kwargs)
            if isinstance(result, tuple):
                return tuple(type(self)(x) for x in result)
            else:
                return type(self)(result)
    
        def __repr__(self):
            return '%s(%r)' % (type(self).__name__, self.value)


    class TransparentArrayLike(UFuncSpecialMethodMixin):
        """A transparent array-like class that wraps a generic duck-array.

        In contrast to the above wrapper class, this version inherits
        all binary operations supported by the wrapped duck-array,
        rather than restricting them to ArrayLike + ndarray.

        """
    
        def __init__(self, value):
            self.value = value

        __array_priority__ = 1000  # backward-compatibility with old Numpy

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            # ArrayLike implements arithmetic and ufuncs by deferring to the wrapped array
            inputs = tuple(x.value if isinstance(self, type(x)) else x
                           for x in inputs)
            if kwargs.get('out') is not None:
                kwargs['out'] = tuple(x.value if isinstance(self, type(x)) else x
                                      for x in kwargs['out'])

            # Masquerade as the wrapped object
            result = self.value.__array_ufunc__(ufunc, method, *inputs, **kwargs)
            if result is NotImplemented:
                return result
            elif isinstance(result, tuple):
                return tuple(type(self)(x) for x in result)
            else:
                return type(self)(result)
    
        def __repr__(self):
            return '%s(%r)' % (type(self).__name__, self.value)



Extension to other numpy functions
==================================

The ``__array_ufunc__`` method is used to override :func:`~numpy.dot`
and :func:`~numpy.matmul` as well, since while these functions are not
(yet) implemented as (generalized) Ufuncs, they are very similar.  For
other functions, such as :func:`~numpy.median`, :func:`~numpy.min`,
etc., implementations as (generalized) Ufuncs may well be possible and
logical as well, in which case it will become possible to override these
as well.

Demo
====

A pull request [8]_ has been made including the changes and revisions
proposed in this NEP.  Here is a demo highlighting the functionality.::

    In [1]: import numpy as np;

    In [2]: a = np.array([1])

    In [3]: class B():
       ...:     def __array_ufunc__(self, func, method, pos, inputs, **kwargs):
       ...:         return "B"
       ...:     

    In [4]: b = B()

    In [5]: np.dot(a, b)
    Out[5]: 'B'

    In [6]: np.multiply(a, b)
    Out[6]: 'B'

As a simple example, one could add the following ``__array_ufunc__`` to
SciPy's sparse matrices (just for ``np.dot`` and ``np.multiply`` as
these are the two most common cases where users would attempt to use
sparse matrices with ufuncs)::

    def __array_ufunc__(self, func, method, pos, inputs, **kwargs):
        """Method for compatibility with NumPy's ufuncs and dot
        functions.
        """

        without_self = list(inputs)
        without_self.pop(self)
        without_self = tuple(without_self)

        if func is np.multiply:
            return self.multiply(*without_self)

        elif func is np.dot:
            if pos == 0:
                return self.__mul__(inputs[1])
            if pos == 1:
                return self.__rmul__(inputs[0])
        else:
            return NotImplemented

So we now get the expected behavior when using ufuncs with sparse matrices.::

        In [1]: import numpy as np; import scipy.sparse as sp

        In [2]: a = np.random.randint(3, size=(3,3))

        In [3]: b = np.random.randint(3, size=(3,3))

        In [4]: asp = sp.csr_matrix(a); bsp = sp.csr_matrix(b)

        In [5]: np.dot(a,b)
        Out[5]: 
        array([[2, 4, 8],
               [2, 4, 8],
                [2, 2, 3]])

        In [6]: np.dot(asp,b)
        Out[6]: 
        array([[2, 4, 8],
               [2, 4, 8],
               [2, 2, 3]], dtype=int64)

        In [7]: np.dot(asp, bsp).A
        Out[7]: 
        array([[2, 4, 8],
               [2, 4, 8],
               [2, 2, 3]], dtype=int64)
                            
.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:

