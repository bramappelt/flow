flow
====

.. toctree::
   :maxdepth: 3

   waterflow

.. warning::
    From the Python_ docs:

    "Private instance variables that cannot be accessed except from inside an
    object don’t exist in Python. However, there is a convention that is followed
    by most Python code: a name prefixed with an underscore (e.g. _spam) should be
    treated as a non-public part of the API (whether it is a function, a method or a data member).
    It should be considered an implementation detail and subject to change without notice."

    However this is also true for this Python module, all private attributes, functions and
    methods are shown in the documentation as if they are `normal` members, but keep in mind that
    they should only be called internally by other objects to ensure the right context.

.. _Python: https://docs.python.org/3/tutorial/classes.html#private-variables
