Symbol Glossary
===============

.. glossary::

    :math:`N`
        Total number of nodes.

    :math:`L`
        Length of segment between two nodes.

    :math:`x`
        Position.

    :math:`s`
        State.

    :math:`t`
        Time.

    :math:`grad`
        Finite derivative of :math:`s` w.r.t. :math:`x`.

    :math:`Q`
        System flux function.

    :math:`S`
        Spatial and state dependent forcing function.

    :math:`P`
        Positional and state dependent forcing function.

    :math:`\Delta s`
        Storage change.

    :math:`F_{external}`
        All point and spatial fluxes.

    :math:`F_{internal}`
        Internal fluxes in the system.

    :math:`F_{total}`
        :math:`F_{internal} + F{external}`, sum of all forcing fluxes.

    :math:`F_{net}`
        :math:`F_{total} - \Delta s`, net fluxes in the system.

    :math:`A`
        Sum of all jacobian matrices.

    :math:`A_{sys}`
        Jacobian system flux function.

    :math:`A_{spat}`
        Jacobian spatial state dependent forcing functions.

    :math:`A_{point}`
        Jacobian positional state dependent forcing function.

    :math:`i`
        Node number (1 indexed).

    :math:`j`
        Index of spatial state dependent forcing function number :math:`n`.

    :math:`k`
        Index of positional state dependent forcing function number :math:`m`.

    :math:`n`
        Total number of spatial state dependent forcing functions.

    :math:`m`
        Total number of positional state dependent forcing functions.

    :math:`P_{\Lambda}`
        Legendre polynomial of degree :math:`\Lambda`.

    :math:`\Lambda`
        Gaussian Quadrature degree.

    :math:`\lambda`
        Index of specific Gaussian quadrature position :math:`p` and weight :math:`w`.

    :math:`X`
        All absolute positions of the Gaussian integration procedure within the domain.

    :math:`p`
        All positions of the Gaussian integration procedure for degree :math:`\Lambda`.

    :math:`w`
        All weights of the Gaussian integration procedure for degree :math:`\Lambda`.

    :math:`lfac`
        Distribution factor for pointflux :math:`P` towards the nearest left node.

    :math:`rfac`
        Distribution factor for pointflux :math:`P` towards the nearest right node.

    :math:`_{l}`
        left node of the segment.

    :math:`_{r}`
        right node of the segment.
