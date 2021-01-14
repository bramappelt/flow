Symbol Glossary
===============

.. glossary::

    :math:`N`
        Total number of nodes, calculated as the length of
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.nodes`.

    :math:`L`
        Lengths of segments between the nodes in the shape :math:`[1 \times N - 1]`.
        See :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE._FE_precalc` for the exact definition.

    :math:`nL`
        Segment lengths in the shape :math:`[1 \times N]`. See
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE._FE_precalc` for the exact definition.

    :math:`X`
        Any of the precalculated absolute Gaussian integration points as saved in
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.xintegration`. The calculation of
        these values is performed in
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE.set_gaussian_quadrature`. Matrix of
        dimension :math:`[\Delta \times N - 1]`.

    :math:`x`
        A position within the domain as defined by
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.nodes`.

    :math:`s`
        A state within the domain as defined by
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.nodes`.

    :math:`t`
        Used as a subscript indicating time.

    :math:`grad`
        Finite derivative of :math:`s` w.r.t. :math:`x` as argument to
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.systemfluxfunc` and
        calculated in
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE.set_systemfluxfunction`.

    :math:`Q`
        System flux function saved in the model under
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.systemfluxfunc`.

    :math:`S`
        Spatial and state dependent forcing functions saved in the model under
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.spatflux` and
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.Sspatflux`.

    :math:`P`
        Positional and State dependent forcing functions saved in the model under
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.pointflux` and
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.Spointflux`.

    :math:`\Delta s`
        Storage change w.r.t. previous time step. The calculation is performed by
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE.calcbalance`. Matrix of
        dimension :math:`[1 \times N]`.

    :math:`F_{external}`
        Accumulation of all point and spatial fluxes. The calculation is perfomed
        by :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE.calcbalance`. Matrix of
        dimension :math:`[1 \times N]`.

    :math:`F_{internal}`
        Internal fluxes in the system. The calculation is performed by
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE._internal_forcing`. Matrix
        of dimension :math:`[1 \times N]`.

    :math:`F_{total}`
        :math:`F_{internal} + F_{external}`, sum of all forcing fluxes,
        **excluding** the storage change forcing. The calculation is performed by
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE.calcbalance`.

    :math:`F_{net}`
        :math:`F_{total} - \Delta s`, net fluxes in the system. The calculation is
        performed by :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE.calcbalance`.

    :math:`F_{forcing}`
        :math:`F_{internal} + F_{external}`, global sum of all forcing fluxes,
        **including** the storage change forcing. This data is saved in
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.forcing`. Matrix of dimension
        :math:`[1 \times N]`.

    :math:`A`
        :math:`A = A_{sys} + A_{spat} + A_{point}`, sum of all jacobian matrices. This
        jacobian is saved in :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.coefmatr`,
        and is calculated in :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE._CMAT`.
        Matrix of dimension :math:`[N \times N]`.

    :math:`A_{sys}`
        Jacobian of :math:`Q`, calculated by
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE._CMAT`. Matrix of dimension
        :math:`[N \times N]`.

    :math:`A_{spat}`
        Jacobian spatial state dependent forcing functions :math:`S`, calculated by
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE._CMAT`. Matrix of dimension
        :math:`[N \times N]`.

    :math:`A_{point}`
        Jacobian state dependent forcing function :math:`P`, calculated by
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE._CMAT`. Matrix of dimension
        :math:`[N \times N]`.

    :math:`i`
        Node number (1 indexed) used a subscript.

    :math:`j`
        Index, used as a subscript, to indicate any of the :math:`n` spatial state
        dependent forcing functions.

    :math:`k`
        Index, used as a subscript, to indicate any of the :math:`m` state dependent
        forcing functions.

    :math:`n`
        Total number of spatial state dependent forcing functions as saved in
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.Sspatflux`.

    :math:`m`
        Total number of state dependent forcing functions as saved in
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.Spointflux`.

    :math:`P_{\Lambda}`
        Legendre polynomial of degree :math:`\Lambda`, see
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE.set_gaussian_quadrature`.

    :math:`\Lambda`
        Gaussian Quadrature degree as saved in
        :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE.gauss_degree`.

    :math:`\lambda`
        Index, used as subscript, of a specific Gaussian quadrature degree.

    :math:`p`
        All positions of the Gaussian integration procedure for degree :math:`\Lambda`.
        Saved in the model under :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE._xgauss`.

    :math:`w`
        All weights of the Gaussian integration procedure for degree :math:`\Lambda`.
        Saved in the model under :py:attr:`~waterflow.flow1d.flowFE1d.Flow1DFE._wgauss`.

    :math:`lfac`
        Distribution factor for pointflux :math:`P` towards the nearest left node.
        The definition and calculation formula can be found in
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE.add_pointflux`.

    :math:`rfac`
        Distribution factor for pointflux :math:`P` towards the nearest right node.
        The definition and calculation formula can be found in
        :py:meth:`~waterflow.flow1d.flowFE1d.Flow1DFE.add_pointflux`.

    :math:`l`
        Subscript to indicate the nearest left element.

    :math:`r`
        Subscript to indicate the nearest right element.
