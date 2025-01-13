
========================
The MHD formalism
========================

The MHD formalism is a fluid approximation of the plasma, which can be obtained from the first moments (0, 1 and 2) of the distribution function :math:`f_p` of each ion population.
The formula for each moment is given by the following equation:

.. math::
   M_i = \int mathbf{v}^i f_p(\mathbf{r}, \mathbf{v}, t) d\mathbf{v}

Thus, integrating the Vlasov equation over the velocity space gives us the equations for the following moments:
The integration of the 0th moment gives the continuity equation which depends on the first moment, which corresponds to the momentum :math:`\rho \mathbf{v}`:

.. math::
   \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0

The integration of the 1st moment gives the momentum equation, which depends on the second moment, which corresponds to the pressure tensor :math:`P`:

.. math::
   \frac{\partial (\rho \mathbf{v})}{\partial t} + \nabla \cdot \left[\rho \mathbf{vv} - \mathbf{BB} + \mathbf{I} P^*\right] = 0

The energy equation is obtained by simplifying the result of the integration of the 2nd moment:

.. math::
   \frac{ \partial E}{ \partial t} + \nabla \cdot [(E + P^*)\mathbf{v} + \frac{\mathbf{E} \times \mathbf{B}}{\mu_0}] = 0

with :math:`P^* = P + \frac{B^2}{2 \mu_0}` the total pressure.

As this system is based on the moments of the distribution function, infinite for any distibution, it requires a closure. We use the equaton of state:

.. math::
   E = \frac{P}{\gamma - 1} + \frac{\rho v^2}{2} + \frac{B^2}{2 \mu_0}

where :math:`\gamma` is the adiabatic index.

The faraday equation is added to the system for field evolution:

.. math::
   \frac{\partial \mathbf{B}}{\partial t} = - \nabla \times \mathbf{E}


Ideal MHD
---------

Ideal MHD use the ideal Ohm's law to compute the electric field used in faraday's equation:

.. math::
   \mathbf{E} = - \mathbf{v} \times \mathbf{B}


Hall MHD
--------

Hall MHD adds the Hall term to Ohm's law:

.. math::
   \mathbf{E} = - \mathbf{v} \times \mathbf{B} + \frac{1}{en} mathbf{J} \times \mathbf{B}

where :math:`\mathbf{J}` is computed with ampere's law:

.. math::
   \mathbf{J} = \frac{1}{\mu_0} \nabla \times \mathbf{B}

The MHD solver in PHARE also supports the resistive (:math:`\eta \mathbf{J}`) and hyper-resistive (:math:`- \nu \nabla^2 \mathbf{j}`) terms in Ohm's law.

.. math::
   \mathbf{E} = - \mathbf{v} \times \mathbf{B} + \frac{1}{en} \mathbf{J} \times \mathbf{B} + \eta \mathbf{J} - \nu \nabla^2 \mathbf{J}

