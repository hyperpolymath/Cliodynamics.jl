<!--
SPDX-License-Identifier: MPL-2.0
Copyright (c) Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
-->
# Population Dynamics

## Malthusian Model

The Malthusian logistic growth model describes population dynamics constrained by carrying capacity:

```math
\frac{dN}{dt} = rN\left(1 - \frac{N}{K}\right)
```

where:
- ``N`` = population size
- ``r`` = intrinsic growth rate
- ``K`` = carrying capacity (resource-limited maximum)

```@docs
MalthusianParams
malthusian_model
```

## Demographic-Structural Theory

The DST model couples three state variables in a system of ODEs:

- **Population** (``N``): Grows logistically, modulated by state capacity
- **Elites** (``E``): Produced from population, subject to competition
- **State capacity** (``S``): Revenue from taxation, eroded by elite demands

```@docs
DemographicStructuralParams
demographic_structural_model
```

## Population Pressure

Measure demographic stress relative to carrying capacity:

```@docs
population_pressure
carrying_capacity_estimate
```
