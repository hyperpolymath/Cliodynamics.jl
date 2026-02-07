# Cliodynamics.jl

[![License](https://img.shields.io/badge/license-PMPL--1.0--or--later-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/julia-1.6+-purple.svg)](https://julialang.org)

A Julia library for mathematical modeling and statistical analysis of historical dynamics.

## Overview

Cliodynamics is the scientific study of historical dynamics - applying mathematical models and quantitative methods to understand long-term patterns in social complexity, state formation, demographic cycles, elite dynamics, and political instability. This package implements frameworks from Peter Turchin's research program in cliodynamics.

## What is Cliodynamics?

Cliodynamics treats history as a science, using mathematical models to analyze:
- Population dynamics and demographic pressures
- Elite overproduction and intra-elite competition
- Political instability and state breakdown
- Secular cycles (150-300 year oscillations in societies)
- State capacity and collective action problems

The field bridges history, mathematics, evolutionary theory, and complex systems science.

## Features

### Population Dynamics
- **Malthusian Models**: Logistic population growth with carrying capacity constraints
- **Demographic-Structural Theory (DST)**: Coupled models of population, elites, and state capacity
- **Population Pressure**: Measurement of stress relative to carrying capacity

### Elite Dynamics
- **Elite Overproduction Index**: Quantify when elite supply exceeds available positions
- **Elite-to-Population Ratios**: Track elite expansion relative to general population
- **Intra-Elite Competition**: Model competition effects on political stability

### Political Instability
- **Political Stress Indicator (PSI)**: Composite measure combining:
  - Mass Mobilization Potential (wage decline, immiseration)
  - Elite Mobilization Potential (elite overproduction)
  - State Fiscal Distress (revenue crisis)
- **Instability Probability**: Convert stress indicators to event probabilities
- **Conflict Intensity**: Aggregate historical instability events over time

### Secular Cycles
- **Cycle Detection**: Identify 150-300 year oscillations in historical data
- **Phase Classification**: Categorize periods as Expansion, Stagflation, Crisis, or Depression
- **Trend-Cycle Decomposition**: Separate long-term trends from cyclical components

### State Formation
- **State Capacity Models**: Tax revenue, military strength, institutional quality
- **Collective Action Problems**: Model cooperation challenges in state building
- **Institutional Development**: Track state-building trajectories

## Installation

```julia
using Pkg
Pkg.add("Cliodynamics")
```

## Quick Start

```julia
using Cliodynamics
using DataFrames
using Plots

# 1. Model Malthusian population dynamics
params = MalthusianParams(r=0.02, K=1000.0, N0=100.0)
sol = malthusian_model(params, tspan=(0.0, 200.0))
plot(sol, xlabel="Time", ylabel="Population", title="Logistic Growth")

# 2. Calculate elite overproduction index
data = DataFrame(
    year = 1800:1900,
    population = 100_000:1000:200_000,
    elites = [1000 + 10*i + 5*i^1.5 for i in 0:100]
)
eoi = elite_overproduction_index(data)
plot(eoi.year, eoi.eoi, xlabel="Year", ylabel="Elite Overproduction Index")

# 3. Calculate Political Stress Indicator
stress_data = DataFrame(
    year = 1800:1900,
    real_wages = 100.0 .- (0:100).^1.2,
    elite_ratio = 0.01 .+ (0:100)./5000,
    state_revenue = 1000.0 .- (0:100).^1.5
)
psi = political_stress_indicator(stress_data)
plot(psi.year, psi.psi, xlabel="Year", ylabel="Political Stress Indicator")

# 4. Detect secular cycles
timeseries = rand(300)  # Replace with historical data
analysis = secular_cycle_analysis(timeseries, window=50)
println("Estimated cycle period: $(analysis.period) years")
```

## Demographic-Structural Theory Example

Model a complete secular cycle with coupled population, elite, and state dynamics:

```julia
using Cliodynamics
using Plots

# Set up DST parameters for a typical agrarian society
params = DemographicStructuralParams(
    r=0.015,      # Population growth rate (1.5% per year)
    K=1000.0,     # Carrying capacity
    w=2.0,        # Elite wage premium
    δ=0.03,       # Elite death/retirement rate
    ε=0.001,      # Elite production rate
    N0=500.0,     # Initial population (50% of capacity)
    E0=10.0,      # Initial elite population
    S0=100.0      # Initial state fiscal capacity
)

# Simulate 300 years (typical secular cycle length)
sol = demographic_structural_model(params, tspan=(0.0, 300.0))

# Extract time series
t = sol.t
population = [u[1] for u in sol.u]
elites = [u[2] for u in sol.u]
state_capacity = [u[3] for u in sol.u]

# Visualize the secular cycle
plot(
    plot(t, population, title="Population", ylabel="N"),
    plot(t, elites, title="Elites", ylabel="E"),
    plot(t, state_capacity, title="State Capacity", ylabel="S"),
    layout=(3,1), xlabel="Years"
)
```

## Historical Analysis Example

Analyze a real historical crisis period:

```julia
using Cliodynamics
using DataFrames
using Plots

# Example: Model the crisis of the Roman Republic (133-27 BCE)
rome_data = DataFrame(
    year = -133:1:27,
    population_pressure = # ... historical data
    elite_overproduction = # ... senatorial class expansion
    instability = # ... civil wars, conspiracies, etc.
)

# Classify secular cycle phases
phases = detect_cycle_phases(rome_data)

# Count phases
phase_counts = combine(groupby(phases, :phase), nrow => :count)

# Identify crisis periods (high instability)
crisis_periods = filter(row -> row.phase == Crisis, phases)

# Calculate political stress
psi_data = DataFrame(
    year = -133:1:27,
    real_wages = # ... wage data
    elite_ratio = # ... elite-to-population ratio
    state_revenue = # ... fiscal data
)
psi = political_stress_indicator(psi_data)

# Find critical threshold
threshold = crisis_threshold(psi.psi, 0.9)
println("Crisis threshold: PSI > $(round(threshold, digits=2))")

# Extract instability events
events = instability_events(
    DataFrame(year=-133:27, indicator=psi.psi),
    threshold
)

# Visualize
plot(psi.year, psi.psi, xlabel="Year", ylabel="Political Stress Indicator",
     title="Crisis of the Roman Republic", label="PSI")
hline!([threshold], label="Crisis Threshold", linestyle=:dash)
```

## Key Concepts

### Malthusian Dynamics
Population grows until constrained by resources (carrying capacity K), following:
```
dN/dt = r*N*(1 - N/K)
```

### Elite Overproduction
When elite aspirants exceed available positions, intra-elite competition intensifies, destabilizing the political system. Measured as:
```
EOI = (E/N) / (E/N)_baseline - 1
```

### Political Stress Indicator
Composite index combining three destabilizing forces:
```
PSI = 0.4*MMP + 0.4*EMP + 0.2*SFD
```
where:
- MMP = Mass Mobilization Potential (popular immiseration)
- EMP = Elite Mobilization Potential (elite overproduction)
- SFD = State Fiscal Distress (revenue crisis)

### Secular Cycles
Long-term oscillations (150-300 years) with four phases:
1. **Expansion**: Low pressure, state strengthening, prosperity
2. **Stagflation**: Rising pressure, elite overproduction begins
3. **Crisis**: Political instability, state breakdown, conflict
4. **Depression/Intercycle**: Population decline, elite winnowing, recovery

## References

### Primary Sources
- Turchin, P. (2003). *Historical Dynamics: Why States Rise and Fall*. Princeton University Press.
- Turchin, P. (2016). *Ages of Discord: A Structural-Demographic Analysis of American History*. Beresta Books.
- Turchin, P., & Nefedov, S. A. (2009). *Secular Cycles*. Princeton University Press.
- Turchin, P. (2023). *End Times: Elites, Counter-Elites, and the Path of Political Disintegration*. Penguin Press.

### Theoretical Foundations
- Goldstone, J. A. (1991). *Revolution and Rebellion in the Early Modern World*. University of California Press.
- Korotayev, A., & Tsirel, S. (2010). "A Spectral Analysis of World GDP Dynamics." *Structure and Dynamics*, 4(1).

### Applications
- Turchin, P., et al. (2018). "Quantitative historical analysis uncovers a single dimension of complexity that structures global variation in human social organization." *PNAS*, 115(2), E144-E151.

## Citation

If you use this package in research, please cite:

```bibtex
@software{cliodynamics_jl,
  author = {Jewell, Jonathan D.A.},
  title = {Cliodynamics.jl: Mathematical Modeling of Historical Dynamics},
  year = {2026},
  url = {https://github.com/hyperpolymath/Cliodynamics.jl}
}
```

And cite the foundational work:

```bibtex
@book{turchin2003historical,
  author = {Turchin, Peter},
  title = {Historical Dynamics: Why States Rise and Fall},
  year = {2003},
  publisher = {Princeton University Press}
}
```

## Related Projects

- [Seshat Global History Databank](http://seshatdatabank.info/) - Database for testing cliodynamic theories
- [Crisis DB](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1743019) - Political instability database
- [Cliometrics.jl](https://github.com/hyperpolymath/Cliometrics.jl) - Quantitative economic history

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Palimpsest License (PMPL-1.0-or-later). See [LICENSE](LICENSE) for details.
