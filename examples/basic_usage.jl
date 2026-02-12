# SPDX-License-Identifier: PMPL-1.0-or-later
# Basic Usage Examples for Cliodynamics.jl

using Cliodynamics
using DataFrames

println("=== Cliodynamics.jl Basic Usage Examples ===\n")

# Example 1: Malthusian Population Model
println("1. Malthusian Population Dynamics")
println("-" ^ 40)

# Define parameters: growth rate r=2%, carrying capacity K=1000, initial population N0=100
params_malthus = MalthusianParams(r=0.02, K=1000.0, N0=100.0)

# Simulate over 200 time units
result_malthus = malthusian_model(params_malthus, tspan=(0.0, 200.0))

println("Initial population: ", params_malthus.N0)
println("Carrying capacity: ", params_malthus.K)
println("Growth rate: ", params_malthus.r)
println("Final population (t=200): ", round(result_malthus[end, 2], digits=2))
println()

# Example 2: Demographic-Structural Theory (DST) Model
println("2. Demographic-Structural Theory Model")
println("-" ^ 40)

# Parameters for DST model with elite-commoner dynamics
params_dst = DemographicStructuralParams(
    r=0.015,        # Population growth rate
    K=1000.0,       # Carrying capacity
    N0=500.0,       # Initial commoner population
    E0=50.0,        # Initial elite population
    elite_ratio=0.05 # Target elite-to-population ratio
)

# Simulate DST model
result_dst = demographic_structural_model(params_dst, tspan=(0.0, 300.0))

println("Initial commoners: ", params_dst.N0)
println("Initial elites: ", params_dst.E0)
println("Final commoners (t=300): ", round(result_dst[end, 2], digits=2))
println("Final elites (t=300): ", round(result_dst[end, 3], digits=2))
println()

# Example 3: Elite Overproduction Index
println("3. Elite Overproduction Index")
println("-" ^ 40)

# Create synthetic historical data
data_eoi = DataFrame(
    year = 1800:1900,
    population = 100 .+ cumsum(randn(101)),  # Random walk population
    elites = 10 .+ cumsum(0.5 .* randn(101))  # Elite population growing faster
)

# Calculate elite overproduction index
eoi = elite_overproduction_index(data_eoi)

println("Time period: 1800-1900")
println("Average elite/population ratio: ", round(mean(data_eoi.elites ./ data_eoi.population), digits=4))
println("Elite overproduction index: ", round(eoi, digits=4))
println("(Higher values indicate greater elite overproduction)")
println()

# Example 4: Political Stress Indicator (PSI)
println("4. Political Stress Indicator")
println("-" ^ 40)

# Create data for PSI calculation
data_psi = DataFrame(
    year = 1850:1950,
    population = 200 .+ 5 .* (1:101),  # Growing population
    resources = 1000 .- 2 .* (1:101),   # Declining resources
    elites = 20 .+ 0.8 .* (1:101)       # Growing elite population
)

# Calculate PSI
psi_values = political_stress_indicator(data_psi)

println("Time period: 1850-1950")
println("Average PSI: ", round(mean(psi_values), digits=4))
println("Max PSI (crisis point): ", round(maximum(psi_values), digits=4))
println("(Higher PSI indicates greater political instability risk)")
println()

# Example 5: Secular Cycle Analysis
println("5. Secular Cycle Analysis")
println("-" ^ 40)

# Create synthetic time series with cyclical pattern
t = 0:200
timeseries = 100 .+ 20 .* sin.(2π .* t ./ 50) .+ 5 .* randn(length(t))

# Analyze secular cycles
cycle_result = secular_cycle_analysis(timeseries, window=25)

println("Time series length: ", length(timeseries))
println("Analysis window: 25 time units")
println("Detected trends: ", length(cycle_result))
println()

println("=== Examples Complete ===")
println("\nFor more advanced examples, see historical_analysis.jl")
