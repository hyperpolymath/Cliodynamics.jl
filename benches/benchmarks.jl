# SPDX-License-Identifier: MPL-2.0
# (PMPL-1.0-or-later preferred; MPL-2.0 required for Julia ecosystem)
# BenchmarkTools benchmarks for Cliodynamics.jl
# Measures ODE model integration, indicator calculation, and spatial diffusion
# at small/medium/large time spans.

using BenchmarkTools
using Cliodynamics
using DataFrames

# ── Model integration benchmarks ──────────────────────────────────────────────

params_m = MalthusianParams(r=0.02, K=1000.0, N0=100.0)
params_d = DemographicStructuralParams(
    r=0.015, K=1000.0, w=2.0, δ=0.03, ε=0.001,
    N0=500.0, E0=10.0, S0=100.0,
)

println("=== malthusian_model (small: tspan 0–100) ===")
@benchmark malthusian_model($params_m, tspan=(0.0, 100.0))

println("=== malthusian_model (medium: tspan 0–500) ===")
@benchmark malthusian_model($params_m, tspan=(0.0, 500.0))

println("=== demographic_structural_model (large: tspan 0–1000) ===")
@benchmark demographic_structural_model($params_d, tspan=(0.0, 1000.0))

# ── Indicator benchmarks ──────────────────────────────────────────────────────

# Small: 50 data points
small_data = DataFrame(
    year       = collect(1800:1849),
    population = collect(100_000.0:1000.0:149_000.0),
    elites     = [1000 + 10i + 2i^2 for i in 0:49],
)

# Medium: 200 data points
medium_data = DataFrame(
    year       = collect(1700:1899),
    population = collect(80_000.0:1000.0:279_000.0),
    elites     = [800 + 8i + i^2 for i in 0:199],
)

println("=== elite_overproduction_index (small: 50 rows) ===")
@benchmark elite_overproduction_index($small_data)

println("=== elite_overproduction_index (medium: 200 rows) ===")
@benchmark elite_overproduction_index($medium_data)

# ── Secular cycle analysis benchmarks ────────────────────────────────────────

data_short = 100.0 .+ 30.0 .* sin.(2π .* (1:100) ./ 50) .+ randn(100)
data_long  = 100.0 .+ 30.0 .* sin.(2π .* (1:500) ./ 100) .+ randn(500)

println("=== secular_cycle_analysis (small: 100 points) ===")
@benchmark secular_cycle_analysis($data_short, window=20)

println("=== secular_cycle_analysis (large: 500 points) ===")
@benchmark secular_cycle_analysis($data_long, window=50)
