# SPDX-License-Identifier: PMPL-1.0-or-later
# Historical Analysis Examples for Cliodynamics.jl

using Cliodynamics
using DataFrames
using Statistics

println("=== Cliodynamics.jl Historical Analysis Examples ===\n")

# Example 1: Phase Detection in Secular Cycles
println("1. Secular Cycle Phase Detection")
println("-" ^ 50)

# Simulate a secular cycle: expansion -> stagflation -> crisis -> depression
t = 0:200
expansion = (t .< 50) .* (100 .+ 0.5 .* t)
stagflation = ((t .>= 50) .& (t .< 100)) .* (125 .- 0.1 .* (t .- 50))
crisis = ((t .>= 100) .& (t .< 150)) .* (120 .- 0.8 .* (t .- 100))
depression = (t .>= 150) .* (80 .+ 0.2 .* (t .- 150))
cycle_data = expansion .+ stagflation .+ crisis .+ depression .+ 5 .* randn(length(t))

# Detect phases
phases = phase_detection(cycle_data, threshold=0.1)

println("Total phases detected: ", length(phases))
println("Phases:")
for (i, phase) in enumerate(phases)
    println("  Phase $i: $(phase.phase_type) (duration: $(phase.duration))")
end
println()

# Example 2: Identifying Instability Events
println("2. Instability Events Identification")
println("-" ^ 50)

# Create data with political stress indicator
years = 1700:1900
psi_data = 0.3 .+ 0.2 .* sin.(2π .* (years .- 1700) ./ 60) .+ 0.1 .* randn(length(years))
psi_data[150:160] .+= 0.8  # Add a crisis period around 1850

# Create DataFrame
instability_data = DataFrame(
    year = years,
    psi = psi_data,
    population_pressure = 0.5 .+ 0.3 .* rand(length(years))
)

# Identify instability events
events = instability_events(instability_data, psi_threshold=0.8)

println("Time period: 1700-1900")
println("Instability events detected: ", length(events))
if !isempty(events)
    println("\nEvents:")
    for event in events
        println("  Year: $(event.year), Severity: $(round(event.severity, digits=3))")
    end
end
println()

# Example 3: Conflict Intensity Analysis
println("3. Conflict Intensity Analysis")
println("-" ^ 50)

# Simulate conflict data: periods of peace and war
years_conflict = 1800:2000
baseline = 10.0
wars = zeros(length(years_conflict))
wars[20:30] .= 80.0   # War 1820-1830
wars[60:65] .= 120.0  # War 1860-1865
wars[114:118] .= 150.0  # War 1914-1918
wars[139:145] .= 180.0  # War 1939-1945

conflict_data = baseline .+ wars .+ 5 .* rand(length(years_conflict))

# Calculate conflict intensity
intensity = conflict_intensity(conflict_data, window=5)

println("Time period: 1800-2000")
println("Average conflict intensity: ", round(mean(intensity), digits=2))
println("Maximum intensity: ", round(maximum(intensity), digits=2))
println("Periods of high intensity (>100): ", count(intensity .> 100))
println()

# Example 4: Population Pressure Metrics
println("4. Population Pressure Analysis")
println("-" ^ 50)

# Create data showing population pressure over time
years_pressure = 1500:1800
pop_data = 100 .* exp.(0.003 .* (years_pressure .- 1500))  # Exponential growth
resource_data = 200 .+ 0.1 .* (years_pressure .- 1500)      # Slow resource growth

pressure_data = DataFrame(
    year = years_pressure,
    population = pop_data,
    resources = resource_data
)

# Calculate population pressure
pressure = population_pressure(pressure_data)

println("Time period: 1500-1800")
println("Initial pressure: ", round(pressure[1], digits=3))
println("Final pressure: ", round(pressure[end], digits=3))
println("Pressure increase: ", round(pressure[end] / pressure[1], digits=2), "x")
println("Years above crisis threshold (>0.8): ", count(pressure .> 0.8))
println()

# Example 5: Integrated Analysis - State Breakdown Risk
println("5. Integrated State Breakdown Risk Assessment")
println("-" ^ 50)

# Combine multiple indicators
years_integrated = 1600:1700

# Generate correlated indicators
elite_comp = 0.3 .+ 0.2 .* sin.(2π .* (years_integrated .- 1600) ./ 40) .+ 0.1 .* randn(length(years_integrated))
pop_press = 0.4 .+ 0.3 .* sin.(2π .* (years_integrated .- 1600) ./ 40 .+ π/4) .+ 0.1 .* randn(length(years_integrated))
state_weakness = 0.2 .+ 0.25 .* sin.(2π .* (years_integrated .- 1600) ./ 40 .+ π/2) .+ 0.1 .* randn(length(years_integrated))

# Composite risk score
risk_score = (elite_comp .+ pop_press .+ state_weakness) ./ 3

integrated_data = DataFrame(
    year = years_integrated,
    elite_competition = elite_comp,
    population_pressure = pop_press,
    state_weakness = state_weakness,
    risk_score = risk_score
)

println("Time period: 1600-1700")
println("\nAverage indicators:")
println("  Elite competition: ", round(mean(elite_comp), digits=3))
println("  Population pressure: ", round(mean(pop_press), digits=3))
println("  State weakness: ", round(mean(state_weakness), digits=3))
println("  Composite risk: ", round(mean(risk_score), digits=3))

# Find high-risk periods
high_risk_years = years_integrated[risk_score .> 0.6]
println("\nHigh-risk years (score > 0.6): ", length(high_risk_years))
if length(high_risk_years) > 0
    println("  First: ", high_risk_years[1], ", Last: ", high_risk_years[end])
end
println()

println("=== Historical Analysis Complete ===")
println("\nNote: These examples use synthetic data for demonstration.")
println("For real historical analysis, integrate with empirical datasets")
println("such as the Seshat Global History Databank.")
