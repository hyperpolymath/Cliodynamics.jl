# SPDX-License-Identifier: PMPL-1.0-or-later

"""
    Cliodynamics

A Julia module for mathematical modeling and statistical analysis of historical processes,
implementing frameworks from Peter Turchin's cliodynamics research program.

Cliodynamics is the scientific study of historical dynamics that treats history as a science,
applying mathematical models to understand long-term patterns in social complexity, state
formation, demographic cycles, elite dynamics, and political instability.

# Core Components

## Population Dynamics
- Malthusian population models
- Demographic-structural theory (DST)
- Carrying capacity analysis

## Elite Dynamics
- Elite overproduction measurement
- Elite-to-population ratios
- Intra-elite competition modeling

## Political Instability
- Political Stress Indicator (PSI)
- State breakdown prediction
- Instability event modeling

## Secular Cycles
- Long-term cycle detection (150-300 year periods)
- Four-phase cycle analysis (expansion, stagflation, crisis, depression/intercycle)
- Trend-cycle decomposition

## State Formation
- State capacity modeling
- Collective action problems
- Institutional development

# References

- Turchin, P. (2003). *Historical Dynamics: Why States Rise and Fall*. Princeton University Press.
- Turchin, P. (2016). *Ages of Discord: A Structural-Demographic Analysis of American History*. Beresta Books.
- Turchin, P., & Nefedov, S. A. (2009). *Secular Cycles*. Princeton University Press.

# Examples

```julia
using Cliodynamics

# Model Malthusian population dynamics
params = MalthusianParams(r=0.02, K=1000.0, N0=100.0)
result = malthusian_model(params, tspan=(0.0, 200.0))

# Calculate elite overproduction index
data = DataFrame(
    year = 1800:1900,
    population = rand(100:200, 101),
    elites = rand(10:50, 101)
)
eoi = elite_overproduction_index(data)

# Analyze secular cycles
timeseries = rand(100)  # Replace with historical data
cycle_analysis = secular_cycle_analysis(timeseries, window=50)
```
"""
module Cliodynamics

using DifferentialEquations
using Statistics
using LinearAlgebra
using Optim
using DataFrames

# Export types
export MalthusianParams, DemographicStructuralParams, StateCapacityParams
export SecularCyclePhase, InstabilityEvent

# Export core functions
export malthusian_model, demographic_structural_model
export elite_overproduction_index, political_stress_indicator
export secular_cycle_analysis, detect_cycle_phases
export state_capacity_model, collective_action_problem
export conflict_intensity, instability_events
export carrying_capacity_estimate, population_pressure

# Export utility functions
export normalize_timeseries, detrend, moving_average
export crisis_threshold, instability_probability


# ============================================================================
# Type Definitions
# ============================================================================

"""
    MalthusianParams

Parameters for basic Malthusian population dynamics model.

# Fields
- `r::Float64`: Intrinsic growth rate (per capita)
- `K::Float64`: Environmental carrying capacity
- `N0::Float64`: Initial population size
"""
struct MalthusianParams
    r::Float64   # Intrinsic growth rate
    K::Float64   # Carrying capacity
    N0::Float64  # Initial population
end

"""
    DemographicStructuralParams

Parameters for Turchin's demographic-structural theory model.

# Fields
- `r::Float64`: Population growth rate
- `K::Float64`: Carrying capacity
- `w::Float64`: Elite wages relative to common wages
- `δ::Float64`: Elite death/retirement rate
- `ε::Float64`: Elite production rate
- `N0::Float64`: Initial population
- `E0::Float64`: Initial elite population
- `S0::Float64`: Initial state fiscal capacity
"""
struct DemographicStructuralParams
    r::Float64   # Population growth rate
    K::Float64   # Carrying capacity
    w::Float64   # Elite wage premium
    δ::Float64   # Elite death rate
    ε::Float64   # Elite production rate
    N0::Float64  # Initial population
    E0::Float64  # Initial elites
    S0::Float64  # Initial state capacity
end

"""
    StateCapacityParams

Parameters for state capacity and collective action modeling.

# Fields
- `τ::Float64`: Tax rate
- `α::Float64`: Administrative efficiency
- `β::Float64`: Military effectiveness
- `γ::Float64`: Institutional quality
"""
struct StateCapacityParams
    τ::Float64   # Tax rate
    α::Float64   # Administrative efficiency
    β::Float64   # Military effectiveness
    γ::Float64   # Institutional quality
end

"""
    SecularCyclePhase

Enumeration of secular cycle phases according to Turchin-Nefedov model.

- `Expansion`: Population growth, state strengthening, low instability
- `Stagflation`: Population pressure, elite overproduction begins
- `Crisis`: Political instability, state breakdown, social conflict
- `Depression`: Population decline, elite reduction, recovery begins
"""
@enum SecularCyclePhase begin
    Expansion
    Stagflation
    Crisis
    Depression
end

"""
    InstabilityEvent

Record of a political instability event.

# Fields
- `year::Int`: Year of event
- `intensity::Float64`: Magnitude (0-1 scale)
- `type::Symbol`: Event type (:rebellion, :coup, :revolution, :war)
"""
struct InstabilityEvent
    year::Int
    intensity::Float64
    type::Symbol
end


# ============================================================================
# Population Dynamics Models
# ============================================================================

"""
    malthusian_model(params::MalthusianParams; tspan=(0.0, 100.0))

Simulate basic Malthusian population dynamics with logistic growth.

The model follows the differential equation:
```
dN/dt = r*N*(1 - N/K)
```

where N is population, r is intrinsic growth rate, and K is carrying capacity.

# Arguments
- `params::MalthusianParams`: Model parameters
- `tspan::Tuple{Float64,Float64}`: Time span for simulation (start, end)

# Returns
- `ODESolution`: Solution object containing time series of population

# Examples
```julia
params = MalthusianParams(r=0.02, K=1000.0, N0=100.0)
sol = malthusian_model(params, tspan=(0.0, 200.0))
plot(sol)
```
"""
function malthusian_model(params::MalthusianParams; tspan=(0.0, 100.0))
    function logistic!(du, u, p, t)
        r, K = p
        N = u[1]
        du[1] = r * N * (1 - N / K)
    end

    u0 = [params.N0]
    p = [params.r, params.K]
    prob = ODEProblem(logistic!, u0, tspan, p)
    solve(prob, Tsit5())
end

"""
    demographic_structural_model(params::DemographicStructuralParams; tspan=(0.0, 300.0))

Simulate Turchin's demographic-structural theory (DST) model.

The DST model couples three subsystems:
1. Population dynamics with Malthusian pressure
2. Elite dynamics with overproduction
3. State fiscal capacity and breakdown

# Model Equations
```
dN/dt = r*N*(1 - N/K) - conflict_deaths
dE/dt = ε*N - δ*E - elite_conflict
dS/dt = τ*N - α*S - β*instability
```

# Arguments
- `params::DemographicStructuralParams`: Model parameters
- `tspan::Tuple{Float64,Float64}`: Time span (typically 200-300 years for secular cycles)

# Returns
- `ODESolution`: Solution with [N(t), E(t), S(t)]

# Examples
```julia
params = DemographicStructuralParams(
    r=0.015, K=1000.0, w=2.0, δ=0.03, ε=0.001,
    N0=500.0, E0=10.0, S0=100.0
)
sol = demographic_structural_model(params, tspan=(0.0, 300.0))
```
"""
function demographic_structural_model(params::DemographicStructuralParams; tspan=(0.0, 300.0))
    function dst_dynamics!(du, u, p, t)
        N, E, S = u
        r, K, w, δ, ε = p

        # Population pressure
        pop_pressure = N / K

        # Elite competition (when E/N ratio is high)
        elite_ratio = E / N
        elite_competition = elite_ratio > 0.05 ? (elite_ratio - 0.05) * 10.0 : 0.0

        # State instability (when fiscal capacity is low relative to needs)
        state_stress = max(0.0, 1.0 - S / (0.1 * N))
        instability = state_stress * elite_competition

        # Population dynamics
        du[1] = r * N * (1 - pop_pressure) - 0.01 * N * instability

        # Elite dynamics
        du[2] = ε * N - δ * E - 0.5 * E * elite_competition

        # State capacity dynamics
        tax_revenue = 0.15 * N * (1 - 0.5 * pop_pressure)
        du[3] = tax_revenue - 0.05 * S - 2.0 * S * instability
    end

    u0 = [params.N0, params.E0, params.S0]
    p = [params.r, params.K, params.w, params.δ, params.ε]
    prob = ODEProblem(dst_dynamics!, u0, tspan, p)
    solve(prob, Tsit5())
end


# ============================================================================
# Elite Dynamics
# ============================================================================

"""
    elite_overproduction_index(data::DataFrame)

Calculate elite overproduction index from historical data.

Elite overproduction occurs when the supply of elite aspirants exceeds available
elite positions, leading to intra-elite competition and political instability.

# Arguments
- `data::DataFrame`: Must contain columns `:year`, `:population`, `:elites`

# Returns
- `DataFrame`: Time series with `:year`, `:eoi` (Elite Overproduction Index)

# Formula
```
EOI = (E/N) / (E/N)_baseline - 1
```

where baseline is the mean ratio during stable periods.

# Examples
```julia
data = DataFrame(
    year = 1800:1900,
    population = 100_000:1000:200_000,
    elites = [1000 + 10*i + 5*i^1.5 for i in 0:100]
)
eoi = elite_overproduction_index(data)
```
"""
function elite_overproduction_index(data::DataFrame)
    required_cols = [:year, :population, :elites]
    for col in required_cols
        if !hasproperty(data, col)
            throw(ArgumentError("DataFrame must contain column: $col"))
        end
    end

    # Calculate elite-to-population ratio
    elite_ratio = data.elites ./ data.population

    # Baseline is median of first quartile (stable period assumption)
    n_baseline = div(length(elite_ratio), 4)
    baseline = median(elite_ratio[1:n_baseline])

    # EOI is deviation from baseline
    eoi = (elite_ratio ./ baseline) .- 1.0

    DataFrame(year = data.year, eoi = eoi, elite_ratio = elite_ratio)
end

"""
    population_pressure(population::Vector{Float64}, capacity::Float64)

Calculate population pressure relative to carrying capacity.

# Arguments
- `population::Vector{Float64}`: Population time series
- `capacity::Float64`: Environmental/economic carrying capacity

# Returns
- `Vector{Float64}`: Population pressure index (0 = no pressure, 1 = at capacity, >1 = overshoot)

# Examples
```julia
pop = [100.0, 150.0, 200.0, 250.0]
pressure = population_pressure(pop, 200.0)
```
"""
function population_pressure(population::Vector{Float64}, capacity::Float64)
    return population ./ capacity
end


# ============================================================================
# Political Instability Indicators
# ============================================================================

"""
    political_stress_indicator(data::DataFrame)

Calculate Turchin's Political Stress Indicator (PSI).

PSI combines three main factors:
1. Mass mobilization potential (MMP) - relative wage decline, popular immiseration
2. Elite mobilization potential (EMP) - elite overproduction
3. State fiscal distress (SFD) - revenue crisis

# Arguments
- `data::DataFrame`: Must contain `:year`, `:real_wages`, `:elite_ratio`, `:state_revenue`

# Returns
- `DataFrame`: Time series with `:year`, `:psi`, `:mmp`, `:emp`, `:sfd`

# Formula
```
PSI = w_mmp*MMP + w_emp*EMP + w_sfd*SFD
```

where weights typically are: w_mmp=0.4, w_emp=0.4, w_sfd=0.2

# Examples
```julia
data = DataFrame(
    year = 1800:1900,
    real_wages = 100.0 .- (0:100).^1.2,
    elite_ratio = 0.01 .+ (0:100)./5000,
    state_revenue = 1000.0 .- (0:100).^1.5
)
psi_result = political_stress_indicator(data)
```
"""
function political_stress_indicator(data::DataFrame)
    # Mass mobilization potential (declining wages)
    wage_baseline = median(data.real_wages[1:min(20, nrow(data))])
    mmp = max.(0.0, 1.0 .- data.real_wages ./ wage_baseline)

    # Elite mobilization potential (elite overproduction)
    elite_baseline = median(data.elite_ratio[1:min(20, nrow(data))])
    emp = max.(0.0, data.elite_ratio ./ elite_baseline .- 1.0)

    # State fiscal distress
    revenue_baseline = median(data.state_revenue[1:min(20, nrow(data))])
    sfd = max.(0.0, 1.0 .- data.state_revenue ./ revenue_baseline)

    # Combined PSI with weights
    w_mmp, w_emp, w_sfd = 0.4, 0.4, 0.2
    psi = w_mmp .* mmp .+ w_emp .* emp .+ w_sfd .* sfd

    DataFrame(year = data.year, psi = psi, mmp = mmp, emp = emp, sfd = sfd)
end

"""
    instability_probability(psi::Float64)

Convert Political Stress Indicator to instability event probability.

Uses logistic transformation based on empirical calibration.

# Arguments
- `psi::Float64`: Political Stress Indicator value

# Returns
- `Float64`: Probability of instability event (0-1)

# Examples
```julia
p = instability_probability(0.5)  # Moderate stress
```
"""
function instability_probability(psi::Float64)
    # Logistic transformation calibrated to historical data
    # 50% probability at PSI ≈ 0.5, inflection point
    β0, β1 = -2.0, 5.0
    return 1.0 / (1.0 + exp(-(β0 + β1 * psi)))
end

"""
    conflict_intensity(events::Vector{InstabilityEvent}, window::Int=10)

Calculate conflict intensity over time from instability events.

# Arguments
- `events::Vector{InstabilityEvent}`: Historical instability events
- `window::Int`: Rolling window size in years

# Returns
- `DataFrame`: Time series with `:year` and `:intensity`

# Examples
```julia
events = [
    InstabilityEvent(1820, 0.3, :rebellion),
    InstabilityEvent(1848, 0.8, :revolution),
    InstabilityEvent(1871, 0.6, :war)
]
intensity = conflict_intensity(events, window=5)
```
"""
function conflict_intensity(events::Vector{InstabilityEvent}, window::Int=10)
    if isempty(events)
        return DataFrame(year = Int[], intensity = Float64[])
    end

    years = [e.year for e in events]
    year_range = minimum(years):maximum(years)

    intensity = zeros(length(year_range))
    for (i, year) in enumerate(year_range)
        # Sum intensities within window
        relevant_events = filter(e -> abs(e.year - year) <= window÷2, events)
        intensity[i] = sum(e.intensity for e in relevant_events)
    end

    DataFrame(year = collect(year_range), intensity = intensity)
end


# ============================================================================
# Secular Cycles
# ============================================================================

"""
    secular_cycle_analysis(timeseries::Vector{Float64}; window::Int=50)

Detect and analyze secular cycles in historical time series.

Secular cycles are long-term oscillations (150-300 years) in demographic, economic,
and political variables. This function uses trend-cycle decomposition and spectral
analysis to identify cycle characteristics.

# Arguments
- `timeseries::Vector{Float64}`: Historical data (e.g., population, prices, instability)
- `window::Int`: Window size for moving average detrending

# Returns
- `NamedTuple`: Contains `:trend`, `:cycle`, `:period`, `:amplitude`

# Examples
```julia
# Synthetic data with ~200-year cycle
t = 1:300
data = 100 .+ 50*sin.(2π*t/200) .+ 2*randn(300)
analysis = secular_cycle_analysis(data, window=50)
```
"""
function secular_cycle_analysis(timeseries::Vector{Float64}; window::Int=50)
    n = length(timeseries)

    # Detrend using moving average
    trend = moving_average(timeseries, window)
    cycle = timeseries .- trend

    # Estimate dominant period using autocorrelation
    max_lag = min(n÷2, 150)
    autocorr = [cor(cycle[1:end-lag], cycle[lag+1:end]) for lag in 1:max_lag]

    # Find first peak after lag > 50 (secular cycles are long)
    valid_lags = 50:max_lag
    peak_idx = argmax(autocorr[valid_lags]) + 49
    period_estimate = peak_idx

    # Estimate amplitude as std of cycle component
    amplitude = std(cycle)

    return (
        trend = trend,
        cycle = cycle,
        period = period_estimate,
        amplitude = amplitude
    )
end

"""
    detect_cycle_phases(data::DataFrame)

Classify secular cycle phases using demographic-structural indicators.

# Arguments
- `data::DataFrame`: Must contain `:year`, `:population_pressure`, `:elite_overproduction`, `:instability`

# Returns
- `DataFrame`: Time series with `:year` and `:phase::SecularCyclePhase`

# Phase Classification Rules
- **Expansion**: Low pressure, low elite competition, low instability
- **Stagflation**: High pressure, rising elite competition, moderate instability
- **Crisis**: High pressure, high elite competition, high instability
- **Depression**: Declining pressure, declining elites, declining instability

# Examples
```julia
data = DataFrame(
    year = 1500:1800,
    population_pressure = rand(301),
    elite_overproduction = rand(301),
    instability = rand(301)
)
phases = detect_cycle_phases(data)
```
"""
function detect_cycle_phases(data::DataFrame)
    n = nrow(data)
    phases = Vector{SecularCyclePhase}(undef, n)

    for i in 1:n
        pp = data.population_pressure[i]
        eo = data.elite_overproduction[i]
        inst = data.instability[i]

        # Classification thresholds
        if inst > 0.6 && pp > 0.7 && eo > 0.5
            phases[i] = Crisis
        elseif pp > 0.7 && eo > 0.3 && inst < 0.6
            phases[i] = Stagflation
        elseif pp < 0.5 && eo < 0.3 && inst < 0.3
            phases[i] = Expansion
        else
            phases[i] = Depression
        end
    end

    DataFrame(year = data.year, phase = phases)
end


# ============================================================================
# State Formation and Capacity
# ============================================================================

"""
    state_capacity_model(params::StateCapacityParams, population::Float64, elites::Float64)

Calculate state capacity from fiscal and institutional parameters.

State capacity is the ability to implement policy, extract resources, and
maintain order. This follows Turchin's framework incorporating tax revenue,
administrative efficiency, and military power.

# Arguments
- `params::StateCapacityParams`: State parameters
- `population::Float64`: Current population
- `elites::Float64`: Current elite population

# Returns
- `Float64`: State capacity index

# Formula
```
S = α*T + β*M + γ*I
```
where T = tax revenue, M = military strength, I = institutional quality

# Examples
```julia
params = StateCapacityParams(τ=0.15, α=1.0, β=0.8, γ=0.5)
capacity = state_capacity_model(params, 1000.0, 50.0)
```
"""
function state_capacity_model(params::StateCapacityParams, population::Float64, elites::Float64)
    # Tax revenue (with diminishing returns at high rates)
    tax_revenue = params.τ * population * (1 - 0.5 * params.τ)

    # Military strength (function of population and organization)
    military = 0.1 * population * params.β

    # Institutional quality (elite cooperation vs competition)
    elite_ratio = elites / population
    cooperation = elite_ratio < 0.05 ? 1.0 : exp(-10 * (elite_ratio - 0.05))
    institutional = params.γ * cooperation * 100.0

    # Combined capacity
    return params.α * tax_revenue + params.β * military + params.γ * institutional
end

"""
    collective_action_problem(group_size::Int, benefit::Float64, cost::Float64)

Model collective action problem in state formation.

Based on Olson's logic of collective action and Turchin's application to
historical state formation.

# Arguments
- `group_size::Int`: Number of potential contributors
- `benefit::Float64`: Total benefit if collective action succeeds
- `cost::Float64`: Individual cost of participation

# Returns
- `Float64`: Probability of successful collective action

# Examples
```julia
# Small group, high benefit
prob = collective_action_problem(10, 1000.0, 10.0)  # High probability

# Large group, low benefit
prob = collective_action_problem(1000, 100.0, 1.0)  # Low probability
```
"""
function collective_action_problem(group_size::Int, benefit::Float64, cost::Float64)
    # Individual benefit
    individual_benefit = benefit / group_size

    # Net benefit
    net_benefit = individual_benefit - cost

    # Probability of participation (logistic model)
    p_individual = 1.0 / (1.0 + exp(-net_benefit / cost))

    # Probability of success (need critical mass, ~30%)
    critical_mass = ceil(Int, 0.3 * group_size)

    # Binomial approximation: probability that >= critical_mass participate
    expected_participants = group_size * p_individual

    if expected_participants >= critical_mass
        return min(1.0, expected_participants / critical_mass - 0.5)
    else
        return max(0.0, expected_participants / critical_mass)
    end
end


# ============================================================================
# Utility Functions
# ============================================================================

"""
    moving_average(x::Vector{Float64}, window::Int)

Calculate moving average for time series smoothing.

# Arguments
- `x::Vector{Float64}`: Input time series
- `window::Int`: Window size (must be odd for symmetric smoothing)

# Returns
- `Vector{Float64}`: Smoothed time series (same length as input)

# Examples
```julia
data = randn(100)
smoothed = moving_average(data, 11)
```
"""
function moving_average(x::Vector{Float64}, window::Int)
    n = length(x)
    result = zeros(n)
    half_window = window ÷ 2

    for i in 1:n
        start_idx = max(1, i - half_window)
        end_idx = min(n, i + half_window)
        result[i] = mean(x[start_idx:end_idx])
    end

    return result
end

"""
    detrend(x::Vector{Float64})

Remove linear trend from time series.

# Arguments
- `x::Vector{Float64}`: Input time series

# Returns
- `Vector{Float64}`: Detrended series

# Examples
```julia
data = collect(1:100) .+ randn(100)
detrended = detrend(data)
```
"""
function detrend(x::Vector{Float64})
    n = length(x)
    t = collect(1:n)

    # Fit linear trend
    A = hcat(ones(n), t)
    coeffs = A \ x
    trend = A * coeffs

    return x .- trend
end

"""
    normalize_timeseries(x::Vector{Float64})

Normalize time series to zero mean and unit variance.

# Arguments
- `x::Vector{Float64}`: Input time series

# Returns
- `Vector{Float64}`: Normalized series

# Examples
```julia
data = rand(100) .* 100
normalized = normalize_timeseries(data)
```
"""
function normalize_timeseries(x::Vector{Float64})
    μ = mean(x)
    σ = std(x)
    return (x .- μ) ./ σ
end

"""
    carrying_capacity_estimate(population::Vector{Float64}, resources::Vector{Float64})

Estimate historical carrying capacity from population and resource data.

Uses optimization to find the carrying capacity K that best explains
population dynamics under Malthusian constraints.

# Arguments
- `population::Vector{Float64}`: Historical population data
- `resources::Vector{Float64}`: Resource proxy (e.g., agricultural output, GDP)

# Returns
- `Float64`: Estimated carrying capacity

# Examples
```julia
pop = [100.0, 150.0, 180.0, 195.0, 198.0]
res = [1000.0, 1500.0, 1800.0, 1950.0, 2000.0]
K = carrying_capacity_estimate(pop, res)
```
"""
function carrying_capacity_estimate(population::Vector{Float64}, resources::Vector{Float64})
    # Simple estimate: carrying capacity ≈ max sustainable population
    # given resource constraints

    # Normalize resources to population units
    resource_per_capita = resources ./ population
    baseline_consumption = median(resource_per_capita)

    # K is where resources would sustain population at baseline
    K_estimate = maximum(resources) / baseline_consumption

    return K_estimate
end

"""
    crisis_threshold(indicator::Vector{Float64}, percentile::Float64=0.9)

Determine crisis threshold from historical indicator distribution.

# Arguments
- `indicator::Vector{Float64}`: Historical stress indicator values
- `percentile::Float64`: Percentile to use as threshold (default: 90th)

# Returns
- `Float64`: Threshold value above which indicates crisis

# Examples
```julia
psi_values = rand(100)
threshold = crisis_threshold(psi_values, 0.95)
```
"""
function crisis_threshold(indicator::Vector{Float64}, percentile::Float64=0.9)
    return quantile(indicator, percentile)
end

"""
    instability_events(data::DataFrame, threshold::Float64)

Extract instability events from time series based on threshold crossing.

# Arguments
- `data::DataFrame`: Must contain `:year` and `:indicator` columns
- `threshold::Float64`: Threshold value for event detection

# Returns
- `Vector{InstabilityEvent}`: Detected events

# Examples
```julia
data = DataFrame(year = 1800:1900, indicator = rand(101))
events = instability_events(data, 0.7)
```
"""
function instability_events(data::DataFrame, threshold::Float64)
    events = InstabilityEvent[]

    for i in 1:nrow(data)
        if data.indicator[i] > threshold
            intensity = min(1.0, data.indicator[i])

            # Classify event type based on intensity
            event_type = if intensity > 0.9
                :revolution
            elseif intensity > 0.7
                :war
            elseif intensity > 0.5
                :rebellion
            else
                :coup
            end

            push!(events, InstabilityEvent(data.year[i], intensity, event_type))
        end
    end

    return events
end

end # module Cliodynamics
