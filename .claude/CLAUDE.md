# Cliodynamics.jl - Project Instructions

## Overview

Cliodynamics.jl is a Julia package for quantitative modeling of historical dynamics, implementing Peter Turchin's cliodynamic theories including demographic-structural theory, elite overproduction indices, and secular cycle analysis.

## Project Structure

- **src/Cliodynamics.jl** - Single-file module implementation (deliberate architectural decision)
- **test/runtests.jl** - Comprehensive test suite with @testset structure
- **examples/** - Julia usage examples (basic_usage.jl, historical_analysis.jl)
- **ffi/zig/** - Zig FFI implementation following Idris2 ABI
- **src/abi/** - Idris2 ABI definitions with formal proofs

## Building and Testing

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run examples
julia --project=examples examples/basic_usage.jl
julia --project=examples examples/historical_analysis.jl

# Interactive REPL
julia --project=.
```

## Code Style

### Julia Conventions
- Follow official Julia style guide
- Use lowercase with underscores for function names
- Use CamelCase for types
- Document all exported functions with docstrings
- Include `@testset` for all test groups
- Use `@test` with descriptive error messages

### Mathematical Functions
- All models return vectors (time series data)
- Parameters follow academic literature conventions
- Use keyword arguments for optional parameters
- Validate inputs at function boundaries

### Example Docstring Format
```julia
"""
    malthusian_model(r::Real, K::Real, P0::Real, t::AbstractVector{<:Real})

Simulate Malthusian population dynamics.

# Arguments
- `r`: Natural growth rate (per capita)
- `K`: Carrying capacity
- `P0`: Initial population
- `t`: Time vector

# Returns
Population vector over time
"""
```

## Architecture

### Single-File Design (ADR-002)
The entire implementation is in `src/Cliodynamics.jl` (not split into modules). This is intentional:
- Simplifies maintenance
- Clear dependency graph
- Easy to audit
- Standard Julia package practice for focused libraries

### Mathematical Models
- **Malthusian dynamics** - Population growth with carrying capacity
- **Demographic-Structural Theory (DST)** - Elite competition and social instability
- **Elite Overproduction Index (EOI)** - Ratio of elite aspirants to elite positions
- **Political Stress Indicator (PSI)** - Combined measure of multiple stressors
- **Secular cycles** - Long-term oscillations in state capacity

### Utility Functions
- Moving average, detrend, normalize, smooth
- Phase detection for secular cycles
- Peak detection for instability events

## Dependencies

### Core
- Julia 1.6+
- LinearAlgebra (stdlib)
- Statistics (stdlib)

### Optional (for examples)
- DataFrames.jl - Time series manipulation
- DifferentialEquations.jl - ODE integration
- Optim.jl - Parameter estimation

### Removed Dependencies
- Plots.jl removed to reduce package load (users import separately)

## Testing

Run full test suite:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Tests validate:
- Mathematical correctness of all models
- Edge cases (zero population, single data points, etc.)
- Numerical stability
- Type stability
- Exported function availability

## Common Workflows

### Adding a New Model
1. Add function to `src/Cliodynamics.jl`
2. Export function in module header
3. Add comprehensive docstring
4. Create `@testset` in `test/runtests.jl`
5. Update ROADMAP.adoc and STATE.scm
6. Run tests to verify

### Updating Documentation
- Update docstrings in source
- Update ROADMAP.adoc for milestones
- Update STATE.scm for completion tracking
- Update examples/ if API changes

### Integration Points
- **Seshat Global History Databank** - Empirical data source
- **DifferentialEquations.jl** - For ODE versions of models
- **Turing.jl** - For Bayesian parameter inference (planned v1.0)

## License

PMPL-1.0-or-later (Palimpsest License)

All files must have SPDX header:
```julia
# SPDX-License-Identifier: PMPL-1.0-or-later
```

## Notes for AI Agents

- This is a **Julia package**, not Rust/Elixir/ReScript
- The single-file design is intentional (don't suggest splitting)
- Mathematical models follow academic literature (cite Turchin if modifying)
- Test coverage is critical - all exported functions must have tests
- Examples should work standalone without Plots dependency
