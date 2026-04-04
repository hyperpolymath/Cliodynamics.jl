# SPDX-License-Identifier: MPL-2.0
# (PMPL-1.0-or-later preferred; MPL-2.0 required for Julia ecosystem)
# E2E pipeline tests for Cliodynamics.jl
# Tests the full cliodynamic analysis workflow: model → indicators → cycle detection
# → fitting → spatial diffusion.

using Test
using Cliodynamics
using DataFrames
using Statistics

@testset "E2E Pipeline Tests" begin

    @testset "Full pipeline: demographic-structural analysis" begin
        # 1. Run the demographic-structural model
        params = DemographicStructuralParams(
            r=0.015, K=1000.0, w=2.0, δ=0.03, ε=0.001,
            N0=500.0, E0=10.0, S0=100.0,
        )
        sol = demographic_structural_model(params, tspan=(0.0, 300.0))
        @test sol.u[1][1] ≈ 500.0
        @test all(u .>= 0.0 for u in sol.u)

        # 2. Extract a time series and compute EOI
        years = collect(0.0:10.0:300.0)
        data = DataFrame(
            year       = Int.(years),
            population = [sol(t)[1] for t in years],
            elites     = [sol(t)[2] for t in years],
        )
        eoi = elite_overproduction_index(data)
        @test hasproperty(eoi, :eoi)
        @test nrow(eoi) == nrow(data)

        # 3. Build PSI input and compute political stress
        psi_data = DataFrame(
            year          = Int.(years),
            real_wages    = 100.0 .- years ./ 3.0,
            elite_ratio   = eoi.elite_ratio,
            state_revenue = [sol(t)[3] for t in years],
        )
        psi_result = political_stress_indicator(psi_data)
        @test hasproperty(psi_result, :psi)
        @test all(psi_result.psi .>= 0.0)

        # 4. Detect instability events from PSI
        psi_data_merged = hcat(psi_data, DataFrame(indicator=psi_result.psi))
        threshold = crisis_threshold(psi_result.psi, 0.8)
        @test threshold >= 0.0
    end

    @testset "Full pipeline: secular cycle detection and spatial model" begin
        # 1. Generate a synthetic secular cycle signal
        t = 1:300
        signal = 100.0 .+ 40.0 .* sin.(2π .* t ./ 100) .+ 2.0 .* randn(300)
        analysis = secular_cycle_analysis(signal, window=30)
        @test haskey(analysis, :period)
        @test 70 < analysis.period < 130  # should detect ~100yr cycle

        # 2. Spatial diffusion between 3 regions
        regions = [
            (name=:RegionA, psi0=0.9, growth_rate=0.04),
            (name=:RegionB, psi0=0.3, growth_rate=0.02),
            (name=:RegionC, psi0=0.1, growth_rate=0.01),
        ]
        adjacency = [0.0 1.0 0.0;
                     1.0 0.0 1.0;
                     0.0 1.0 0.0]
        result = spatial_instability_diffusion(regions, adjacency;
                                               diffusion_rate=0.15, tspan=(0.0, 50.0))
        @test length(result.t) > 0
        @test size(result.psi, 2) == 3
        # B (adjacent to high-PSI A) should gain instability over time
        @test result.psi[end, 2] > result.psi[1, 2]
    end

    @testset "Error handling: missing required columns" begin
        bad_df = DataFrame(year=2000:2010)
        @test_throws ArgumentError elite_overproduction_index(bad_df)

        @test_throws ArgumentError spatial_instability_diffusion(
            [(name=:A, psi0=0.5, growth_rate=0.01)],
            zeros(2, 2),  # dimension mismatch
            diffusion_rate=0.1,
        )
    end

    @testset "Round-trip consistency: Malthusian model fit" begin
        # Simulate data from known parameters
        true_params = MalthusianParams(r=0.025, K=600.0, N0=80.0)
        sol = malthusian_model(true_params, tspan=(0.0, 100.0))
        years = collect(0.0:10.0:100.0)
        population = [sol(t)[1] for t in years]

        result = fit_malthusian(years, population; r_init=0.01, K_init=700.0)
        @test result.converged
        @test abs(result.params.r - 0.025) < 0.02
        @test abs(result.params.K - 600.0) < 80.0
    end

end
