# SPDX-License-Identifier: MPL-2.0
# (PMPL-1.0-or-later preferred; MPL-2.0 required for Julia ecosystem)
# Property-based tests for Cliodynamics.jl
# Verifies mathematical invariants of cliodynamic models across random inputs.

using Test
using Cliodynamics
using DataFrames
using Statistics

@testset "Property-Based Tests" begin

    @testset "Invariant: Malthusian model population stays non-negative" begin
        for _ in 1:50
            r  = rand() * 0.05          # growth rate 0–0.05
            K  = rand(100.0:100.0:2000.0)
            N0 = rand(10.0:10.0:K)     # start below K
            params = MalthusianParams(r=r, K=K, N0=N0)
            sol = malthusian_model(params, tspan=(0.0, 100.0))
            @test all(u[1] >= 0.0 for u in sol.u)
        end
    end

    @testset "Invariant: instability_probability in [0, 1]" begin
        for _ in 1:50
            psi = rand()  # random PSI value in [0,1]
            prob = instability_probability(psi)
            @test 0.0 <= prob <= 1.0
        end
    end

    @testset "Invariant: instability_probability is monotone in PSI" begin
        for _ in 1:50
            psi_low  = rand() * 0.5
            psi_high = psi_low + rand() * 0.5
            @test instability_probability(psi_low) <= instability_probability(psi_high)
        end
    end

    @testset "Invariant: collective_action_problem result in [0, 1]" begin
        for _ in 1:50
            group_size = rand(1:500)
            benefit    = rand(10.0:10.0:10_000.0)
            cost       = rand(1.0:1.0:100.0)
            prob = collective_action_problem(group_size, benefit, cost)
            @test 0.0 <= prob <= 1.0
        end
    end

    @testset "Invariant: population_pressure scales linearly with population" begin
        for _ in 1:50
            K   = rand(200.0:100.0:2000.0)
            n   = rand(2:8)
            pop = sort(rand(n) .* K .* 1.5)  # random ascending population values
            pressure = population_pressure(pop, K)
            @test length(pressure) == n
            # pressure[i] = pop[i] / K, so it increases with pop
            @test issorted(pressure)
            @test pressure[end] ≈ pop[end] / K  atol=1e-10
        end
    end

    @testset "Invariant: EOI increases when elite growth accelerates" begin
        for _ in 1:50
            n = rand(20:50)
            base_pop = collect(100_000:1000:(100_000 + 1000*(n-1)))
            # Accelerating elite growth outpaces linear population growth
            elites = [500 + 5*i + 2*i^2 for i in 0:(n-1)]
            data = DataFrame(
                year       = collect(1800:(1800+n-1)),
                population = Float64.(base_pop),
                elites     = Float64.(elites),
            )
            eoi = elite_overproduction_index(data)
            @test nrow(eoi) == n
            # EOI should be increasing (elite ratio grows faster than pop)
            @test eoi.eoi[end] > eoi.eoi[1]
        end
    end

end
