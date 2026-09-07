using Test
using LinearAlgebra
using Random
using QuantumNaturalfPEPS
using ITensors
using ITensorMPS

function _sampling_cache_hamiltonian(parameters, number_of_sites)
    hopping, pairing, chemical_potential = parameters
    particle_block = diagm(
        0 => fill(-chemical_potential, number_of_sites),
        1 => fill(-hopping, number_of_sites - 1),
        -1 => fill(-conj(hopping), number_of_sites - 1),
    )
    pairing_block = diagm(
        1 => fill(pairing, number_of_sites - 1),
        -1 => fill(-pairing, number_of_sites - 1),
    )
    return Hermitian([
        particle_block pairing_block
        pairing_block' -transpose(particle_block)
    ])
end

@testset "Projected parent-Slater Schur cache" begin
    Lx = Ly = 3
    number_of_sites = Lx * Ly
    hopping = fill(0.2 + 0.0im, Lx, Ly, 3)
    fields = zeros(Float64, Lx, Ly, 3)
    fields[:, :, 1] .= 10.0
    gaussian_state = triangular_aux_gaussian_state(
        Lx,
        Ly;
        hopping,
        fields,
        particle_number=number_of_sites,
    )
    projected_state = gutzwiller_project(gaussian_state)
    order = [1, 4, 7, 2, 5, 8, 3, 6, 9]
    cache = ProjectedGaussianSchurCache(projected_state; order)
    prefix = Dict{Int,Int}()
    prefix_probability = 1.0

    for site in order
        probabilities = projected_conditional_probabilities(cache)
        exact = map(0:1) do spin
            candidate = copy(prefix)
            candidate[site] = spin
            QuantumNaturalfPEPS.get_prob(projected_state, candidate) /
                prefix_probability
        end
        @test probabilities ≈ exact atol=1e-10
        spin = argmax(probabilities) - 1
        prefix[site] = spin
        prefix_probability = QuantumNaturalfPEPS.get_prob(projected_state, prefix)
        condition_projected_gaussian!(cache, spin)
    end

    fixed_state = gutzwiller_project(gaussian_state; Nup=5)
    fixed_cache = ProjectedGaussianSchurCache(fixed_state; order)
    for position in eachindex(order)
        probabilities = projected_conditional_probabilities(fixed_cache)
        spin = if fixed_cache.measured_up < fixed_cache.target_Nup
            0
        else
            1
        end
        condition_projected_gaussian!(fixed_cache, spin)
        if position == length(order)
            @test fixed_cache.measured_up == fixed_cache.target_Nup
        end
        @test all(>=(0), probabilities)
    end

    hilbert = siteinds("S=1/2", Lx, Ly)
    peps = PEPS(ComplexF64, hilbert; bond_dim=1, show_warning=false)
    write!(peps, fill(ComplexF64(inv(sqrt(2))), length(peps)))
    sample, _, _ = QuantumNaturalfPEPS.get_sample(peps; trial_state=fixed_state)
    @test count(==(0), sample) == fixed_state.Nup
end

@testset "Gaussian low-rank sampling caches" begin
    number_of_sites = 4
    state = QuantumNaturalfPEPS.GaussianState(
        _sampling_cache_hamiltonian,
        number_of_sites;
        η=[1.0 + 0.2im, 0.7 - 0.1im, 0.2],
        parity_sector=0,
    )

    @testset "direct Schur conditioning" begin
        order = [1, 3, 2, 4]
        cache = GaussianSchurCache(state; order)
        prefix = Dict{Int,Int}()
        prefix_probability = 1.0

        for site in order
            probabilities = gaussian_conditional_probabilities(cache)
            exact = ntuple(2) do occupation_index
                candidate = copy(prefix)
                candidate[site] = occupation_index - 1
                QuantumNaturalfPEPS.get_prob(state, candidate) / prefix_probability
            end
            @test collect(probabilities) ≈ collect(exact) atol=1e-11

            occupation = argmax(probabilities) - 1
            prefix[site] = occupation
            prefix_probability = QuantumNaturalfPEPS.get_prob(state, prefix)
            @test condition_gaussian!(cache, occupation) ≈ probabilities[occupation + 1]
        end

        @test isempty(cache.remaining_sites)
        @test exp(cache.log_probability) ≈ prefix_probability atol=1e-11
    end

    @testset "MCMC flip ratios and Woodbury updates" begin
        configurations = [digits(index, base=2, pad=number_of_sites)
                          for index in 0:(2^number_of_sites - 1)]
        probabilities = map(configuration ->
            QuantumNaturalfPEPS.get_prob(state, configuration), configurations)
        configuration = configurations[argmax(probabilities)]
        cache = GaussianOccupationCache(state, configuration)

        @test exp(cache.log_probability) ≈
            QuantumNaturalfPEPS.get_prob(state, configuration) atol=1e-11
        @test gaussian_flip_probability_ratio(cache, 1) == 0
        @test_throws ArgumentError accept_gaussian_flip!(cache, 1)

        accepted = 0
        for sites in ([1, 2], [1, 3], [2, 4])
            proposed = copy(cache.configuration)
            proposed[sites] .= 1 .- proposed[sites]
            exact_ratio = QuantumNaturalfPEPS.get_prob(state, proposed) /
                QuantumNaturalfPEPS.get_prob(state, cache.configuration)
            @test gaussian_flip_probability_ratio(cache, sites) ≈ exact_ratio atol=1e-10

            if exact_ratio > 1e-10
                accept_gaussian_flip!(cache, sites; rebuild_after=2)
                accepted += 1
                @test cache.configuration == proposed
                @test exp(cache.log_probability) ≈
                    QuantumNaturalfPEPS.get_prob(state, proposed) atol=1e-10
                @test cache.determinant_matrix * cache.inverse_matrix ≈
                    I atol=1e-10
            end
        end
        @test accepted >= 2
    end

    @testset "direct sampler defaults to Schur" begin
        Random.seed!(8723)
        sample, log_probability = QuantumNaturalfPEPS.get_sample(state)
        configuration = collect(vec(sample))
        @test exp(log_probability) ≈
            QuantumNaturalfPEPS.get_prob(state, configuration) atol=1e-10
        @test mod(sum(configuration), 2) == state.parity_sector
    end
end
