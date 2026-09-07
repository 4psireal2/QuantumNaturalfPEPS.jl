using Test
using LinearAlgebra
using QuantumNaturalfPEPS

@testset "truncation of null Vbar columns at the noise floor" begin
        # Regression: for a *near*-Slater trial state — an optimized η whose pairing
        # components have decayed to ~1e-9 but not to zero — half the columns of Vbar
        # carry only leaked pairing at ~1e-10, exactly where an *absolute* 1e-10 cutoff
        # stops recognising them as zero. (An exactly pairing-free η, e.g. a pure π-flux
        # hopping state, gives clean 0.0 columns and never triggers this.) A null
        # column then survives, Q_mat comes out one dimension too large, and pfaffian()
        # returns 0 for *every* configuration: logψ = -Inf everywhere, so the importance
        # weights come out NaN and the optimisation dies with
        # "ArgumentError: weights cannot contain Inf or NaN values".
        # truncated_bloch_messiah only slices blocks, so synthetic input suffices here:
        # n_pair paired modes (v_p ~ 0.5) followed by n_null columns of pure noise.
        function synthetic_bloch_messiah(n_pair, n_null, noise)
            n = n_pair + n_null
            Vbar = zeros(ComplexF64, n, n)
            for p in 1:2:n_pair
                v = 0.3 + 0.05p # deterministic stand-in for a Bloch-Messiah singular value
                Vbar[p, p+1] = im * v
                Vbar[p+1, p] = -im * v
            end
            for c in n_pair+1:n, r in 1:n
                Vbar[r, c] = noise * (1 + 0.1r) # deterministic stand-in for eigensolver noise
            end
            Ubar = Matrix{ComplexF64}(I, n, n)
            D = Matrix{ComplexF64}(I, n, n)
            C = Matrix{ComplexF64}(I, n, n)
            return ([D zeros(size(D)); zeros(size(D)) conj.(D)],
                    [Ubar Vbar; Vbar Ubar],
                    [C zeros(size(C)); zeros(size(C)) conj.(C)])
        end

        # The null columns must be truncated at every noise level, not just far below 1e-10.
        n_pair = 8
        n_null = 8
        for noise in (1e-16, 1e-13, 1e-11, 1e-10, 5e-10)
            Dmat, UVmat, Cmat = synthetic_bloch_messiah(n_pair, n_null, noise)
            _, UVmat_prime, _ = QuantumNaturalfPEPS.truncated_bloch_messiah(Dmat, UVmat, Cmat)
            @test size(UVmat_prime, 1) ÷ 2 == n_pair # the null columns are truncated away
        end

        # A Vbar that is nothing but noise carries no paired modes at all and must still be
        # truncated away completely.
        Dmat, UVmat, Cmat = synthetic_bloch_messiah(0, n_null, 1e-13)
        _, UVmat_prime, _ = QuantumNaturalfPEPS.truncated_bloch_messiah(Dmat, UVmat, Cmat)
        @test size(UVmat_prime, 1) ÷ 2 == 0 # no paired modes in this test
    end

@testset "truncate null modes of a physical pi-flux Slater state" begin
    Lx = Ly = 4
    N = Lx * Ly

    # pi-flux hopping matrix on an Lx x Ly open lattice, site (i, j) -> i + (j-1) * Lx
    function build_T_pi_flux(t, Lx, Ly)
        N = Lx * Ly
        hopping_x = [(a % Lx == 0) ? 0.0 : -t for a in 1:N-1]   # every Lx-th link crosses the boundary
        hopping_y = [-t * (-1)^mod1(a, Lx) for a in 1:N-Lx]     # staggered sign -> pi flux per plaquette

        T = diagm(1 => hopping_x, -1 => conj.(hopping_x), Lx => hopping_y, -Lx => conj.(hopping_y))
        return T
    end

    T = build_T_pi_flux(1.0, Lx, Ly)

    # This is the BdG representation of H = sum_ab T_ab c†_a c_b.
    pairing = zeros(ComplexF64, N, N)
    H_BdG = Hermitian([T pairing; pairing -transpose(T)])

    @test T ≈ T'
    @test iszero(H_BdG[1:N, N+1:2N])

    _, M = QuantumNaturalfPEPS.bogoliubov(H_BdG) # M diagonalizes H_BdG
    Dmat, UVmat, Cmat =
        QuantumNaturalfPEPS.bloch_messiah_decomposition(M)
    _, _, Vbar, _ =
        QuantumNaturalfPEPS.get_mats_from_bloch_messiah(Dmat, UVmat, Cmat)

    # At half filling Vbar has rank N/2: eight active occupied modes followed
    # by eight inactive modes. The truncation must remove the inactive half.
    @test rank(Vbar) == N ÷ 2
    @test all(maximum.(abs, eachcol(Vbar)[1:N÷2]) .> 1e-2) # the active half is well above the noise floor
    @test all(maximum.(abs, eachcol(Vbar)[N÷2+1:end]) .≈ 0.0) # the inactive half is zero as we dont have pairing

    Dmat_trunc, UVmat_trunc, Cmat_trunc =
        QuantumNaturalfPEPS.truncated_bloch_messiah(Dmat, UVmat, Cmat)

    @test size(Dmat_trunc) == (2N, N)
    @test size(Cmat_trunc) == (N, 2N)
    @test rank(UVmat_trunc[1:(N ÷ 2), 1:(N ÷ 2)]) == 0 # the truncated Vbar is empty

    Vbar_trunc = UVmat_trunc[(N ÷ 2 + 1):end, 1:(N ÷ 2)]
    @test Vbar_trunc' * Vbar_trunc ≈ I
end

