#= Auxiliary functions for constructing triangular-lattice mean-field ansätze =#

function _check_three_sublattice_torus(Lx::Integer, Ly::Integer, shear::Integer)
    mod(2Lx, 3) == 0 || throw(ArgumentError(
        "three-sublattice order requires 2Lx divisible by 3, got Lx=$Lx",
    ))
    mod(2shear - Ly, 3) == 0 || throw(ArgumentError(
        "three-sublattice order requires 2shear-Ly divisible by 3, " *
        "got Ly=$Ly and shear=$shear",
    ))
    return nothing
end

function _check_stripe_torus(Lx::Integer, Ly::Integer, shear::Integer)
    iseven(Lx) || throw(ArgumentError(
        "the selected canted-stripe orientation requires even Lx, got $Lx",
    ))
    iseven(shear - Ly) || throw(ArgumentError(
        "the selected canted-stripe orientation requires even shear-Ly, " *
        "got Ly=$Ly and shear=$shear",
    ))
    return nothing
end

_triangular_k_index(x::Integer, y::Integer) = mod(2(x - 1) - (y - 1), 3)
_triangular_k_angle(x::Integer, y::Integer) = 2pi * _triangular_k_index(x, y) / 3
_triangular_sublattice(x::Integer, y::Integer) = _triangular_k_index(x, y) + 1
_triangular_stripe_sign(x::Integer, y::Integer) =
    iseven((x - 1) - (y - 1)) ? 1.0 : -1.0

function _real_ansatz_parameters(
    η::AbstractVector,
    expected_length::Integer,
    state_name::AbstractString,
)
    length(η) == expected_length || throw(DimensionMismatch(
        "$state_name requires $expected_length mean-field parameters, " *
        "got $(length(η))",
    ))
    all(isreal, η) || throw(ArgumentError(
        "$state_name mean-field parameters must be real",
    ))
    all(isfinite, η) || throw(ArgumentError(
        "$state_name mean-field parameters must be finite",
    ))
    return Float64.(real.(η))
end

"""
    staggered_pi_flux_hoppings(Lx, Ly; amplitude=1.0)

Construct an `Lx × Ly × 3` hopping array for the triangular-lattice Dirac
spin-liquid ansatz with staggered `[0, π]` flux through the two triangles of
every square-grid plaquette. The direction index follows
`1 => (1, 0)`, `2 => (0, 1)`, and `3 => (1, 1)`.

The gauge choice is

```text
t₁(x,y) = (-1)^y amplitude,
t₂(x,y) =        amplitude,
t₃(x,y) = (-1)^y amplitude.
```

It gives zero flux around the oriented triangle
`(x,y) → (x+1,y) → (x+1,y+1) → (x,y)` and `π` flux around
`(x,y) → (x+1,y+1) → (x,y+1) → (x,y)`. Consequently, every
primitive rhombus has `π` flux. This gauge doubles the unit cell along `y`,
so `Ly` must be even on a torus.
"""
function staggered_pi_flux_hoppings(
    Lx::Integer,
    Ly::Integer;
    amplitude::Real=1.0,
)
    Lx >= 3 || throw(ArgumentError("Lx must be at least 3 to avoid duplicate torus bonds"))
    Ly >= 3 || throw(ArgumentError("Ly must be at least 3 to avoid duplicate torus bonds"))
    iseven(Ly) || throw(ArgumentError(
        "Ly must be even for the chosen staggered-[0, π] torus gauge",
    ))
    amplitude > 0 || throw(ArgumentError("amplitude must be positive"))

    hopping = fill(ComplexF64(amplitude), Lx, Ly, 3)
    for y in 1:Ly
        staggered_sign = isodd(y) ? -1.0 : 1.0
        hopping[:, y, 1] .*= staggered_sign
        hopping[:, y, 3] .*= staggered_sign
    end

    return hopping
end

"""
    uniform_flux_staggered_pi_hoppings(Lx, Ly, Q; amplitude=1.0)

Add `Q` quantized units of uniform U(1) flux to the staggered-`[0, pi]`
nearest-neighbor hopping ansatz on an `Lx x Ly` torus. The extra flux through
each primitive rhombus is

```math
phi = 2 pi Q/(L_x L_y),
```

and each elementary triangle carries `phi/2` in addition to its staggered
zero- or `pi`-flux background. Magnetic boundary phases are included in the
returned `Lx x Ly x 3` hopping array. `Q` is understood modulo `Lx*Ly`.

The direction index is the same as for [`staggered_pi_flux_hoppings`](@ref):
`1 => (1,0)`, `2 => (0,1)`, and `3 => (1,1)`.

Returns:
- hopping_Q: `Lx x Ly x 3` array of complex hoppings given the staggered-`[0, pi]`
             background with the uniform flux added.
"""
function uniform_flux_staggered_pi_hoppings(
    Lx::Integer,
    Ly::Integer,
    Q::Integer;
    amplitude::Real=1.0,
)
    N = Lx * Ly
    additional_flux_per_rhombus = 2pi * mod(Q, N) / N
    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude)

    # Landau gauge with the compensating y-boundary phase required on a torus.
    for y in 1:Ly, x in 1:Lx
        x0, y0 = x - 1, y - 1
        phases = if y < Ly
            (
                -additional_flux_per_rhombus * y0,
                0.0,
                -additional_flux_per_rhombus * (y0 + 0.5),
            )
        else
            y_boundary_phase = additional_flux_per_rhombus * Ly * x0
            (
                -additional_flux_per_rhombus * y0,
                y_boundary_phase,
                y_boundary_phase + additional_flux_per_rhombus / 2,
            )
        end
        hopping[x, y, :] .*= exp.(im .* phases)
    end

    return hopping, additional_flux_per_rhombus
end

"""
    y_hopping_fields(Lx, Ly, η; hopping_amplitude=1, shear=0)

Construct the hopping and fictitious-field arrays for the triangular-lattice
Y ansatz. The reduced variational vector follows the paper's ordering
`η = [h1, h2, h3, Delta]`. Sublattices are selected by
`mod(2(x-1)-(y-1), 3)` and carry fields

```text
A: (  0, 0,  h1)
B: ( h2, 0, -h3)
C: (-h2, 0, -h3).
```

The `Delta`-magnitude bonds connect the B and C sublattices. All other bonds
have magnitude `hopping_amplitude`; their signs retain the staggered `[0, pi]`
Dirac flux pattern.
"""
function y_hopping_fields(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector;
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
)
    _check_three_sublattice_torus(Lx, Ly, shear)
    h1, h2, h3, delta = _real_ansatz_parameters(η, 4, "the Y ansatz")
    delta > 0 || throw(ArgumentError("the Y hopping magnitude Delta must be positive"))

    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude=hopping_amplitude)
    for bond in triangular_torus_bonds(Lx, Ly; shear)
        source_sublattice = _triangular_sublattice(bond.source...)
        target_sublattice = _triangular_sublattice(bond.target...)
        if (source_sublattice == 2 && target_sublattice == 3) ||
           (source_sublattice == 3 && target_sublattice == 2)
            x, y = bond.source
            hopping[x, y, bond.direction] *= delta / hopping_amplitude
        end
    end

    fields = zeros(Float64, Lx, Ly, 3)
    for y in 1:Ly, x in 1:Lx
        sublattice = _triangular_sublattice(x, y)
        fields[x, y, :] .= if sublattice == 1
            (0.0, 0.0, h1)
        elseif sublattice == 2
            (h2, 0.0, -h3)
        else
            (-h2, 0.0, -h3)
        end
    end
    return hopping, fields
end

"""
    umbrella_hopping_fields(Lx, Ly, η; hopping_amplitude=1, shear=0)

Construct the hopping and fields for the triangular-lattice umbrella ansatz,
whose only optimized mean-field parameter is `η = [h]`. The hopping magnitudes
are uniform and retain the staggered `[0, pi]` flux. The field is
`M_i = h(cos(K*R_i), sin(K*R_i), 0)` with
`K*R_i = 2pi(2(x-1)-(y-1))/3`.
"""
function umbrella_hopping_fields(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector;
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
)
    _check_three_sublattice_torus(Lx, Ly, shear)
    h = only(_real_ansatz_parameters(η, 1, "the umbrella ansatz"))
    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude=hopping_amplitude)
    fields = zeros(Float64, Lx, Ly, 3)
    for y in 1:Ly, x in 1:Lx
        angle = _triangular_k_angle(x, y)
        fields[x, y, :] .= (h * cos(angle), h * sin(angle), 0.0)
    end
    return hopping, fields
end

"""
    cs_hopping_fields(
        Lx, Ly, η; hopping_amplitude=1, stripe_delta=0.8, shear=0
    )

Construct one of the three symmetry-related canted-stripe ansatze. The paper
optimizes only `η = [h]`; `stripe_delta` is therefore a fixed hopping magnitude.
The selected orientation is invariant along `a2 = (1, 1)` and alternates along
`a1 = (1, 0)`, with `M_i = ((-1)^(x-y) h, 0, 0)`. Direction-3 bonds have
magnitude `stripe_delta`, and the remaining bonds have magnitude
`hopping_amplitude`, all with the staggered `[0, pi]` signs.
"""
function cs_hopping_fields(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector;
    hopping_amplitude::Real=1.0,
    stripe_delta::Real=0.8,
    shear::Integer=0,
)
    _check_stripe_torus(Lx, Ly, shear)
    h = only(_real_ansatz_parameters(η, 1, "the canted-stripe ansatz"))
    stripe_delta > 0 || throw(ArgumentError(
        "the fixed canted-stripe hopping magnitude must be positive",
    ))
    hopping = staggered_pi_flux_hoppings(Lx, Ly; amplitude=hopping_amplitude)
    hopping[:, :, 3] .*= stripe_delta / hopping_amplitude
    fields = zeros(Float64, Lx, Ly, 3)
    for y in 1:Ly, x in 1:Lx
        fields[x, y, :] .= (h * _triangular_stripe_sign(x, y), 0.0, 0.0)
    end
    return hopping, fields
end

#= triangular-lattice mean-field ansätze for hamiltonian_J1J2_H =#

function _triangular_parton_state(
    Lx::Integer,
    Ly::Integer,
    η::AbstractVector,
    hopping::AbstractArray,
    fields::AbstractArray,
    expand_parameters::Function;
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
    hopping_parameterization::Symbol=:fixed_phase,
)
    number_of_sites = Lx * Ly
    number_of_modes = 2number_of_sites

    0 <= particle_number <= number_of_modes || throw(ArgumentError(
        "particle_number must lie between 0 and $number_of_modes, " *
        "got $particle_number",
    ))
    gap_tolerance >= 0 || throw(ArgumentError(
        "gap_tolerance must be nonnegative, got $gap_tolerance",
    ))

    initial_Haux = hamiltonian_aux_triangular_torus(
        Lx,
        Ly;
        hopping,
        fields,
        shear,
    )
    chemical_potential = _auxiliary_chemical_potential(
        initial_Haux,
        particle_number,
        gap_tolerance,
    )

    hopping_phases = map(hopping) do value
        iszero(value) ? one(value) : value / abs(value)
    end

    H_BdG_func = _triangular_aux_bdg_function(
        Lx,
        Ly,
        triangular_torus_bonds(Lx, Ly; shear),
        hopping_phases,
        chemical_potential,
        length(η),
        expand_parameters;
        hopping_parameterization,
    )

    return GaussianState(
        H_BdG_func,
        number_of_modes;
        η=copy(η),
        parity_sector=mod(particle_number, 2),
        target_state=0,
        cache_gradients,
    )
end

function monopole_state(
    Lx::Integer,
    Ly::Integer,
    Q::Integer;
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)
    hopping, _ = uniform_flux_staggered_pi_hoppings(
        Lx,
        Ly,
        Q;
        amplitude=hopping_amplitude,
    )
    fields = zeros(Float64, Lx, Ly, 3)

    number_of_sites = Lx * Ly
    fixed_parameters = zeros(Float64, 6number_of_sites)

    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x

        for direction in 1:3
            fixed_parameters[3(site - 1) + direction] =
                abs(hopping[x, y, direction])
        end

        field_offset = 3number_of_sites + 3(site - 1)
        fixed_parameters[field_offset+1:field_offset+3] .=
            fields[x, y, :]
    end

    expand_parameters = _ -> fixed_parameters

    return _triangular_parton_state(
        Lx,
        Ly,
        Float64[],
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end

"""
    y_state(Lx, Ly; η, kwargs...)

Construct the full unprojected Y Gaussian state with the four optimized
parameters `η = [h1, h2, h3, Delta]`.
"""
function y_state(
    Lx::Integer,
    Ly::Integer;
    η::AbstractVector=[0.35, 0.25, 0.15, 0.8],
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)
    hopping, fields = y_hopping_fields(
        Lx,
        Ly,
        η;
        hopping_amplitude,
        shear,
    )
    bonds = triangular_torus_bonds(Lx, Ly; shear)
    delta_bond = falses(Lx, Ly, 3)
    for bond in bonds
        source_sublattice = _triangular_sublattice(bond.source...)
        target_sublattice = _triangular_sublattice(bond.target...)
        if (source_sublattice == 2 && target_sublattice == 3) ||
           (source_sublattice == 3 && target_sublattice == 2)
            x, y = bond.source
            delta_bond[x, y, bond.direction] = true
        end
    end
    parameter_offset = zeros(Float64, 6Lx * Ly)
    parameter_map = zeros(Float64, 6Lx * Ly, 4)
    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x
        for direction in 1:3
            hopping_index = 3(site - 1) + direction
            if delta_bond[x, y, direction]
                parameter_map[hopping_index, 4] = 1.0
            else
                parameter_offset[hopping_index] = hopping_amplitude
            end
        end
        field_offset = 3Lx * Ly + 3(site - 1)
        sublattice = _triangular_sublattice(x, y)
        if sublattice == 1
            parameter_map[field_offset + 3, 1] = 1.0
        elseif sublattice == 2
            parameter_map[field_offset + 1, 2] = 1.0
            parameter_map[field_offset + 3, 3] = -1.0
        else
            parameter_map[field_offset + 1, 2] = -1.0
            parameter_map[field_offset + 3, 3] = -1.0
        end
    end
    expand_parameters = parameters -> parameter_offset + parameter_map * parameters
    return _triangular_parton_state(
        Lx,
        Ly,
        η,
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end


"""
    umbrella_state(Lx, Ly; η, kwargs...)

Construct the full unprojected umbrella Gaussian state with its single
optimized parameter `η = [h]`.
"""
function umbrella_state(
    Lx::Integer,
    Ly::Integer;
    η::AbstractVector=[0.3],
    hopping_amplitude::Real=1.0,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)

    hopping, fields = umbrella_hopping_fields(
        Lx,
        Ly,
        η;
        hopping_amplitude,
        shear,
    )
    parameter_offset = zeros(Float64, 6Lx * Ly)
    parameter_offset[1:(3Lx * Ly)] .= hopping_amplitude
    parameter_map = zeros(Float64, 6Lx * Ly, 1)
    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x
        field_offset = 3Lx * Ly + 3(site - 1)
        angle = _triangular_k_angle(x, y)
        parameter_map[field_offset + 1, 1] = cos(angle)
        parameter_map[field_offset + 2, 1] = sin(angle)
    end
    expand_parameters = parameters -> parameter_offset + parameter_map * parameters
    return _triangular_parton_state(
        Lx,
        Ly,
        η,
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end

"""
    cs_state(Lx, Ly; η, kwargs...)

Construct the full unprojected canted-stripe Gaussian state with the paper's
single optimized parameter `η = [h]`. The hopping ratio is supplied separately
through the fixed keyword `stripe_delta`.
"""
function cs_state(
    Lx::Integer,
    Ly::Integer;
    η::AbstractVector=[0.3],
    hopping_amplitude::Real=1.0,
    stripe_delta::Real=0.8,
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
)
    hopping, fields = cs_hopping_fields(
        Lx,
        Ly,
        η;
        hopping_amplitude,
        stripe_delta,
        shear,
    )
    parameter_offset = zeros(Float64, 6Lx * Ly)
    parameter_map = zeros(Float64, 6Lx * Ly, 1)
    for y in 1:Ly, x in 1:Lx
        site = (y - 1) * Lx + x
        for direction in 1:3
            hopping_index = 3(site - 1) + direction
            parameter_offset[hopping_index] =
                direction == 3 ? stripe_delta : hopping_amplitude
        end
        field_offset = 3Lx * Ly + 3(site - 1)
        parameter_map[field_offset + 1, 1] = _triangular_stripe_sign(x, y)
    end
    expand_parameters = parameters -> parameter_offset + parameter_map * parameters
    return _triangular_parton_state(
        Lx,
        Ly,
        η,
        hopping,
        fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
    )
end


"""
    triangular_spanning_tree(Lx, Ly; shear=0)

Return deterministic nearest-neighbor bond indices for a breadth-first
spanning tree rooted at site (1, 1). Indices refer to triangular_torus_bonds.
"""
function triangular_spanning_tree(Lx::Integer, Ly::Integer; shear::Integer=0)
    bonds = triangular_torus_bonds(Lx, Ly; shear)
    number_of_sites = Lx * Ly
    adjacency = [Tuple{Int,Int}[] for _ in 1:number_of_sites]
    for (bond_index, bond) in enumerate(bonds)
        source = (bond.source[2] - 1) * Lx + bond.source[1]
        target = (bond.target[2] - 1) * Lx + bond.target[1]
        push!(adjacency[source], (target, bond_index))
        push!(adjacency[target], (source, bond_index))
    end

    visited = falses(number_of_sites)
    visited[1] = true
    queue = [1]
    tree_bonds = Int[]
    next_site = 1
    while next_site <= length(queue)
        site = queue[next_site]
        next_site += 1
        for (neighbor, bond_index) in adjacency[site]
            visited[neighbor] && continue
            visited[neighbor] = true
            push!(queue, neighbor)
            push!(tree_bonds, bond_index)
        end
    end
    all(visited) || error("the triangular torus bond graph is disconnected")
    length(tree_bonds) == number_of_sites - 1 || error(
        "internal error while constructing the triangular spanning tree",
    )
    return tree_bonds
end

"""
    canonicalize_spanning_tree_phases(hopping, Lx, Ly; kwargs...)

Apply a site-local U(1) transformation that makes every selected spanning-tree
hopping real and positive. Return (canonical_hopping, site_gauges, tree_bonds).
The convention is t_ij -> conj(g_i) * t_ij * g_j. A zero tree hopping is a
boundary of this gauge chart, so a different tree must be chosen in that case.
"""
function canonicalize_spanning_tree_phases(
    hopping::AbstractArray,
    Lx::Integer,
    Ly::Integer;
    shear::Integer=0,
    tree_bonds::AbstractVector{<:Integer}=triangular_spanning_tree(
        Lx,
        Ly;
        shear,
    ),
    tolerance::Real=1e-12,
)
    size(hopping) == (Lx, Ly, 3) || throw(DimensionMismatch(
        "hopping must have size ($Lx, $Ly, 3), got $(size(hopping))",
    ))
    tolerance >= 0 || throw(ArgumentError("tolerance must be nonnegative"))

    bonds = triangular_torus_bonds(Lx, Ly; shear)
    number_of_sites = Lx * Ly
    length(tree_bonds) == number_of_sites - 1 || throw(ArgumentError(
        "tree_bonds must contain $(number_of_sites - 1) bonds",
    ))
    length(unique(tree_bonds)) == length(tree_bonds) || throw(ArgumentError(
        "tree_bonds must not contain duplicates",
    ))
    all(index -> index in eachindex(bonds), tree_bonds) ||
        throw(BoundsError(bonds, tree_bonds))

    adjacency = [Tuple{Int,Int,Bool}[] for _ in 1:number_of_sites]
    for bond_index in tree_bonds
        bond = bonds[bond_index]
        source = (bond.source[2] - 1) * Lx + bond.source[1]
        target = (bond.target[2] - 1) * Lx + bond.target[1]
        push!(adjacency[source], (target, bond_index, true))
        push!(adjacency[target], (source, bond_index, false))
    end

    phases = zeros(Float64, number_of_sites)
    visited = falses(number_of_sites)
    visited[1] = true
    queue = [1]
    next_site = 1
    while next_site <= length(queue)
        parent = queue[next_site]
        next_site += 1
        for (child, bond_index, follows_orientation) in adjacency[parent]
            visited[child] && continue
            bond = bonds[bond_index]
            x, y = bond.source
            value = hopping[x, y, bond.direction]
            abs(value) > tolerance || throw(ArgumentError(
                "spanning-tree bond $bond_index has vanishing hopping; " *
                "choose a tree that avoids zero hoppings",
            ))
            phases[child] = phases[parent] +
                (follows_orientation ? -angle(value) : angle(value))
            visited[child] = true
            push!(queue, child)
        end
    end
    all(visited) || throw(ArgumentError("tree_bonds do not span the lattice"))

    gauges = cis.(phases)
    canonical = ComplexF64.(hopping)
    tree_set = Set(tree_bonds)
    for (bond_index, bond) in enumerate(bonds)
        source = (bond.source[2] - 1) * Lx + bond.source[1]
        target = (bond.target[2] - 1) * Lx + bond.target[1]
        x, y = bond.source
        value = conj(gauges[source]) *
            canonical[x, y, bond.direction] * gauges[target]
        canonical[x, y, bond.direction] = bond_index in tree_set ?
            ComplexF64(abs(value), 0) : value
    end
    return canonical, reshape(gauges, Lx, Ly), collect(Int, tree_bonds)
end

"""
    free_state(Lx, Ly; hopping_parameterization=:cartesian, kwargs...)

Construct a trainable, unprojected, half-filled triangular-lattice Gaussian
state. The legacy cartesian parameterization contains real and imaginary parts
of all hoppings plus three real fields per site (9LxLy parameters) and retains
local U(1) redundancy.

The tree_loop parameterization first canonicalizes a spanning tree. It retains
one real component for every bond, an imaginary component only for each
non-tree chord, and three fields per site (8LxLy+1 parameters). Tree hoppings
remain real. Chord arguments are the independent fundamental-loop phases, so
all continuous local U(1) gauge directions are removed while the Hamiltonian
remains affine in the optimization coordinates.
"""
function free_state(
    Lx::Integer,
    Ly::Integer;
    η::Union{Nothing,AbstractVector}=nothing,
    hopping=1.0,
    fields=nothing,
    randomize_hopping_phases::Bool=false,
    rng::AbstractRNG=Random.default_rng(),
    shear::Integer=0,
    particle_number::Integer=Lx * Ly,
    gap_tolerance::Real=1e-8,
    cache_gradients::Bool=false,
    hopping_parameterization::Symbol=:cartesian,
)
    number_of_sites = Lx * Ly
    number_of_hoppings = 3number_of_sites
    hopping_parameterization in (:cartesian, :tree_loop) || throw(ArgumentError(
        "hopping_parameterization must be :cartesian or :tree_loop, got " *
        "$hopping_parameterization",
    ))

    bonds = triangular_torus_bonds(Lx, Ly; shear)
    tree_bonds = triangular_spanning_tree(Lx, Ly; shear)
    tree_set = Set(tree_bonds)
    chord_bonds = [index for index in eachindex(bonds) if !(index in tree_set)]
    number_of_parameters = hopping_parameterization === :cartesian ?
        9number_of_sites : 6number_of_sites + length(chord_bonds)

    initial_hopping = zeros(ComplexF64, Lx, Ly, 3)
    initial_fields = zeros(Float64, Lx, Ly, 3)
    parameters = if isnothing(η)
        hopping_values = zeros(ComplexF64, Lx, Ly, 3)
        for bond in bonds
            x, y = bond.source
            value = ComplexF64(_aux_triangular_hopping(
                hopping,
                bond.source,
                bond.unwrapped_target,
                bond.direction,
            ))
            isfinite(value) || throw(ArgumentError(
                "the initial hoppings must be finite",
            ))
            if randomize_hopping_phases && !iszero(value)
                value *= cis(2π * rand(rng))
            end
            hopping_values[x, y, bond.direction] = value
        end

        field_values = zeros(Float64, Lx, Ly, 3)
        for y in 1:Ly, x in 1:Lx
            field = _aux_triangular_field(fields, (x, y))
            all(isreal, field) || throw(ArgumentError(
                "the initial fictitious fields must be real",
            ))
            all(isfinite, field) || throw(ArgumentError(
                "the initial fictitious fields must be finite",
            ))
            field_values[x, y, :] .= real.(field)
        end

        if hopping_parameterization === :tree_loop
            hopping_values, _, _ = canonicalize_spanning_tree_phases(
                hopping_values,
                Lx,
                Ly;
                shear,
                tree_bonds,
            )
        end
        initial_hopping .= hopping_values
        initial_fields .= field_values

        initial_parameters = Vector{Float64}(undef, number_of_parameters)
        for (bond_index, bond) in enumerate(bonds)
            x, y = bond.source
            value = hopping_values[x, y, bond.direction]
            initial_parameters[bond_index] = real(value)
            if hopping_parameterization === :cartesian
                initial_parameters[number_of_hoppings + bond_index] = imag(value)
            end
        end
        if hopping_parameterization === :tree_loop
            for (chord_index, bond_index) in enumerate(chord_bonds)
                bond = bonds[bond_index]
                x, y = bond.source
                initial_parameters[number_of_hoppings + chord_index] =
                    imag(hopping_values[x, y, bond.direction])
            end
        end
        field_parameter_offset = hopping_parameterization === :cartesian ?
            2number_of_hoppings : number_of_hoppings + length(chord_bonds)
        for y in 1:Ly, x in 1:Lx
            site = (y - 1) * Lx + x
            parameter_index = field_parameter_offset + 3(site - 1)
            initial_parameters[parameter_index+1:parameter_index+3] .=
                field_values[x, y, :]
        end
        initial_parameters
    else
        randomize_hopping_phases && throw(ArgumentError(
            "randomize_hopping_phases cannot be used when η is supplied",
        ))
        _real_ansatz_parameters(η, number_of_parameters, "the free ansatz")
    end

    if !isnothing(η)
        chord_positions = Dict(
            bond => index for (index, bond) in enumerate(chord_bonds)
        )
        for (bond_index, bond) in enumerate(bonds)
            x, y = bond.source
            imaginary_part = if hopping_parameterization === :cartesian
                parameters[number_of_hoppings + bond_index]
            else
                chord_index = get(chord_positions, bond_index, 0)
                iszero(chord_index) ? 0.0 :
                    parameters[number_of_hoppings + chord_index]
            end
            initial_hopping[x, y,bond.direction] =
                parameters[bond_index] + im * imaginary_part
        end
        field_parameter_offset = hopping_parameterization === :cartesian ?
            2number_of_hoppings : number_of_hoppings + length(chord_bonds)
        for y in 1:Ly, x in 1:Lx
            site = (y - 1) * Lx + x
            parameter_index = field_parameter_offset + 3(site - 1)
            initial_fields[x, y, :] .=
                parameters[parameter_index+1:parameter_index+3]
        end
    end

    expand_parameters = if hopping_parameterization === :cartesian
        identity
    else
        function (reduced_parameters)
            full_parameters = zeros(
                eltype(reduced_parameters),
                9number_of_sites,
            )
            full_parameters[1:number_of_hoppings] .=
                reduced_parameters[1:number_of_hoppings]
            for (chord_index, bond_index) in enumerate(chord_bonds)
                full_parameters[number_of_hoppings + bond_index] =
                    reduced_parameters[number_of_hoppings + chord_index]
            end
            reduced_field_offset = number_of_hoppings + length(chord_bonds)
            full_parameters[2number_of_hoppings+1:end] .=
                reduced_parameters[reduced_field_offset+1:end]
            return full_parameters
        end
    end

    return _triangular_parton_state(
        Lx,
        Ly,
        parameters,
        initial_hopping,
        initial_fields,
        expand_parameters;
        shear,
        particle_number,
        gap_tolerance,
        cache_gradients,
        hopping_parameterization=:cartesian,
    )
end
