# Contraction and padding place ln last. Restore its original slot so that
# enlargement does not change the virtual-index order used by vec and write!.
function enlarged_index_order(o::ITensor, l, ln)
    return map(i -> i == l ? ln : i, inds(o))
end

function increase_bond_dim(o1::ITensor, l, ln; value=0)
    ol = noncommoninds(o1, l)
    perm = NDTensors.getperm((ol..., l), inds(o1))
    o1t = permutedims(o1.tensor, perm)
    o1t = reshape(o1t.storage, :, dim(l))

    o1n = randomITensor(eltype(o1), ol..., ln) .* (std(o1.tensor) * value)
    o1nt = reshape(o1n.tensor.storage, :, dim(ln))
    o1nt[:, 1:dim(l)] .= o1t
    return permute(o1n, enlarged_index_order(o1, l, ln)...)
end

function increase_bond_dim(o1::ITensor, o2::ITensor, new_bond_dim::Int; value=0, svalue=0)
    l = commonind(o1, o2)
    @assert dim(l) <= new_bond_dim
    ln = Index(new_bond_dim; tags=tags(l))
    ElT = eltype(o1)
    
    local U
    if value == 0 && svalue == 0
        t = NDTensors.random_unitary(ElT, dim(l), dim(ln))
        U = ITensor(t, l, ln)
        # The two embeddings must be conjugates to preserve the contraction
        # for complex tensors as well as real tensors.
        return permute(o1 * U, enlarged_index_order(o1, l, ln)...),
               permute(o2 * dag(U), enlarged_index_order(o2, l, ln)...)
    end
    if value == 0
        value = 1.
    end
    o1n = increase_bond_dim(o1, l, ln; value=value)
    o2n = increase_bond_dim(o2, l, ln; value=svalue/value)

    #@show norm(o1n*o2n - o1*o2)
    return o1n, o2n
end

"""
    increase_bond_dim!(peps::AbstractPEPS, new_bond_dim::Int; value=0, svalue=0)

Increase each virtual bond while preserving its position in each tensor's index
order. With value = svalue = 0, conjugate isometries preserve the contracted PEPS.
Otherwise the added entries are randomized; svalue = 0 leaves their contracted
contribution zero, while nonzero svalue can change the state.
"""
function increase_bond_dim!(peps::AbstractPEPS, new_bond_dim::Int; value=0, svalue=0)

    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        if i < size(peps, 1)
            peps[i, j], peps[i+1, j] = increase_bond_dim(peps[i, j], peps[i+1,j], new_bond_dim; value, svalue)
        end
        if j < size(peps, 2)
            peps[i, j], peps[i, j+1] = increase_bond_dim(peps[i, j], peps[i,j+1], new_bond_dim; value, svalue)
        end
    end
    peps.bond_dim = new_bond_dim
    peps.double_layer_envs = nothing
    return peps
end

increase_bond_dim(peps::AbstractPEPS, new_bond_dim::Int; kwargs...) = increase_bond_dim!(deepcopy(peps), new_bond_dim; kwargs...)
