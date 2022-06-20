function MLUtils.getobs!(buffer::AbstractArray, A::AbstractArray)
    Base.setindex_shape_check(buffer, size(A)...)
    copyto!(buffer, A)
    return buffer
end

function MLUtils.getobs!(buffer::AbstractArray, A::AbstractArray{<:Any, N}, idx) where N
    I = ntuple(_ -> :, N-1)
    src = view(A, I..., idx)
    copyto!(buffer, src)
    return buffer
end

# ------------------------------------------------------------------------------------------
# ArrayDataset
# ------------------------------------------------------------------------------------------
abstract type LabeledDataset end

function Base.show(io::IO, d::D) where D <: LabeledDataset
    print(io, "$(D.name.name) with $(numobs(d)) observations")
end

struct ArrayDataset{F<:AbstractArray,T<:AbstractMatrix} <: LabeledDataset
    features::F
    targets::T
end

function init_buffer(d::ArrayDataset, n)
    return (
        features = zeros(Float32, size(d.features)[1:end-1]..., n),
        targets = similar(d.targets, 1, n),
    )
end

MLUtils.numobs(d::ArrayDataset) = numobs(d.features)

function MLUtils.getobs(d::ArrayDataset)
    return (features = d.features, targets = d.targets)
end

function MLUtils.getobs(d::ArrayDataset, inds)
    return (features = obsview(d.features, inds), targets = obsview(d.targets, inds))
end

function MLUtils.getobs!(buffer, d::ArrayDataset)
    getobs!(buffer.features, d.features)
    getobs!(buffer.targets, d.targets)
    return buffer
end

function MLUtils.getobs!(buffer, d::ArrayDataset, inds)
    getobs!(buffer.features, d.features, inds)
    getobs!(buffer.targets, d.targets, inds)
    return buffer
end

# ------------------------------------------------------------------------------------------
# FileDataset
# ------------------------------------------------------------------------------------------
struct FileDataset{F,T<:AbstractMatrix, N} <: LabeledDataset
    loadfn::F
    shape::NTuple{N, Int}
    paths::Vector{String}
    targets::T
    use_threads::Bool
end

function init_buffer(d::FileDataset, n)
    return (
        features = zeros(Float32, d.shape..., n),
        targets = similar(d.targets, 1, n),
    )
end

MLUtils.numobs(d::FileDataset) = length(d.paths)

offset(x, i) = (i - 1) * prod(size(x)[1:end-1]) + 1

function _load_file!(buffer, d::FileDataset, isrc::Int, idest::Int)
    x = d.loadfn(d.paths[isrc])
    copyto!(buffer, offset(buffer, idest), x, 1, length(x))
    return
end

function _get_features!(buffer, d::FileDataset, inds)
    if d.use_threads
        Threads.@threads for i in 1:length(inds)
            _load_file!(buffer, d, inds[i], i)
        end
    else
        for i in 1:length(inds)
            _load_file!(buffer, d, inds[i], i)
        end
    end
    return buffer
end

function _get_features(d::FileDataset, inds)
    shape = d.shape
    buffer = Array{Float32, length(shape) + 1}(undef, shape..., length(inds))
    _get_features!(buffer, d, inds)
    return buffer
end

function MLUtils.getobs(d::FileDataset)
    features = _get_features(d, 1:numobs(d))
    return (features = features, targets = d.targets)
end

function MLUtils.getobs(d::FileDataset, inds)
    features = _get_features(d, inds)
    return (features = features, targets = getobs(d.targets, inds))
end

function MLUtils.getobs!(buffer, d::FileDataset)
    _get_features!(buffer.features, d, 1:numobs(d))
    getobs!(buffer.targets, d.targets)
    return buffer
end

function MLUtils.getobs!(buffer, d::FileDataset, inds)
    _get_features!(buffer.features, d, inds)
    getobs!(buffer.targets, d.targets, inds)
    return buffer
end

# ------------------------------------------------------------------------------------------
# BatchLoader
# ------------------------------------------------------------------------------------------
struct BatchLoader{D<:LabeledDataset, B}
    data::D
    buffer
    batch_neg::Int
    batch_pos::Int
    neg::Vector{Int}
    pos::Vector{Int}
    last_batch::Vector{Int}
    batch_buffer::B

    function BatchLoader(
        data::LabeledDataset;
        buffer=() -> Int[],
        batch_neg::Integer,
        batch_pos::Integer
    )

        y = vec(data.targets)
        batch_buffer = init_buffer(data, batch_neg + batch_pos)

        return new{typeof(data), typeof(batch_buffer)}(
            data,
            buffer,
            batch_neg,
            batch_pos,
            findall(y .== 0),
            findall(y .== 1),
            rand(1:length(y), batch_neg + batch_pos),
            batch_buffer,
        )
    end
end

function _find_inds!(loader::BatchLoader)
    batch_neg = loader.batch_neg
    batch_pos = loader.batch_pos
    batch_size = batch_neg + batch_pos
    inds = vcat(
        sample(loader.neg, batch_neg; replace=length(loader.neg) < batch_neg),
        sample(loader.pos, batch_pos; replace=length(loader.pos) < batch_pos),
    ) |> shuffle

    # buffer for aatp
    delayed_inds = loader.buffer()
    filter!(i -> 1 <= i <= batch_size, delayed_inds)
    k = length(delayed_inds)
    if !isempty(delayed_inds) && k <= batch_size
        position = sample(1:batch_size, k; replace=false)
        inds[position] .= loader.last_batch[delayed_inds]
    end

    # update last batch
    loader.last_batch .= inds
    return inds
end

function Base.length(d::BatchLoader)
    batch_size = d.batch_neg + d.batch_pos
    return ceil(Int, numobs(d.data) / batch_size)
end

Base.IteratorEltype(d::BatchLoader) = Base.EltypeUnknown()

function Base.iterate(d::BatchLoader, iter = 1)
    l = length(d)
    if l < 1 || l < iter
        return nothing
    end
    inds = _find_inds!(d)
    getobs!(d.batch_buffer, d.data, inds)
    return d.batch_buffer, iter + 2
end

# ------------------------------------------------------------------------------------------
# EvalLoader
# ------------------------------------------------------------------------------------------
struct EvalLoader{D<:LabeledDataset, B}
    data::D
    batch_size::Int
    batch_buffer::B

    function EvalLoader(data::LabeledDataset, batch_size::Integer)
        if batch_size == 0 || batch_size >= numobs(data)
            batch_size = 0
            batch_buffer = nothing
        else
            batch_buffer = init_buffer(data, batch_size)
        end

        return new{typeof(data), typeof(batch_buffer)}(
            data,
            batch_size,
            batch_buffer,
        )
    end
end

Base.IteratorEltype(d::EvalLoader) = Base.EltypeUnknown()
MLUtils.numobs(d::EvalLoader) = numobs(d.data)

function Base.length(d::EvalLoader)
    if d.batch_size == 0
        return 1
    else
        return ceil(Int, numobs(d.data) / d.batch_size)
    end
end

function Base.iterate(d::EvalLoader, iter = 1)
    l = length(d)
    if l < 1 || l < iter
        return nothing
    end
    if l == 1
        return (1:numobs(d), getobs(d.data)), iter + 1
    end

    start = 1 + (iter - 1)*d.batch_size
    stop = min(d.batch_size + (iter - 1)*d.batch_size, numobs(d))
    inds = start:stop

    getobs!(d.batch_buffer, d.data, inds)
    if length(inds) == d.batch_size
        return (inds, d.batch_buffer), iter + 1
    else
        return (inds, getobs(d.batch_buffer, 1:length(inds))), iter + 1
    end
end
