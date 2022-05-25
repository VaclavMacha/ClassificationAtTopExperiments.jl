struct BatchLoader{D<:LabeledDataset}
    data::D
    buffer
    batch_neg::Int
    batch_pos::Int
    neg::Vector{Int}
    pos::Vector{Int}
    last_batch::Vector{Int}

    function BatchLoader(
        data::LabeledDataset;
        buffer=() -> Int[],
        batch_neg::Integer=0,
        batch_pos::Integer=0
    )

        y = vec(data.targets)

        return new{typeof(data)}(
            data,
            buffer,
            batch_neg,
            batch_pos,
            findall(y .== 0),
            findall(y .== 1),
            rand(1:length(y), batch_neg + batch_pos),
        )
    end
end

function (loader::BatchLoader)()
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
    return obsview(loader.data, inds)
end

struct Batch{F1,T1,F2,T2}
    device
    x::F1
    y::T1
    x_device::F2
    y_device::T2
end

function Batch(device, shape::Int...)
    x = zeros(Float32, shape...)
    y = zeros(Float32, 1, shape[end])

    if isa(device, typeof(cpu))
        return Batch(device, x, y, nothing, nothing)
    else
        return Batch(device, x, y, device(x), device(y))
    end
end

Batch(device) = Batch(device, nothing, nothing, nothing, nothing)

offset(x, i) = (i - 1) * prod(size(x)[1:end-1]) + 1

function get_batch!(batch::Batch, data)
    Threads.@threads for i in 1:length(data)
        x, y = getobs(data, i)
        copyto!(batch.x, offset(batch.x, i), x, 1, length(x))
        copyto!(batch.y, offset(batch.y, i), y, 1, length(y))
    end
    if !isnothing(batch.x_device)
        copyto!(batch.x_device, batch.x)
        copyto!(batch.y_device, batch.y)
        return batch.x_device, batch.y_device
    else
        return batch.x, batch.y
    end
end

function eval_model!(
    batch::Batch,
    data::LabeledDataset,
    model,
)
    batchsize = length(batch.y)
    n = length(data)
    S = zeros(Float32, 1, n)
    Y = zeros(Bool, 1, n)

    for (inds, data_i) in BatchView((1:n, data); batchsize, partial=true)
        x, y = get_batch!(batch, data_i)
        jnds = 1:length(inds)
        S[inds] .= cpu(model(x))[jnds]
        Y[inds] .= cpu(y)[jnds]
    end
    return Y, S
end

function eval_model!(
    batch::Batch{Nothing},
    data::LabeledDataset,
    model,
)

    x = batch.device(data[1][:])
    y = reshape(data[2][:], 1, :)
    return cpu(model(x)), y
end
