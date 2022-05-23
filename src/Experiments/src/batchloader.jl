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
        buffer = () -> Int[],
        batch_neg::Integer = 0,
        batch_pos::Integer = 0,
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
    if loader.batch_neg == loader.batch_pos == 0
        return loader.data
    end
    inds = vcat(
        sample(loader.neg, loader.batch_neg; replace=length(loader.neg) < loader.batch_neg),
        sample(loader.pos, loader.batch_pos; replace=length(loader.pos) < loader.batch_pos),
    ) |> shuffle

    # buffer for aatp
    delayed_inds = loader.buffer()
    filter!(i -> 1 <= i <= loader.batch_size, delayed_inds)
    k = length(delayed_inds)
    if !isempty(delayed_inds) && k <= loader.batch_size
        position = sample(1:loader.batch_size, k; replace=false)
        inds[position] .= loader.last_batch[delayed_inds]
    end

    # update last batch
    loader.last_batch .= inds
    return ObsView(loader.data, inds)
end

function get_batch(data, device = cpu)
    data_batch = Vector{Any}(undef, length(data))
    Threads.@threads for i in 1:length(data)
        data_batch[i] = getobs(data, i)
    end

    x = MLUtils.batch(first.(data_batch)) |> device
    y = reshape(MLUtils.batch(last.(data_batch)), 1, :) |> device
    return x, y
end

function eval_model(
    data::LabeledDataset,
    model,
    batchsize,
    device
)

    n = length(data)
    S = zeros(Float32, 1, n)
    Y = zeros(Bool, 1, n)

    for (inds, batch) in BatchView((1:n, data); batchsize, partial=true)
        x, y = get_batch(batch, device)
        S[inds] .= cpu(model(x))[:]
        Y[inds] .= cpu(y)[:]
    end
    return Y, S
end
