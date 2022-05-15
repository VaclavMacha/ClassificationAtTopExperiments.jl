struct BatchLoader{A, B<:AbstractArray}
    x::A
    y::B
    neg::Vector{Int}
    pos::Vector{Int}
    batch_size::Int
    batch_neg::Int
    batch_pos::Int
    buffer::Bool
    device
    last_batch::Vector{Int}
    t_inds::Vector{Int}
    ts::Vector{Float32}
    position::Vector{Int}

    function BatchLoader(
        x::A,
        y::B;
        buffer::Bool = false,
        batch_neg = 16,
        batch_pos = 16,
        device = identity,
        postprocess = identity
    ) where {A, B<:AbstractArray}

        batch_size = batch_pos + batch_neg

        return new{A,B}(
            x,
            y,
            findall(vec(y) .== 0),
            findall(vec(y) .== 1),
            batch_size,
            batch_neg,
            batch_pos,
            buffer,
            device,
            sample(1:length(y), batch_size; replace = false),
            Int[],
            Float32[],
            Int[],
        )
    end
end

function (b::BatchLoader)()
    inds = vcat(
        sample(b.neg, b.batch_neg; replace = length(b.neg) < b.batch_neg),
        sample(b.pos, b.batch_pos; replace = length(b.pos) < b.batch_pos),
    ) |> shuffle

    if b.buffer
        buff_inds = AccuracyAtTop.buffer_inds()
        filter!(i -> 1 <= i <= b.batch_size, buff_inds)
        k = length(buff_inds)
        if !isempty(buff_inds) && k <= b.batch_size
            position = sample(1:b.batch_size, k; replace = false)
            inds[position] .= b.last_batch[buff_inds]
            append!(b.t_inds, b.last_batch[buff_inds])

            # add position
            deleteat!(b.position, 1:length(b.position))
            append!(b.position, position)
        end
    end
    append!(b.ts, AccuracyAtTop.buffer_ts())
    b.last_batch .= inds
    x, y = getobs((b.x, b.y), inds)
    x = cat(x...; dims = ndims(x) + 1)
    return b.device(x), b.device(Float32.(y))
end
