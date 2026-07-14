function flatten_gradient(grads)
    flat_grad = Float32[]
    for p in grads
        if p !== nothing
            append!(flat_grad, vec(p))
        end
    end
    return flat_grad
end


function vector_angle(a, b)
    return acosd(clamp(dot(a,b)/(norm(a)*norm(b)), -1, 1))
end


function gradient_angle(grad1, grad2)
    return vector_angle(flatten_gradient(grad1), flatten_gradient(grad2))
end

struct DataWrapper
    X_neg::Matrix{Float32}
    X_pos::Matrix{Float32}
    n_neg::Int
    n_pos::Int
    batch_neg::Int
    batch_pos::Int
    inds_neg::Vector{Int}
    inds_pos::Vector{Int}
end

function update_batch(data::DataWrapper)
    data.inds_neg .= sample(1:data.n_neg, data.batch_neg; replace=false)
    data.inds_pos .= sample(1:data.n_pos, data.batch_pos; replace=false)
end

function get_batch(data::DataWrapper)
    x = @views hcat(data.X_neg[:,data.inds_neg], data.X_pos[:,data.inds_pos])
    y = reshape(1:(data.batch_neg + data.batch_pos) .> data.batch_neg, 1, :)
    return x, y
end

function get_batch(data::DataWrapper, t_ind::Int)
    x = @views hcat(data.X_neg[:, t_ind], data.X_neg[:,data.inds_neg[2:end]], data.X_pos[:,data.inds_pos])
    y = reshape(1:(data.batch_neg + data.batch_pos) .> data.batch_neg, 1, :)
    return x, y
end

function sampled_gradient(model, pars, loss, data)
    local s
    local L
    x, y = get_batch(data)

    grads = Flux.Zygote.gradient(pars) do
        s = model(x)
        L = loss(y, s, pars)
        return L
    end

    aatp = AccuracyAtTop.DeepTopPush()
    t, t_ind = AccuracyAtTop.find_threshold(aatp.threshold_type, y, s)
    return (
        loss = L,
        grads = grads,
        threshold = t,
        threshold_ind = data.inds_neg[t_ind[1]]
    )
end

function enhanced_gradient(model, pars, loss, data, threshold_ind = 0)
    local s
    local L
    if threshold_ind > 0
        x, y = get_batch(data, threshold_ind)
    else
        x, y = get_batch(data)
    end

    grads = Flux.Zygote.gradient(pars) do
        s = model(x)
        L = loss(y, s, pars)
        return L
    end

    aatp = AccuracyAtTop.DeepTopPush()
    t, t_ind = AccuracyAtTop.find_threshold(aatp.threshold_type, y, s)
    return (
        loss = L,
        grads = grads,
        threshold = t,
        threshold_ind = data.inds_neg[t_ind[1]]
    )
end

function actual_gradient(model, pars, loss, data)
    local s
    local L
    aatp = AccuracyAtTop.DeepTopPush()
    _, t_ind_true = AccuracyAtTop.find_threshold(aatp.threshold_type, repeat([0], data.n_neg), vec(model(data.X_neg)))
    x, y = get_batch(data, t_ind_true[1])

    grads = Flux.Zygote.gradient(pars) do
        s = model(x)
        L = loss(y, s, pars)
        return L
    end

    t, t_ind = AccuracyAtTop.find_threshold(aatp.threshold_type, y, s)
    return (
        loss = L,
        grads = grads,
        threshold = t,
        threshold_ind = data.inds_neg[t_ind[1]]
    )
end
