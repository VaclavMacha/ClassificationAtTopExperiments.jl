abstract type ModelType end

# linear models
@option "Linear" struct Linear <: ModelType
    pretrained = false
end

function materialize(::AbstractNsf5, m::Linear; device=identity)
    if m.pretrained
        error("pretrained model not available")
    else
        @info "Generating new network"
        model = Dense(22510, 1) |> device
    end
    pars = Flux.params(model)
    delete!(pars, model.bias)
    return model, pars
end

function materialize(::AbstractJMiPOD, m::Linear; device=identity)
    if m.pretrained
        error("pretrained model not available")
    else
        @info "Generating new network"
        model = Chain(Flux.flatten, Dense(196608, 1)) |> device
    end
    pars = Flux.params(model)
    delete!(pars, model[end].bias)
    return model, pars
end

# Efficient net
@option "EffNet" struct EffNet <: ModelType
    pretrained = false
    type = "b0"
end

Base.string(m::EffNet) = "EffNet($(m.pretrained), $(m.type))"

function materialize(::AbstractJMiPOD, m::EfficientNet; device=identity)
    if m.type == "b0"
        T = B0
    else
        error("not supported")
    end
    if m.pretrained
        error("pretrained model not available")
    else
        @info "Generating new network"
        model = EfficientNet(
            T;
            classes=1,
            channels=3
        ) |> device
    end
    return model, Flux.params(model)
end

# ModelConfig
@option struct ModelConfig
    model::Union{Linear,EffNet}
end

Base.string(m::ModelConfig) = string(m.model)
function materialize(d::DataConfig, m::ModelConfig; kwargs...)
    return materialize(d.dataset, m.model; kwargs...)
end
