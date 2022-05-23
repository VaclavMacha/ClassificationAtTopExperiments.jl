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
@option "Efficientnet" struct Efficientnet <: ModelType
    pretrained = false
end

Base.string(m::Efficientnet) = "Efficientnet($(m.pretrained))"

function materialize(::AbstractJMiPOD, m::Efficientnet; device=identity)
    if m.pretrained
        error("pretrained model not available")
    else
        @info "Generating new network"
        model = EfficientNet.EffNet(
            "efficientnet-b0";
            n_classes=1,
            in_channels=3
        ) |> device
    end
    return model, Flux.params(model)
end

# ModelConfig
@option struct ModelConfig
    model::Union{Linear,Efficientnet}
end

Base.string(m::ModelConfig) = string(m.model)
function materialize(d::DataConfig, m::ModelConfig; kwargs...)
    return materialize(d.dataset, m.model; kwargs...)
end
