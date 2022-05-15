abstract type ModelType end

@option "Linear" struct Linear <: ModelType
    pretrained = false
end

function materialize(m::Linear; device=identity)
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

@option "EfficientNet" struct EfficientNet <: ModelType
    pretrained = false
    type = "b0"
end

function materialize(m::EfficientNet; device=identity)
    if m.pretrained
        error("pretrained model not available")
    else
        @info "Generating new network"
        # model = EffNet(
        #     "efficientnet-$(m.type)";
        #     n_classes=1,
        #     in_channels=3
        # ) |> device
        model = Chain(Flux.flatten, Dense(196608, 1)) |> device
    end
    return model, Flux.params(model)
end

@option "unknown" struct Dummy end

@option struct ModelConfig
    model::Union{Linear,EfficientNet,Dummy}
end

Base.string(m::ModelConfig) = string(m.model)
materialize(m::ModelConfig; kwargs...) = materialize(m.model; kwargs...)
