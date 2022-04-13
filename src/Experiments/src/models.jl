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

@option struct ModelConfig
    model::Union{Linear}
end