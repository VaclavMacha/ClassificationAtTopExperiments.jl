@option "model" struct Model
    type::Symbol = :Linear
    pretrained = false
end

materialize(m::Model) = materialize(Val(m.type), m)

function materialize(::Val{:Linear}, m::Model; device=identity)
    if m.pretrained
        error("pretrained model not available")
    else
        @info "Generating new network"
        model = Dense(22510, 1) |> device
    end
    pars = Flux.params(model)
    delete!(pars, model.b)
    return model, pars
end