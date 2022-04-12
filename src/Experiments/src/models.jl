@option "model" struct Model
    type::Symbol = :Linear
    pretrained = false
end

materialize(m::Model; kwargs...) = materialize(Val(m.type), m; kwargs...)

function materialize(::Val{:Linear}, m::Model; device=identity)
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