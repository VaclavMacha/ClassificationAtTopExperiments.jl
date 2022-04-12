@option "model" struct Model
    type::String = "Linear"
    pretrained = false
end

materialize(m::Model) = materialize(Val(Symbol(m.type)), m)

function materialize(::Val{:Linear}, m::Model)
    if m.pretrained
        error("pretrained model not available")
        return
    else
        @info "Generating new network"
        return Dense(22510, 1)
    end
end