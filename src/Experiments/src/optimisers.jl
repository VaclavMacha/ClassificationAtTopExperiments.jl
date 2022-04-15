@option struct OptConfig
    type::String = "Descent"
    eta::Float32 = 0.01
    decay_every::Int = 1
    decay_step::Float32 = 1
    decay_min::Float32 = 1e-6
end

function Base.string(o::OptConfig)
    vals = dir_string.((o.eta, o.decay_every, o.decay_step, o.decay_min))
    return "$(o.type)($(join(vals, ", ")))"
end

materialize(o::OptConfig) = materialize(Val(Symbol(o.type)), o)
materialize(::Val{:Descent}, o::OptConfig) = Descent(o.eta)
materialize(::Val{:ADAM}, o::OptConfig) = ADAM(o.eta)