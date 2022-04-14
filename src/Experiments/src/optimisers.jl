@option struct OptConfig
    type::String = "Descent"
    eta::Real = 0.01
    decay_every::Int = 1
    decay_step::Real = 1
    decay_min::Real = 1e-6
end

function Base.string(o::OptConfig)
    vals = string.([o.eta, o.decay_every, o.decay_step, o.decay_min])
    return "$(o.type)($(join(vals, ", ")))"
end

materialize(o::OptConfig) = materialize(Val(Symbol(o.type)), o)
materialize(::Val{:Descent}, o::OptConfig) = Descent(o.eta)
materialize(::Val{:ADAM}, o::OptConfig) = ADAM(o.eta)