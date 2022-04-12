@option "optimiser" struct Optimiser
    type::Symbol = :Descent
    eta::Real = 0.01
    decay_every::Int = 1
    decay_step::Real = 1
    decay_min::Real = 1e-6
end

materialize(o::Optimiser) = materialize(Val(o.type), o)
materialize(::Val{:Descent}, o::Optimiser) = Descent(o.eta)
materialize(::Val{:ADAM}, o::Optimiser) = ADAM(o.eta)