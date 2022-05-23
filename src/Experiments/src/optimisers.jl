abstract type OptimiserType end

@kwdef struct OptDescent <: OptimiserType
    eta::Float64 = 0.1
end

materialize(o::OptDescent) = Descent(o.eta)
parse_type(::Val{:OptDescent}) = OptDescent

@kwdef struct OptADAM <: OptimiserType
    eta::Float64 = 0.001
    beta1::Float64 = 0.9
    beta2::Float64 = 0.999
end

materialize(o::OptADAM) = ADAM(o.eta, (o.beta1, o.beta2))
parse_type(::Val{:OptADAM}) = OptADAM
