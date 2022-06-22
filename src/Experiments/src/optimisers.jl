abstract type OptimiserType end

@kwdef struct OptDescent <: OptimiserType
    eta::Float64 = 0.1
    decay_every::Int = 5
    decay::Float64 = 0.8
end

parse_type(::Val{:OptDescent}) = OptDescent
function materialize(o::OptDescent, max_iter::Int)
    opt = Descent(o.eta)
    return if o.decay_every == 0
        opt
    else
        decay = Flux.ExpDecay(1.0, o.decay, o.decay_every * max_iter)
        Flux.Optimiser(opt, decay)
    end
end

@kwdef struct OptADAM <: OptimiserType
    eta::Float64 = 0.001
    beta1::Float64 = 0.9
    beta2::Float64 = 0.999
    decay_every::Int = 5
    decay::Float64 = 0.8
end

parse_type(::Val{:OptADAM}) = OptADAM
function materialize(o::OptADAM, max_iter::Int)
    opt = ADAM(o.eta, (o.beta1, o.beta2))
    return if o.decay_every == 0
        opt
    else
        decay = Flux.ExpDecay(1.0, o.decay, o.decay_every * max_iter)
        Flux.Optimiser(opt, decay)
    end
end

@kwdef struct OptRMSProp <: OptimiserType
    eta::Float64 = 0.001
    rho::Float64 = 0.9
    decay_every::Int = 5
    decay::Float64 = 0.8
end

parse_type(::Val{:OptRMSProp}) = OptRMSProp
function materialize(o::OptRMSProp, max_iter::Int)
    opt = RMSProp(o.eta, o.rho)
    return if o.decay_every == 0
        opt
    else
        decay = Flux.ExpDecay(1.0, o.decay, o.decay_every * max_iter)
        Flux.Optimiser(opt, decay)
    end
end
