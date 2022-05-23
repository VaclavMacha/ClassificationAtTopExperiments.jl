abstract type LossType end

sqsum(x) = sum(abs2, x)
aggregation(y, s, ϵ=0.5f0) = mean(ϵ .* y .* s .+ (1 - ϵ) .* (1 .- y) .* s)

# CrossEntropy
@kwdef struct CrossEntropy <: LossType
    λ::Float64 = 0
    ϵ::Float64 = 0.5
end

parse_type(::Val{:CrossEntropy}) = CrossEntropy

function materialize(o::CrossEntropy)
    ϵ = Float32(o.ϵ)
    λ = Float32(o.λ)

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        λ * sum(sqsum, pars) + logitbinarycrossentropy(s, y; agg=s -> aggregation(y, s, ϵ))
    end
    return loss
end

# surrogates
surrogate(name::String, args...) = surrogate(Val(Symbol(name)), args...)
surrogate(::Val{:Hinge}, ϑ=1) = x -> hinge(x, ϑ)
surrogate(::Val{:Quadratic}, ϑ=1) = x -> quadratic(x, ϑ)

# PatMatNP
@kwdef struct PatMatNP
    τ::Float64 = 0.01
    λ::Float64 = 0
    surrogate::String = "Hinge"
    ϑ::Float64 = 1
end

parse_type(::Val{:PatMatNP}) = PatMatNP

function materialize(o::PatMatNP)
    τ = Float32(o.τ)
    λ = Float32(o.λ)
    ϑ = Float32(o.ϑ)
    l1 = surrogate(o.surrogate, 1)
    l2 = surrogate(o.surrogate, ϑ)
    aatp = AccuracyAtTopPrimal.PatMatNP(τ, l2)

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        t = AccuracyAtTopPrimal.threshold(aatp, y, s)
        λ * sum(sqsum, pars) + AccuracyAtTopPrimal.fnr(y, s, t, l1)
    end
    return loss
end

# DeepTopPush
@kwdef struct DeepTopPush <: LossType
    λ::Float64 = 0
    surrogate::String = "Hinge"
end

parse_type(::Val{:DeepTopPush}) = DeepTopPush

function materialize(o::DeepTopPush)
    λ = Float32(o.λ)
    l = surrogate(o.surrogate, 1)
    aatp = AccuracyAtTop.DeepTopPush()

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        λ * sum(sqsum, pars) + AccuracyAtTop.objective(aatp, y, s; surrogate=l)
    end
    return loss
end
