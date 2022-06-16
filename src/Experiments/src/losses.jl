abstract type LossType end

sqsum(x) = sum(abs2, x)
aggregation(y, s, ϵ=0.5f0) = mean(ϵ .* y .* s .+ (1 - ϵ) .* (1 .- y) .* s)

# surrogates
surrogate(name::String, args...) = surrogate(Val(Symbol(name)), args...)
surrogate(::Val{:Hinge}, ϑ=1) = x -> AccuracyAtTopPrimal.hinge(x, ϑ)
surrogate(::Val{:Quadratic}, ϑ=1) = x -> AccuracyAtTopPrimal.quadratic(x, ϑ)

# ------------------------------------------------------------------------------------------
# CrossEntropy
# ------------------------------------------------------------------------------------------
@kwdef struct CrossEntropy <: LossType
    λ::Float64 = 1e-3
    ϵ::Float64 = 0.5
end

parse_type(::Val{:CrossEntropy}) = CrossEntropy

function materialize(o::CrossEntropy)
    ϵ = Float32(o.ϵ)
    λ = Float32(o.λ)

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        λ/2 * sum(sqsum, pars) + logitbinarycrossentropy(s, y; agg=s -> aggregation(y, s, ϵ))
    end
    return loss
end

# ------------------------------------------------------------------------------------------
# PatMat and PatMatNP
# ------------------------------------------------------------------------------------------
abstract type AbstractPatMat <: LossType end

@kwdef struct PatMat <: AbstractPatMat
    τ::Float64 = 0.01
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
    ϑ::Float64 = 1
end

@kwdef struct PatMatNP <: AbstractPatMat
    τ::Float64 = 0.01
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
    ϑ::Float64 = 1
end

parse_type(::Val{:PatMat}) = PatMat
parse_type(::Val{:PatMatNP}) = PatMatNP

materialize_threshold(::PatMat, args...) = AccuracyAtTopPrimal.PatMat(args...)
materialize_threshold(::PatMatNP, args...) = AccuracyAtTopPrimal.PatMatNP(args...)

function materialize(o::AbstractPatMat)
    τ = Float32(o.τ)
    λ = Float32(o.λ)
    ϑ = Float32(o.ϑ)
    l1 = surrogate(o.surrogate, 1)
    l2 = surrogate(o.surrogate, ϑ)
    aatp = materialize_threshold(o, τ, l2)

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        t = AccuracyAtTopPrimal.threshold(aatp, y, s)
        λ/2 * sum(sqsum, pars) + AccuracyAtTopPrimal.fnr(y, s, t, l1)
    end
    return loss
end

# ------------------------------------------------------------------------------------------
# TopPush, TopPushK, TopMeanτ, tauFPL
# ------------------------------------------------------------------------------------------
abstract type AbstractTopPush <: LossType end

@kwdef struct TopPush <: AbstractTopPush
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
end

@kwdef struct TopPushK <: AbstractTopPush
    K::Int = 5
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
end

@kwdef struct TopMeanTau <: AbstractTopPush
    τ::Float64 = 0.1
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
end

@kwdef struct TauFPL <: AbstractTopPush
    τ::Float64 = 0.1
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
end

parse_type(::Val{:TopPush}) = TopPush
parse_type(::Val{:TopPushK}) = TopPushK
parse_type(::Val{:TopMeanTau}) = TopMeanTau
parse_type(::Val{:TauFPL}) = TauFPL

materialize_threshold(::TopPush) = AccuracyAtTopPrimal.TopPush()
materialize_threshold(o::TopPushK) = AccuracyAtTopPrimal.TopPushK(o.K)
materialize_threshold(o::TopMeanTau) = AccuracyAtTopPrimal.TopMean(Float32(o.τ))
materialize_threshold(o::TauFPL) = AccuracyAtTopPrimal.τFPL(Float32(o.τ))

function materialize(o::AbstractTopPush)
    λ = Float32(o.λ)
    l = surrogate(o.surrogate, 1)
    aatp = materialize_threshold(o)

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        t = AccuracyAtTopPrimal.threshold(aatp, y, s)
        λ/2 * sum(sqsum, pars) + AccuracyAtTopPrimal.fnr(y, s, t, l)
    end
    return loss
end

# ------------------------------------------------------------------------------------------
# Grill and GrillNP
# ------------------------------------------------------------------------------------------
abstract type AbstractGrill <: LossType end

@kwdef struct Grill <: AbstractGrill
    τ::Float64 = 0.1
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
end

@kwdef struct GrillNP <: AbstractGrill
    τ::Float64 = 0.1
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
end

parse_type(::Val{:Grill}) = Grill
parse_type(::Val{:GrillNP}) = GrillNP

materialize_threshold(o::Grill) = AccuracyAtTopPrimal.Grill(Float32(o.τ))
materialize_threshold(o::GrillNP) = AccuracyAtTopPrimal.GrillNP(Float32(o.τ))

function materialize(o::AbstractGrill)
    λ = Float32(o.λ)
    l = surrogate(o.surrogate, 1)
    aatp = materialize_threshold(o)

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        t = AccuracyAtTopPrimal.threshold(aatp, y, s)
        λ/2 * sum(sqsum, pars) + AccuracyAtTopPrimal.fnr(y, s, t, l) + AccuracyAtTopPrimal.fpr(y, s, t, l)
    end
    return loss
end


# ------------------------------------------------------------------------------------------
# DeepTopPush
# ------------------------------------------------------------------------------------------
@kwdef struct DeepTopPush <: LossType
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
end

parse_type(::Val{:DeepTopPush}) = DeepTopPush

function materialize(o::DeepTopPush)
    λ = Float32(o.λ)
    l = surrogate(o.surrogate, 1)
    aatp = AccuracyAtTop.DeepTopPush()

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        λ/2 * sum(sqsum, pars) + AccuracyAtTop.objective(aatp, y, s; surrogate=l)
    end
    return loss
end

@kwdef struct DeepTopPushCross <: LossType
    λ::Float64 = 1e-3
    α::Float64 = 0.5
    surrogate::String = "Hinge"
end

parse_type(::Val{:DeepTopPushCross}) = DeepTopPushCross

function materialize(o::DeepTopPushCross)
    λ = Float32(o.λ)
    α = Float32(o.α)
    l = surrogate(o.surrogate, 1)
    aatp = AccuracyAtTop.DeepTopPush()

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        λ/2 * sum(sqsum, pars) + α * logitbinarycrossentropy(s, y) + (1 - α) * AccuracyAtTop.objective(aatp, y, s; surrogate=l)
    end
    return loss
end
