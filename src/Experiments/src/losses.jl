abstract type LossType end

sqsum(x) = sum(abs2, x)
aggregation(y, s, ϵ=0.5f0) = mean(ϵ .* y .* s .+ (1 - ϵ) .* (1 .- y) .* s)

# surrogates
surrogate(name::String, args...) = surrogate(Val(Symbol(name)), args...)
surrogate(::Val{:Hinge}, ϑ=1) = x -> AccuracyAtTopPrimal.hinge(x, ϑ)
surrogate(::Val{:Quadratic}, ϑ=1) = x -> AccuracyAtTopPrimal.quadratic(x, ϑ)
surrogate(::Val{:Softplus}, ϑ=1) = x -> softplus(x)

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
        λ / 2 * sum(sqsum, pars) + logitbinarycrossentropy(s, y; agg=s -> aggregation(y, s, ϵ))
    end
    return loss
end

# ECM
@kwdef struct ECM <: LossType
    λ::Float64 = 1e-3
end

parse_type(::Val{:ECM}) = ECM

function materialize(o::ECM)
    λ = Float32(o.λ)

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        mask = y .== +1
        mn = mean(s[mask])
        λ / 2 * sum(sqsum, pars) + mean(Flux.softplus(s[.!mask] .- mn))
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
        λ / 2 * sum(sqsum, pars) + AccuracyAtTopPrimal.fnr(y, s, t, l1)
    end
    return loss
end

function materialize_dual(o::AbstractPatMat, n_pos::Int)
    τ = Float32(o.τ)
    λ = Float32(o.λ)
    ϑ = Float32(o.ϑ)
    if λ == 0
        C = Float32(1)
    else
        C = 1 / (λ * n_pos)
    end

    S = if o.surrogate == "Hinge"
        ClassificationAtTopDual.Hinge
    else
        ClassificationAtTopDual.Quadratic
    end
    return if isa(o, PatMat)
        ClassificationAtTopDual.PatMat(τ; ϑ, C, S)
    else
        ClassificationAtTopDual.PatMatNP(τ; ϑ, C, S)
    end
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

@kwdef struct TopMeanK <: AbstractTopPush
    τ::Float64 = 0.01
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
end

@kwdef struct TauFPL <: AbstractTopPush
    τ::Float64 = 0.01
    λ::Float64 = 1e-3
    surrogate::String = "Hinge"
end

parse_type(::Val{:TopPush}) = TopPush
parse_type(::Val{:TopPushK}) = TopPushK
parse_type(::Val{:TopMeanK}) = TopMeanK
parse_type(::Val{:TauFPL}) = TauFPL

materialize_threshold(::TopPush) = AccuracyAtTopPrimal.TopPush()
materialize_threshold(o::TopPushK) = AccuracyAtTopPrimal.TopPushK(o.K)
materialize_threshold(o::TopMeanK) = AccuracyAtTopPrimal.TopMean(Float32(o.τ))
materialize_threshold(o::TauFPL) = AccuracyAtTopPrimal.τFPL(Float32(o.τ))

function materialize(o::AbstractTopPush)
    λ = Float32(o.λ)
    l = surrogate(o.surrogate, 1)
    aatp = materialize_threshold(o)

    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        t = AccuracyAtTopPrimal.threshold(aatp, y, s)
        λ / 2 * sum(sqsum, pars) + AccuracyAtTopPrimal.fnr(y, s, t, l)
    end
    return loss
end

function materialize_dual(o::AbstractTopPush, n_pos::Int)
    λ = Float32(o.λ)
    if λ == 0
        C = Float32(1)
    else
        C = 1 / (λ * n_pos)
    end

    S = if o.surrogate == "Hinge"
        ClassificationAtTopDual.Hinge
    else
        ClassificationAtTopDual.Quadratic
    end
    return if isa(o, TopPush)
        ClassificationAtTopDual.TopPush(; C, S)
    elseif isa(o, TopPushK)
        ClassificationAtTopDual.TopPushK(o.K; C, S)
    elseif isa(o, TopMeanK)
        τ = Float32(o.τ)
        ClassificationAtTopDual.TopMeanK(τ; C, S)
    else
        τ = Float32(o.τ)
        ClassificationAtTopDual.tauFPL(τ; C, S)
    end
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
        λ / 2 * sum(sqsum, pars) + AccuracyAtTopPrimal.fnr(y, s, t, l) + AccuracyAtTopPrimal.fpr(y, s, t, l)
    end
    return loss
end


# ------------------------------------------------------------------------------------------
# DeepTopPush
# ------------------------------------------------------------------------------------------
abstract type AbstractDeepTopPush <: LossType end

@kwdef struct DeepTopPush <: AbstractDeepTopPush
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
        λ / 2 * sum(sqsum, pars) + AccuracyAtTop.objective(aatp, y, s; surrogate=l)
    end
    return loss
end

@kwdef struct DeepTopPushCross <: AbstractDeepTopPush
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
        λ / 2 * sum(sqsum, pars) + α * logitbinarycrossentropy(s, y) + (1 - α) * AccuracyAtTop.objective(aatp, y, s; surrogate=l)
    end
    return loss
end

# ------------------------------------------------------------------------------------------
# SVM
# ------------------------------------------------------------------------------------------
@kwdef struct SVM <: LossType
    λ::Float64 = 1e-3
end

parse_type(::Val{:SVM}) = SVM


function materialize_dual(o::SVM, n_pos::Int)
    λ = Float32(o.λ)
    if λ == 0
        C = Float32(1)
    else
        C = 1 / (λ * n_pos)
    end
    return ClassificationAtTopDual.SVM(; C)
end


# ------------------------------------------------------------------------------------------
# MODE loss
# ------------------------------------------------------------------------------------------
@kwdef struct MODE <: LossType
    fpr::Float64 = 1e-3
end

parse_type(::Val{:MODE}) = MODE

"""
    interp1d(x, y, xnew)

Linear 1D interpolation for Julia vectors with scalar query point (CPU/GPU compatible).
This function returns the interpolated value at the desired query point `xnew`.

Parameters
----------
x : Vector
    A 1-D vector of real values (must be sorted).
y : Vector  
    A 1-D vector of real values, same length as x.
xnew : Number
    A scalar value where interpolation is desired.

Returns
-------
ynew : Number
    Interpolated value at query point xnew
"""
function interp1d(x::AbstractVector, y::AbstractVector, xnew::Number)
    @assert length(x) == length(y) "x and y must have the same length"
    @assert length(x) >= 2 "x and y must have at least 2 points"
    
    n = length(x)
    ϵ = eps(eltype(y))
    
    # Find the index where xnew should be inserted
    idx = clamp(searchsortedlast(x, xnew), 1, n - 1)
    x₁, x₂ = x[idx], x[idx + 1]
    y₁, y₂ = y[idx], y[idx + 1]
    slope = (y₂ - y₁) / (ϵ + (x₂ - x₁))
    ynew = y₁ + slope * (xnew - x₁)
    return ynew
end


function init_y(nobs, number_of_covers)
    start = 1/nobs
    stop = 1.0
    y = collect(range(1 ./ nobs, 1, nobs))
    if number_of_covers > 0 
        y[end-1] = 1 - 1/number_of_covers
    end
    y
end

function eicdf(x, x_new, number_of_covers = 0)
    x = sort(x)
    y = Zygote.@ignore init_y(length(x), number_of_covers)
    interp1d(y, x, x_new)
end

function materialize(o::MODE, number_of_covers::Integer)
    fpr = Float32(o.fpr)
    loss(x, y, model, pars) = loss(y, model(x), pars)

    function loss(y, s::AbstractArray, pars)
        cover_logits = s[y .== 0]
        stego_logits = s[y .== 1]
        worst_index = partialsortperm(cover_logits, 1:2, rev = true)
        AccuracyAtTop.update_buffer!(cover_logits[worst_index], worst_index)

        γ = eicdf(cover_logits, 1-fpr, number_of_covers)
        return mean(softplus.(γ .- stego_logits)) + mean(softplus.(cover_logits .- γ))
    end
    return loss
end