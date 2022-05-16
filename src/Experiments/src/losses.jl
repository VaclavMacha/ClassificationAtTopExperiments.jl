abstract type LossType end

sqsum(x) = sum(abs2, x)
loss(o::LossType, x, y, model, pars) = loss(o, y, model(x), pars)

# CrossEntropy
@option "CrossEntropy" struct CrossEntropy <: LossType
    λ::Float64 = 0
    ϵ::Float64 = 0.5
end

Base.string(o::CrossEntropy) = "CrossEntropy($(dir_string(o.λ)), $(dir_string(o.ϵ)))"

function loss(o::CrossEntropy, y, s::AbstractArray{T}, pars) where {T}
    ϵ = T(o.ϵ)
    agg = s -> mean(ϵ .* y .* s .+ (1 - ϵ) .* (1 .- y) .* s)
    return T(o.λ) * sum(sqsum, pars) + logitbinarycrossentropy(s, y; agg)
end

# surrogates
surrogate(name::String, args...) = surrogate(Val(Symbol(name)), args...)
surrogate(::Val{:Hinge}, ϑ=1) = x -> hinge(x, ϑ)
surrogate(::Val{:Quadratic}, ϑ=1) = x -> quadratic(x, ϑ)

# PatMatNP
@option "PatMatNP" struct PatMatNP
    τ::Float64 = 0.01
    λ::Float64 = 0
    surrogate::String = "Hinge"
    ϑ::Float64 = 1
end

function Base.string(l::PatMatNP)
    vals = dir_string.((l.τ, l.λ, l.surrogate, l.ϑ))
    return "PatMatNP($(join(vals, ", ")))"
end

function loss(o::PatMatNP, y, s::AbstractArray{T}, pars) where {T}
    l1 = surrogate(o.surrogate, 1)
    l2 = surrogate(o.surrogate, T(o.ϑ))
    t_type = AccuracyAtTopPrimal.PatMatNP(o.τ, l2)
    t = AccuracyAtTopPrimal.threshold(t_type, y, s)
    return T(o.λ) * sum(sqsum, pars) + AccuracyAtTopPrimal.fnr(y, s, t, l1)
end

# DeepTopPush
@option "DeepTopPush" struct DeepTopPush <: LossType
    λ::Float64 = 0
    surrogate::String = "Hinge"
end

function loss(o::DeepTopPush, y, s::AbstractArray{T}, pars) where {T}
    l = surrogate(o.surrogate, 1)
    aatp = AccuracyAtTop.DeepTopPush()
    return T(o.λ) * sum(sqsum, pars) + AccuracyAtTop.objective(aatp, y, s; surrogate=l)
end

# LossConfig
@option struct LossConfig
    loss::Union{CrossEntropy,PatMatNP,DeepTopPush}
end

Base.string(m::LossConfig) = string(m.loss)
loss(o::LossConfig, x, y, model, pars) = loss(o.loss, x, y, model, pars)
loss(o::LossConfig, y, s, pars) = loss(o.loss, y, s, pars)
