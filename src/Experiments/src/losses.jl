abstract type LossType end

sqsum(x) = sum(abs2, x)
loss(o::LossType, x, y, model, pars) = loss(o, y, model(x), pars)

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
@option "Hinge" struct Hinge
    ϑ::Float64 = 1
end

materialize(l::Hinge, ϑ=l.ϑ) = x -> hinge(x, ϑ)
Base.string(l::Hinge) = "Hinge($(dir_string(l.ϑ)))"

@option "Quadratic" struct Quadratic
    ϑ::Float64 = 1
end

materialize(l::Quadratic, ϑ=l.ϑ) = x -> quadratic(x, ϑ)
Base.string(l::Quadratic) = "Quadratic($(dir_string(l.ϑ)))"

# thresholds
@option "PatMatNP" struct PatMatType
    τ::Float64 = 0.01
end

materialize(t::PatMatType, l) = PatMatNP(t.τ, l)
Base.string(t::PatMatType) = "PatMatNP($(dir_string(t.τ)))"

@option "TopPush" struct TopPushType end
materialize(::TopPushType, l) = TopPush(l)
Base.string(::TopPushType) = "TopPush"

# loss
@option "AATP" struct AATP <: LossType
    λ::Float64 = 0
    surrogate::Union{Hinge,Quadratic} = Hinge()
    threshold::Union{PatMatType,TopPushType} = PatMatType()
end

function Base.string(l::AATP)
    vals = dir_string.((l.λ, l.surrogate, l.threshold))
    return "AATP($(join(vals, ", ")))"
end

function loss(o::AATP, y, s::AbstractArray{T}, pars) where {T}
    l1 = materialize(o.surrogate, 1)
    l2 = materialize(o.surrogate, T(o.ϑ))
    t_type = materialize(o.threshold, l2)
    t = threshold(t_type, y, s)
    return T(o.λ) * sum(sqsum, pars) + fnr(y, s, t, l1)
end

@option struct LossConfig
    loss::Union{CrossEntropy,AATP}
end

Base.string(m::LossConfig) = string(m.loss)
loss(o::LossConfig, x, y, model, pars) = loss(o.loss, x, y, model, pars)
loss(o::LossConfig, y, s, pars) = loss(o.loss, y, s, pars)
