abstract type LossType end

sqsum(x) = sum(abs2, x)
loss(o::LossType, x, y, model, pars) = loss(o, y, model(x), pars)

@option "CrossEntropy" struct CrossEntropy <: LossType
    λ::Float32 = 0
    ϵ::Float32 = 0.5
end

function loss(o::CrossEntropy, y, s::AbstractArray{T}, pars) where {T}
    agg = s -> mean(o.ϵ .* y .* s .+ (1 - o.ϵ) .* (1 .- y) .* s)
    return T(o.λ) * sum(sqsum, pars) + logitbinarycrossentropy(s, y; agg)
end

# surrogates
@option "Hinge" struct Hinge
    ϑ::Float32 = 1
end

materialize(l::Hinge, ϑ=l.ϑ) = x -> hinge(x, ϑ)

@option "Quadratic" struct Quadratic
    ϑ::Float32 = 1
end

materialize(l::Quadratic, ϑ=l.ϑ) = x -> quadratic(x, ϑ)

# thresholds
@option "PatMatNP" struct PatMatType
    τ::Float32 = 0.01
end

materialize(t::PatMatType, l) = PatMatNP(t.τ, l)
Base.string(t::PatMatType) = "PatMatNP($(t.τ))"

@option "TopPush" struct TopPushType end
materialize(::TopPushType, l) = TopPush(l)
Base.string(::TopPushType) = "TopPush"

# loss
@option "AATP" struct AATP <: LossType
    λ::Float32 = 0
    surrogate::Union{Hinge,Quadratic} = Hinge()
    threshold::Union{PatMatType,TopPushType} = PatMatType()
end

function loss(o::AATP, y, s::AbstractArray{T}, pars) where {T}
    l1 = materialize(o.surrogate, 1)
    l2 = materialize(o.surrogate)
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
