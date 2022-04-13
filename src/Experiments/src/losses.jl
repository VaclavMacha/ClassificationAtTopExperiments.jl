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

@option "AATP" struct AATP <: LossType
    λ::Float32 = 0
    τ::Float32 = 0.01
    l::Function = hinge
    ϑ::Float32 = 1
end

function loss(o::AATP, y, s::AbstractArray{T}, pars) where {T}
    t = threshold(PatMatNP(o.τ, x -> o.l(x, o.ϑ)), y, s)
    return T(o.λ) * sum(sqsum, pars) + fnr(y, s, t, o.l)
end

@option struct LossConfig
    loss::Union{CrossEntropy,AATP}
end