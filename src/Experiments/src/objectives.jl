sqsum(x) = sum(abs2, x)
loss(o, x, y, model, pars) = loss(o, y, model(x), pars)

Base.@kwdef struct CrossEntropy
    λ::Float32 = 0
    ϵ::Float32 = 0.5
end

function loss(o::CrossEntropy, y, s::AbstractArray{T}, pars) where {T}
    agg = s -> mean(o.ϵ .* y .* s .+ (1 - o.ϵ) .* (1 .- y) .* s)
    return T(o.λ) * sum(sqsum, pars) + logitbinarycrossentropy(s, y; agg)
end

Base.@kwdef struct PatMatObjective
    λ::Float32 = 0
    τ::Float32 = 0.01
    l::Function = hinge
end

function loss(o::PatMatObjective, y, s::AbstractArray{T}, pars) where {T}
    t = threshold(PatMatNP(o.τ, o.l), y, s)
    return T(o.λ) * sum(sqsum, pars) + fnr(y, s, t, o.l)
end