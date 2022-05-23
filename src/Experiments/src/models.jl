abstract type ModelType end

# linear models
struct Linear <: ModelType end

parse_type(::Val{:Linear}) = Linear

function materialize(::AbstractNsf5, m::Linear)
    return Dense(22510 => 1; bias=false)
end

function materialize(::AbstractJMiPOD, m::Linear)
    return Chain(Flux.flatten, Dense(196608 => 1; bias=false))
end

# Efficient net
abstract type AbstractEfficientNet <: ModelType end

@kwdef struct EfficientNetB0 <: AbstractEfficientNet
    pretrained::Bool = false
end

parse_type(::Val{:EfficientNetB0}) = EfficientNetB0
efficientnet_type(::EfficientNetB0) = "efficientnet-b0"

function materialize(::AbstractJMiPOD, m::AbstractEfficientNet)
    return if m.pretrained
        error("pretrained model not available")
    else
        EfficientNet.EffNet(efficientnet_type(m); n_classes=1, in_channels=3)
    end
end
