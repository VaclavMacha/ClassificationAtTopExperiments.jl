abstract type ModelType end

# ------------------------------------------------------------------------------------------
# linear models
# ------------------------------------------------------------------------------------------
struct Linear <: ModelType end

parse_type(::Val{:Linear}) = Linear

function materialize(::AbstractNsf5, m::Linear)
    return Dense(22510 => 1; bias=false)
end

function materialize(::AbstractJMiPOD, m::Linear)
    return Chain(Flux.flatten, Dense(196608 => 1; bias=false))
end

# ------------------------------------------------------------------------------------------
# Efficient net
# ------------------------------------------------------------------------------------------
abstract type AbstractEfficientNet <: ModelType end

@kwdef struct EfficientNetB0 <: AbstractEfficientNet
    pretrained::Bool = false
end

parse_type(::Val{:EfficientNetB0}) = EfficientNetB0
efficientnet_type(::EfficientNetB0) = "efficientnet-b0"

function materialize(::AbstractJMiPOD, m::AbstractEfficientNet)
    return if m.pretrained
        tmp = EfficientNet.from_pretrained(efficientnet_type(m))
        inp = size(tmp.top.weight, 2)
        EfficientNet.EffNet(
            tmp.stem,
            tmp.blocks,
            tmp.head,
            tmp.pooling,
            Dense(inp => 1),
            tmp.stages,
            tmp.stages_channels,
        )
    else
        EfficientNet.EffNet(efficientnet_type(m); n_classes=1, in_channels=3)
    end
end

@kwdef struct GoogLeNet <: AbstractEfficientNet end

parse_type(::Val{:GoogLeNet}) = GoogLeNet

function materialize(::AbstractJMiPOD, ::GoogLeNet)
    return Metalhead.GoogLeNet(; nclasses=1)
end
