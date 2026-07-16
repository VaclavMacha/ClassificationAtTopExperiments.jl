abstract type ModelType end
abstract type DualModelType <: ModelType end

# ------------------------------------------------------------------------------------------
# linear models
# ------------------------------------------------------------------------------------------
struct Linear <: DualModelType end

parse_type(::Val{:Linear}) = Linear

function materialize(::AbstractNsf5, ::Linear)
    return Dense(22510 => 1; bias=false)
end

function materialize(::AbstractJMiPOD, ::Linear)
    return Chain(Flux.flatten, Dense(196608 => 1; bias=false))
end

function materialize(D::AbstractVision, ::Linear)
    return Chain(Flux.flatten, Dense(prod(obs_size(D)) => 1; bias=false))
end

function materialize(::AbstractEmber, ::Linear)
    return Dense(2381 => 1; bias=false)
end

materialize_dual(::Linear) = ClassificationAtTopDual.Linear()

# ------------------------------------------------------------------------------------------
# Gaussian
# ------------------------------------------------------------------------------------------
struct Gaussian <: DualModelType end

parse_type(::Val{:Gaussian}) = Gaussian
materialize_dual(::Gaussian) = ClassificationAtTopDual.Gaussian()

# ------------------------------------------------------------------------------------------
# Efficient net, GoogLeNet, MobileNetv3
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

@kwdef struct MobileNetv3 <: AbstractEfficientNet end

parse_type(::Val{:MobileNetv3}) = MobileNetv3

function materialize(::AbstractJMiPOD, ::MobileNetv3)
    return Metalhead.MobileNetv3(; nclasses=1)
end

# ------------------------------------------------------------------------------------------
# SRNet
# ------------------------------------------------------------------------------------------
@kwdef struct SRNet <: ModelType
    pretrained::Bool = false
    in_channels::Int = 3
    n_out::Int = 1
end

parse_type(::Val{:SRNet}) = SRNet

function materialize(::AbstractJMiPOD, m::SRNet)
    if m.pretrained
        # The pretrained checkpoint has a 2-logit head (cover, stego).
        net = build_srnet(; in_channels=m.in_channels, n_out=2)
        load_srnet_weights!(net, pretraineddir("srnet.h5"))
        m.n_out == 2 && return net
        m.n_out == 1 || error("SRNet: pretrained head has 2 outputs; n_out must be 1 or 2")
        # Collapse to a single score s = logit(stego) - logit(cover) — the
        # same decision function, compatible with the binary framework.
        fc = net.layers[end]
        head = Dense(fc.weight[2:2, :] .- fc.weight[1:1, :], fc.bias[2:2] .- fc.bias[1:1])
        return Chain(net.layers[1:end-1]..., head)
    else
        return build_srnet(; in_channels=m.in_channels, n_out=m.n_out)
    end
end

# ------------------------------------------------------------------------------------------
# Custom conv
# ------------------------------------------------------------------------------------------
struct SimpleConv <: ModelType end

parse_type(::Val{:SimpleConv}) = SimpleConv

function materialize(::AbstractVisionGray, ::SimpleConv)
    return Chain(
        Conv((5, 5), 1 => 20, relu; stride=(1, 1)),
        MaxPool((2, 2)),
        Conv((5, 5), 20 => 50, relu; stride=(1, 1)),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(800, 1)
    )
end

function materialize(::AbstractVisionColor, ::SimpleConv)
    return Chain(
        Conv((3, 3), 3 => 64, relu; pad=SamePad()),
        MaxPool((2, 2)),
        Conv((3, 3), 64 => 128, relu; pad=SamePad()),
        MaxPool((2, 2)),
        Conv((3, 3), 128 => 128, relu; pad=SamePad()),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(2048, 1)
    )
end
