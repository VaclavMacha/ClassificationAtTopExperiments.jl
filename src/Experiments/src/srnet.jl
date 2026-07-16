# SRNet — Deep Residual Network for Steganalysis (Boroumand et al., 2019)

# Type 2: identity residual block (same in/out channels)
# output = type1(x) |> convbn .+ x
struct SRNetType2
    type1
    convbn
end

Flux.@layer SRNetType2
(b::SRNetType2)(x) = b.convbn(b.type1(x)) .+ x

# Type 3: downsampling residual block (may change channels)
# main:     type1 → convbn → MeanPool(3×3, stride=2)
# shortcut: Conv(1×1, stride=2) → BN
struct SRNetType3
    shortcut_conv
    shortcut_bn
    type1
    convbn
end

Flux.@layer SRNetType3
function (b::SRNetType3)(x)
    shortcut = b.shortcut_bn(b.shortcut_conv(x))
    main = MeanPool((3, 3); stride=2, pad=1)(b.convbn(b.type1(x)))
    return shortcut .+ main
end

function srnet_convbn(in_ch, out_ch)
    Chain(Conv((3, 3), in_ch => out_ch; bias=false, pad=1), BatchNorm(out_ch))
end

function srnet_type1(in_ch, out_ch)
    Chain(srnet_convbn(in_ch, out_ch), relu)
end

function srnet_type2(ch)
    SRNetType2(srnet_type1(ch, ch), srnet_convbn(ch, ch))
end

function srnet_type3(in_ch, out_ch)
    SRNetType3(
        Conv((1, 1), in_ch => out_ch; bias=false, stride=2),
        BatchNorm(out_ch),
        srnet_type1(in_ch, out_ch),
        srnet_convbn(out_ch, out_ch),
    )
end

function build_srnet(; in_channels=1, n_out=1)
    return Chain(
        # type1s
        srnet_type1(in_channels, 64),
        srnet_type1(64, 16),
        # type2s ×5
        srnet_type2(16),
        srnet_type2(16),
        srnet_type2(16),
        srnet_type2(16),
        srnet_type2(16),
        # type3s
        srnet_type3(16, 16),
        srnet_type3(16, 64),
        srnet_type3(64, 128),
        srnet_type3(128, 256),
        # type4: type1 → convbn → GlobalMeanPool
        srnet_type1(256, 512),
        srnet_convbn(512, 512),
        GlobalMeanPool(),
        Flux.flatten,
        Dense(512 => n_out),
    )
end

# Load weights from an HDF5 file produced by scripts_python/prepare_srnet_weights.py.
# Keys are zero-padded integers ("0000", "0001", ...) in the exact Flux.params order.
function load_srnet_weights!(model, path::AbstractString)
    h5open(path, "r") do f
        ps = Flux.params(model)
        ks = sort(keys(f))
        length(ks) == length(ps) || error(
            "Parameter count mismatch: file has $(length(ks)) tensors, model has $(length(ps)).\n" *
            "Re-export with scripts_python/prepare_srnet_weights.py."
        )
        for (p, k) in zip(ps, ks)
            w = read(f, k)
            size(p) == size(w) || error("Shape mismatch at key $k: model=$(size(p)), file=$(size(w))")
            copyto!(p, w)
        end
    end
    return model
end
