# SRNet — Deep Residual Network for Steganalysis (Boroumand et al., 2019)

# Type 2: identity residual block (same in/out channels)
# output = type1(x) |> convbn .+ x
struct SRNetType2
    type1
    convbn
end

Flux.@functor SRNetType2
(b::SRNetType2)(x) = b.convbn(b.type1(x)) .+ x

# Type 3: downsampling residual block (may change channels)
# main:     type1 → convbn → MeanPool(3×3, stride=2)
# shortcut: Conv(1×1, stride=2) → BN
struct SRNetType3
    shortcut_conv
    shortcut_bn
    type1
    convbn
    pool
end

Flux.@functor SRNetType3
function (b::SRNetType3)(x)
    shortcut = b.shortcut_bn(b.shortcut_conv(x))
    main = b.pool(b.convbn(b.type1(x)))
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
        MeanPool((3, 3); stride=2, pad=1),
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
        ks = sort(filter(k -> all(isdigit, k), keys(f)))
        length(ks) == length(ps) || error(
            "Parameter count mismatch: file has $(length(ks)) tensors, model has $(length(ps)).\n" *
            "Re-export with scripts_python/prepare_srnet_weights.py."
        )
        for (p, k) in zip(ps, ks)
            w = read(f, k)
            # HDF5.jl reverses numpy dim order on read; undo that per tensor type.
            if ndims(w) == 4 && size(w) != size(p)
                w = permutedims(w, (3, 4, 2, 1))  # (out,in,H,W) → (H,W,in,out)
                # PyTorch does cross-correlation, Flux.Conv flips kernels —
                # rotate spatial dims 180° to compensate.
                w = w[end:-1:1, end:-1:1, :, :]
            elseif ndims(w) == 2 && size(w) != size(p)
                w = permutedims(w, (2, 1))         # (in,out) → (out,in)
            end
            size(p) == size(w) || error("Shape mismatch at key $k: model=$(size(p)), file=$(size(w))")
            copyto!(p, w)
        end

        # BatchNorm running statistics (not part of Flux.params). Groups
        # "running_mean"/"running_var" hold one vector per BatchNorm, indexed
        # in depth-first model order — the same order as Flux.modules.
        if haskey(f, "running_mean")
            bns = filter(m -> m isa BatchNorm, Flux.modules(model))
            mks = sort(keys(f["running_mean"]))
            length(mks) == length(bns) || error(
                "BatchNorm count mismatch: file has $(length(mks)) stats, model has $(length(bns))."
            )
            for (bn, k) in zip(bns, mks)
                copyto!(bn.μ, read(f, "running_mean/$k"))
                copyto!(bn.σ², read(f, "running_var/$k"))
            end
        else
            @warn "No BatchNorm running statistics in $path — test-mode inference " *
                  "will use uninitialized stats. Re-export with prepare_srnet_weights.py."
        end
    end
    return model
end
