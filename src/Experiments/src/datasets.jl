abstract type DatasetType end

struct LabeledDataset{F,T} <: SupervisedDataset
    features::F
    targets::T
end

flux_shape(x) = x
flux_shape(x::AbstractVector) = reshape(x, 1, :)
function flux_shape(x::AbstractArray{T, 3}) where {T}
    return reshape(x, size(x, 1), size(x, 2), 1, size(x, 3))
end

function _split_inds(inds, at::NTuple{2,AbstractFloat}, ratio::Real=1)
    train, valid, test = splitobs(inds; at, shuffle=true)
    if 0 < ratio < 1
        train = train[1:round(Int64, ratio * length(train))]
    end
    return train, valid, test
end

stego_ratio(y) = round(100 * sum(y .== 1) / length(y); digits=2)

function split_data(data, at, ratio)
    x, y = data
    cover = _split_inds(findall(vec(y) .== 0), at)
    stego = _split_inds(findall(vec(y) .== 1), at, ratio)

    train = vcat(cover[1], stego[1])
    valid = vcat(cover[2], stego[2])
    test = vcat(cover[3], stego[3])

    @info """
    Dataset:
    ⋅ Train: $(length(train)) ($(stego_ratio(y[train]))% stego)
    ⋅ Valid: $(length(valid)) ($(stego_ratio(y[valid]))% stego)
    ⋅ Test:  $(length(test)) ($(stego_ratio(y[test]))% stego)
    """
    return (
        LabeledDataset(flux_shape(obsview(x, train)), flux_shape(y[train])),
        LabeledDataset(flux_shape(obsview(x, valid)), flux_shape(y[valid])),
        LabeledDataset(flux_shape(obsview(x, test)), flux_shape(y[test])),
        LabeledDataset(obsview(x, train), reshape(y[train], 1, :)),
        LabeledDataset(obsview(x, valid), reshape(y[valid], 1, :)),
        LabeledDataset(obsview(x, test), reshape(y[test], 1, :)),
    )
end

# ------------------------------------------------------------------------------------------
# NSF5 datasets
# ------------------------------------------------------------------------------------------
abstract type AbstractNsf5 <: DatasetType end

obs_size(::AbstractNsf5) = (22510,)

@kwdef struct Nsf5 <: AbstractNsf5
    payload::Float64 = 0.2
    ratio::Float64 = 1.0
    at_train::Float64 = 0.45
    at_valid::Float64 = 0.05
end

parse_type(::Val{:Nsf5}) = Nsf5

@kwdef struct Nsf5Small <: AbstractNsf5
    payload::Float64 = 0.2
    ratio::Float64 = 1.0
    at_train::Float64 = 0.45
    at_valid::Float64 = 0.05
end

parse_type(::Val{:Nsf5Small}) = Nsf5Small

function load_hdf5(path::AbstractString)
    @info """
    Loading Data:
    ⋅ File: $(path)
    ⋅ It may take several minutes to load data into memory
    """
    return h5open(path, "r") do fid
        read(fid, "data")
    end
end

nsf5dir(args...) = datasetsdir("Nsf5", args...)
cover_path(::Nsf5Small) = nsf5dir("partial", "cover_jrm.h5")
cover_path(::Nsf5) = nsf5dir("full", "cover_jrm.h5")
stego_path(d::Nsf5Small) = nsf5dir("partial", "nsf5_$(d.payload)_jrm.h5")
stego_path(d::Nsf5) = nsf5dir("full", "nsf5_$(d.payload)_jrm.h5")

function load(d::AbstractNsf5)
    x_cover = load_hdf5(cover_path(d))
    x_stego = load_hdf5(stego_path(d))
    x = hcat(x_cover, x_stego)
    y = (1:size(x, 2)) .> size(x_cover, 2)

    return split_data((x, y), (d.at_train, d.at_valid), d.ratio)
end

# ------------------------------------------------------------------------------------------
# NSF5 datasets
# ------------------------------------------------------------------------------------------
abstract type AbstractJMiPOD <: DatasetType end

obs_size(::AbstractJMiPOD) = (256, 256, 3)

@kwdef struct JMiPOD <: AbstractJMiPOD
    payload::Float64 = 0.4
    ratio::Float64 = 1.0
    at_train::Float64 = 0.45
    at_valid::Float64 = 0.05
end

parse_type(::Val{:JMiPOD}) = JMiPOD
get_ids(::JMiPOD) = 1:47807

@kwdef struct JMiPODSmall <: AbstractJMiPOD
    payload::Float64 = 0.4
    ratio::Float64 = 1.0
    at_train::Float64 = 0.45
    at_valid::Float64 = 0.05
end

parse_type(::Val{:JMiPODSmall}) = JMiPODSmall
get_ids(::JMiPODSmall) = setdiff(1:111, 6:10:116)

jmipoddir(args...) = datasetsdir("JMiPOD", args...)
actordir(id::Int, args...) = jmipoddir("actor$(lpad(id, 5, "0"))", args...)
list_jpgs(dir) = filter(file -> endswith(file, ".jpg"), readdir(dir; join=true))

function load_image(path)
    img = jpeg_decode(path)
    return PermutedDimsArray(channelview(img), (3, 2, 1))
end

function load(d::AbstractJMiPOD)
    ids = get_ids(d)

    x_cover = String[]
    x_stego = String[]
    for id in ids
        cover = list_jpgs(actordir(id))
        append!(x_cover, cover)

        type = lpad(round(Int, 100 * d.payload), 3, "0")
        stego = list_jpgs(actordir(id, "stego-$(type)"))
        append!(x_stego, stego)
    end

    x = FileDataset(load_image, vcat(x_cover, x_stego))
    y = (1:length(x)) .> length(x_cover)

    return split_data((x, y), (d.at_train, d.at_valid), d.ratio)
end
