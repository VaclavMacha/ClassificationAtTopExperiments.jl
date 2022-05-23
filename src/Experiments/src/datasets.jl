struct LabeledDataset{F,T} <: SupervisedDataset
    features::F
    targets::T
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
    return LabeledDataset(ObsView(x, train), y[train]),
    LabeledDataset(ObsView(x, valid), y[valid]),
    LabeledDataset(ObsView(x, test), y[test])
end

# NSF5 datasets
abstract type DatasetType end
abstract type AbstractStegoData <: DatasetType end
abstract type AbstractNsf5 <: AbstractStegoData end

@kwdef struct Nsf5 <: AbstractNsf5
    payload::Float64 = 0.2
    ratio::Float64 = 1.0
    at_train = 0.45
    at_valid::Float64 = 0.05
end

parse_type(::Val{:Nsf5}) = Nsf5

@kwdef struct Nsf5Small <: AbstractNsf5
    payload::Float64 = 0.2
    ratio::Float64 = 1.0
    at_train = 0.45
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

nsf5dir(args...) = datadir("dataset", args...)
cover_path(::Nsf5Small) = nsf5dir("partial", "cover_jrm.h5")
cover_path(::Nsf5) = nsf5dir("full", "cover_jrm.h5")
stego_path(::Nsf5Small) = nsf5dir("partial", "cover_jrm.h5")
stego_path(::Nsf5) = nsf5dir("full", "cover_jrm.h5")

function load(d::AbstractNsf5)
    x_cover = load_hdf5(cover_path(d))
    x_stego = load_hdf5(stego_path(d))
    x = hcat(x_cover, x_stego)
    y = (1:size(x, 2)) .> size(x_cover, 2)

    return split_data((x, y), (d.at_train, d.at_valid), d.ratio)
end

# JMiPOD datasets
abstract type AbstractJMiPOD <: AbstractStegoData end

@kwdef struct JMiPOD <: AbstractJMiPOD
    payload::Float64 = 0.4
    ratio::Float64 = 1.0
    at_train = 0.45
    at_valid::Float64 = 0.05
end

parse_type(::Val{:JMiPOD}) = JMiPOD
get_ids(::JMiPOD) = 1:47807

@kwdef struct JMiPODSmall <: AbstractJMiPOD
    payload::Float64 = 0.4
    ratio::Float64 = 1.0
    at_train = 0.45
    at_valid::Float64 = 0.05
end

parse_type(::Val{:JMiPODSmall}) = JMiPODSmall
get_ids(::JMiPODSmall) = setdiff(1:111, 6:10:116)

jmipoddir(args...) = datadir("dataset", "jmipod", args...)
actordir(id::Int, args...) = jmipoddir("actor$(lpad(id, 5, "0"))", args...)
list_jpgs(dir) = filter(file -> endswith(file, ".jpg"), readdir(dir; join=true))

function load_image(path)
    img = Float32.(channelview(jpeg_decode(path)))
    return permutedims(img, (3, 2, 1))
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
