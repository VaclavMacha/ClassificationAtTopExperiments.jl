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
    )
end

_binarize(y::Real, pos_labels) = y in pos_labels
binarize(y, pos_labels) = _binarize.(y, Ref(pos_labels))

function split_data(train, test, at, pos_labels)
    inds_train, inds_valid = splitobs(1:numobs(train); at, shuffle=true)

    x_train = obsview(train.features, inds_train)
    y_train = binarize(train.targets[inds_train], pos_labels)

    x_valid = obsview(train.features, inds_valid)
    y_valid = binarize(train.targets[inds_valid], pos_labels)

    x_test = test.features
    y_test = binarize(test.targets, pos_labels)

    @info """
    Dataset:
    ⋅ Train: $(numobs(y_train))
    ⋅ Valid: $(numobs(y_valid))
    ⋅ Test:  $(numobs(y_test))
    """
    return (
        LabeledDataset(flux_shape(x_train), flux_shape(y_train)),
        LabeledDataset(flux_shape(x_valid), flux_shape(y_valid)),
        LabeledDataset(flux_shape(x_test), flux_shape(y_test)),
    )
end

# ------------------------------------------------------------------------------------------
# NSF5 datasets
# ------------------------------------------------------------------------------------------
abstract type DatasetType end

load_with_threads(::DatasetType) = false

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

load_with_threads(::AbstractJMiPOD) = true

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

# ------------------------------------------------------------------------------------------
# Vision datasets from MLDatasets
# ------------------------------------------------------------------------------------------
abstract type AbstractVision <: DatasetType end

function load(d::AbstractVision)
    train = load_dataset(d, :train)
    test = load_dataset(d, :test)
    return split_data(train, test, d.at_train, d.pos_labels)
end

@kwdef struct MNIST <: AbstractVision
    pos_labels::Vector{Int} = [0]
    at_train::Float64 = 0.9
end

obs_size(::MNIST) = (28, 28, 1)
parse_type(::Val{:MNIST}) = MNIST
load_dataset(::MNIST, key::Symbol) = MLDatasets.MNIST(Float32, key)

@kwdef struct FashionMNIST <: AbstractVision
    pos_labels::Vector{Int} = [0]
    at_train::Float64 = 0.9
end

obs_size(::FashionMNIST) = (28, 28, 1)
parse_type(::Val{:FashionMNIST}) = FashionMNIST
load_dataset(::FashionMNIST, key::Symbol) = MLDatasets.FashionMNIST(Float32, key)

@kwdef struct CIFAR10 <: AbstractVision
    pos_labels::Vector{Int} = [0]
    at_train::Float64 = 0.9
end

obs_size(::CIFAR10) = (32, 32, 3)
parse_type(::Val{:CIFAR10}) = CIFAR10
load_dataset(::CIFAR10, key::Symbol) = MLDatasets.CIFAR10(Float32, key)

@kwdef struct CIFAR20 <: AbstractVision
    pos_labels::Vector{Int} = [0]
    at_train::Float64 = 0.9
end

obs_size(::CIFAR20) = (32, 32, 3)
parse_type(::Val{:CIFAR20}) = CIFAR100
function load_dataset(::CIFAR20, key::Symbol)
    data = MLDatasets.CIFAR100(Float32, key)
    return LabeledDataset(data.features, data.targets.coarse)
end

@kwdef struct CIFAR100 <: AbstractVision
    pos_labels::Vector{Int} = [0]
    at_train::Float64 = 0.9
end

obs_size(::CIFAR100) = (32, 32, 3)
parse_type(::Val{:CIFAR100}) = CIFAR100
function load_dataset(::CIFAR100, key::Symbol)
    data = MLDatasets.CIFAR100(Float32, key)
    return LabeledDataset(data.features, data.targets.fine)
end

@kwdef struct SVHN2 <: AbstractVision
    pos_labels::Vector{Int} = [0]
    at_train::Float64 = 0.9
end

obs_size(::SVHN2) = (32, 32, 3)
parse_type(::Val{:SVHN2}) = SVHN2
load_dataset(::SVHN2, key::Symbol) = MLDatasets.SVHN2(Float32, key)

@kwdef struct SVHN2Extra <: AbstractVision
    pos_labels::Vector{Int} = [0]
    at_train::Float64 = 0.9
end

obs_size(::SVHN2Extra) = (32, 32, 3)
parse_type(::Val{:SVHN2Extra}) = SVHN2
function  load_dataset(::SVHN2Extra, key::Symbol)
    return if key == :train
        d1 = MLDatasets.SVHN2(Float32, :train)
        d2 = MLDatasets.SVHN2(Float32, :extra)

        x = cat(d1.features, d2.features; dims = 4)
        y = vcat(d1.targets, d2.targets)
        return LabeledDataset(x, y)
    else
        MLDatasets.SVHN2(Float32, :test)
    end
end
