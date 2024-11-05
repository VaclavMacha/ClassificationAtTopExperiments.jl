flux_shape(x) = x
flux_shape(x::AbstractVector) = reshape(x, 1, :)
function flux_shape(x::AbstractArray{T,3}) where {T}
    return reshape(x, size(x, 1), size(x, 2), 1, size(x, 3))
end

dual_shape(x::AbstractVector) = reshape(x, 1, :)
dual_shape(x::AbstractMatrix) = x
dual_shape(x::AbstractArray) = Matrix(Flux.flatten(x))

function _split_inds(inds, at::NTuple{2,AbstractFloat}, ratio::Real=1)
    train, valid, test = splitobs(inds; at, shuffle=true)
    if 0 < ratio < 1
        train = train[1:round(Int64, ratio * length(train))]
    end
    return train, valid, test
end

# ------------------------------------------------------------------------------------------
# DatasetType
# ------------------------------------------------------------------------------------------
abstract type DatasetType end

load_with_threads(::DatasetType) = false
round_perc(val, digits=2) = round(100 * val; digits)

function summary_dataset(d::DatasetType)
    train, valid, test = load(d)
    return summary_dataset(d, train.targets, valid.targets, test.targets)
end

function summary_dataset(d::DatasetType, y_train, y_valid, y_test)
    return DataFrame(
        dataset=_string(d),
        size=join(obs_size(d), "x"),
        train=length(y_train),
        train_ratio=round_perc(sum(y_train .== 1) / length(y_train)),
        valid=length(y_valid),
        valid_ratio=round_perc(sum(y_valid .== 1) / length(y_valid)),
        test=length(y_test),
        test_ratio=round_perc(sum(y_test .== 1) / length(y_test)),
    )
end

function summary_dataset(ds::DatasetType...)
    return vcat([summary_dataset(d) for d in ds]...; cols=:union)
end

# ------------------------------------------------------------------------------------------
# NSF5 datasets
# ------------------------------------------------------------------------------------------
abstract type AbstractNsf5 <: DatasetType end

obs_size(::AbstractNsf5) = (22510,)

@kwdef struct Nsf5 <: AbstractNsf5
    payload::Float64 = 0.2
    ratio::Float64 = 1.0
    at_train::Float64 = 0.375
    at_valid::Float64 = 0.125
end

parse_type(::Val{:Nsf5}) = Nsf5

@kwdef struct Nsf5Small <: AbstractNsf5
    payload::Float64 = 0.2
    ratio::Float64 = 1.0
    at_train::Float64 = 0.375
    at_valid::Float64 = 0.125
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

function hdf5_data_size(path)
    return h5open(path, "r") do fid
        size(fid["data"])
    end
end

nsf5dir(args...) = datasetsdir("Nsf5", args...)
cover_path(::Nsf5Small) = nsf5dir("partial", "cover_jrm.h5")
cover_path(::Nsf5) = nsf5dir("full", "cover_jrm.h5")
stego_path(d::Nsf5Small) = nsf5dir("partial", "nsf5_$(d.payload)_jrm.h5")
stego_path(d::Nsf5) = nsf5dir("full", "nsf5_$(d.payload)_jrm.h5")

function load(d::AbstractNsf5, reshape_func=flux_shape)
    x_cover = load_hdf5(cover_path(d))
    x_stego = load_hdf5(stego_path(d))

    # split data
    itr, ivl, its = _split_inds(1:numobs(x_cover), (d.at_train, d.at_valid))
    jtr, jvl, jts = _split_inds(1:numobs(x_stego), (d.at_train, d.at_valid), d.ratio)

    x_train = @views hcat(x_cover[:, itr], x_stego[:, jtr])
    y_train = (1:numobs(x_train)) .> length(itr)

    x_valid = @views hcat(x_cover[:, ivl], x_stego[:, jvl])
    y_valid = (1:numobs(x_valid)) .> length(ivl)

    x_test = @views hcat(x_cover[:, its], x_stego[:, jts])
    y_test = (1:numobs(x_test)) .> length(its)

    return (
        ArrayDataset(reshape_func(x_train), reshape_func(y_train)),
        ArrayDataset(reshape_func(x_valid), reshape_func(y_valid)),
        ArrayDataset(reshape_func(x_test), reshape_func(y_test)),
    )
end

function summary_dataset(d::AbstractNsf5)
    ~, n_cover = hdf5_data_size(cover_path(d))
    ~, n_stego = hdf5_data_size(stego_path(d))

    # split data
    itr, ivl, its = _split_inds(1:n_cover, (d.at_train, d.at_valid))
    jtr, jvl, jts = _split_inds(1:n_stego, (d.at_train, d.at_valid), d.ratio)

    y_train = flux_shape((1:(length(itr)+length(jtr))) .> length(itr))
    y_valid = flux_shape((1:(length(ivl)+length(jvl))) .> length(ivl))
    y_test = flux_shape((1:(length(its)+length(jts))) .> length(its))

    return summary_dataset(d, y_train, y_valid, y_test)
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
    at_train::Float64 = 0.375
    at_valid::Float64 = 0.125
end

parse_type(::Val{:JMiPOD}) = JMiPOD
get_ids(::JMiPOD) = 1:47807

@kwdef struct JMiPODSmall <: AbstractJMiPOD
    payload::Float64 = 0.4
    ratio::Float64 = 1.0
    at_train::Float64 = 0.375
    at_valid::Float64 = 0.125
end

parse_type(::Val{:JMiPODSmall}) = JMiPODSmall
get_ids(::JMiPODSmall) = setdiff(1:111, 6:10:116)

@kwdef struct JMiPODDebug <: AbstractJMiPOD
    payload::Float64 = 0.1
    ratio::Float64 = 1.0
    at_train::Float64 = 0.375
    at_valid::Float64 = 0.125
end

parse_type(::Val{:JMiPODDebug}) = JMiPODDebug
get_ids(::JMiPODDebug) = 1:1

# jmipoddir(args...) = datasetsdir("JMiPOD", args...)
jmipoddir(args...) = datasetsdir("JMiPOD_julia", args...)
actordir(id::Int, args...) = jmipoddir("actor$(lpad(id, 5, "0"))", args...)
list_jpgs(dir) = filter(file -> endswith(file, ".jpg"), readdir(dir; join=true))

load_image(path) = image_to_array(jpeg_decode(path))

function image_to_array(img::AbstractMatrix{<:RGB})
    return PermutedDimsArray(channelview(img), (3, 2, 1))
end

function image_to_array(img::AbstractMatrix{<:Gray})
    return image_to_array(convert.(RGB, img))
end


function load(d::AbstractJMiPOD)
    ids = get_ids(d)

    x_cover = String[]
    x_stego = String[]
    for id in ids
        isdir(actordir(id)) || continue
        cover = list_jpgs(actordir(id))
        append!(x_cover, cover)

        type = lpad(round(Int, 100 * d.payload), 3, "0")

        isdir(actordir(id, "stego-$(type)")) || continue
        stego = list_jpgs(actordir(id, "stego-$(type)"))
        append!(x_stego, stego)
    end

    x = vcat(x_cover, x_stego)
    y = flux_shape((1:length(x)) .> length(x_cover))

    # split data
    cover = _split_inds(findall(vec(y) .== 0), (d.at_train, d.at_valid))
    stego = _split_inds(findall(vec(y) .== 1), (d.at_train, d.at_valid), d.ratio)

    train = vcat(cover[1], stego[1])
    valid = vcat(cover[2], stego[2])
    test = vcat(cover[3], stego[3])

    shape = obs_size(d)

    return (
        FileDataset(load_image, shape, x[train], flux_shape(y[train]), true),
        FileDataset(load_image, shape, x[valid], flux_shape(y[valid]), true),
        FileDataset(load_image, shape, x[test], flux_shape(y[test]), true),
    )
end

function load(d::JMiPODDebug)
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

    x = vcat(x_cover, x_stego)
    y = flux_shape((1:length(x)) .> length(x_cover))

    # split data
    cover = _split_inds(findall(vec(y) .== 0), (d.at_train, d.at_valid))
    stego = _split_inds(findall(vec(y) .== 1), (d.at_train, d.at_valid), d.ratio)

    train = vcat(cover[1], stego[1])
    valid = vcat(cover[2], stego[2])
    test = vcat(cover[3], stego[3])

    shape = obs_size(d)
    dtr = getobs(FileDataset(load_image, shape, x[train], flux_shape(y[train]), true))
    dva = getobs(FileDataset(load_image, shape, x[valid], flux_shape(y[valid]), true))
    dts = getobs(FileDataset(load_image, shape, x[test], flux_shape(y[test]), true))

    return (
        ArrayDataset(flux_shape(dtr.features), flux_shape(dtr.targets)),
        ArrayDataset(flux_shape(dva.features), flux_shape(dva.targets)),
        ArrayDataset(flux_shape(dts.features), flux_shape(dts.targets)),
    )
end

# ------------------------------------------------------------------------------------------
# Vision datasets from MLDatasets
# ------------------------------------------------------------------------------------------
abstract type AbstractVision <: DatasetType end
abstract type AbstractVisionGray <: AbstractVision end
abstract type AbstractVisionColor <: AbstractVision end

binarize(y, pos_labels) = in.(y, Ref(pos_labels))

function load(d::AbstractVision, reshape_func=flux_shape)
    train = load_dataset(d, :train)
    itrain, ivalid = splitobs(1:numobs(train); at=d.at_train, shuffle=true)
    test = load_dataset(d, :test)
    pos = d.pos_labels

    x_train = obsview(train.features, itrain)
    y_train = obsview(train.targets, itrain)
    x_valid = obsview(train.features, ivalid)
    y_valid = obsview(train.targets, ivalid)

    return (
        ArrayDataset(reshape_func(x_train), reshape_func(binarize(y_train, pos))),
        ArrayDataset(reshape_func(x_valid), reshape_func(binarize(y_valid, pos))),
        ArrayDataset(reshape_func(test.features), reshape_func(binarize(test.targets, pos))),
    )
end

@kwdef struct MNIST <: AbstractVisionGray
    pos_labels::Vector{Int} = [1]
    at_train::Float64 = 0.75
end

obs_size(::MNIST) = (28, 28, 1)
parse_type(::Val{:MNIST}) = MNIST
load_dataset(::MNIST, key::Symbol) = MLDatasets.MNIST(Float32, key)

@kwdef struct FashionMNIST <: AbstractVisionGray
    pos_labels::Vector{Int} = [1]
    at_train::Float64 = 0.75
end

obs_size(::FashionMNIST) = (28, 28, 1)
parse_type(::Val{:FashionMNIST}) = FashionMNIST
load_dataset(::FashionMNIST, key::Symbol) = MLDatasets.FashionMNIST(Float32, key)

@kwdef struct CIFAR10 <: AbstractVisionColor
    pos_labels::Vector{Int} = [1]
    at_train::Float64 = 0.75
end

obs_size(::CIFAR10) = (32, 32, 3)
parse_type(::Val{:CIFAR10}) = CIFAR10
load_dataset(::CIFAR10, key::Symbol) = MLDatasets.CIFAR10(Float32, key)

@kwdef struct CIFAR20 <: AbstractVisionColor
    pos_labels::Vector{Int} = [1]
    at_train::Float64 = 0.75
end

obs_size(::CIFAR20) = (32, 32, 3)
parse_type(::Val{:CIFAR20}) = CIFAR20
function load_dataset(::CIFAR20, key::Symbol)
    data = MLDatasets.CIFAR100(Float32, key)
    return ArrayDataset(data.features, flux_shape(data.targets.coarse))
end

@kwdef struct CIFAR100 <: AbstractVisionColor
    pos_labels::Vector{Int} = [1]
    at_train::Float64 = 0.75
end

obs_size(::CIFAR100) = (32, 32, 3)
parse_type(::Val{:CIFAR100}) = CIFAR100
function load_dataset(::CIFAR100, key::Symbol)
    data = MLDatasets.CIFAR100(Float32, key)
    return ArrayDataset(data.features, flux_shape(data.targets.fine))
end

@kwdef struct SVHN2 <: AbstractVisionColor
    pos_labels::Vector{Int} = [1]
    at_train::Float64 = 0.75
end

obs_size(::SVHN2) = (32, 32, 3)
parse_type(::Val{:SVHN2}) = SVHN2
load_dataset(::SVHN2, key::Symbol) = MLDatasets.SVHN2(Float32, key)

@kwdef struct SVHN2Extra <: AbstractVisionColor
    pos_labels::Vector{Int} = [1]
    at_train::Float64 = 0.75
end

obs_size(::SVHN2Extra) = (32, 32, 3)
parse_type(::Val{:SVHN2Extra}) = SVHN2Extra
function load_dataset(::SVHN2Extra, key::Symbol)
    return if key == :train
        d1 = MLDatasets.SVHN2(Float32, :train)
        d2 = MLDatasets.SVHN2(Float32, :extra)

        x = cat(d1.features, d2.features; dims=4)
        y = vcat(d1.targets, d2.targets)
        return ArrayDataset(x, flux_shape(y))
    else
        MLDatasets.SVHN2(Float32, :test)
    end
end

# ------------------------------------------------------------------------------------------
# Ember
# ------------------------------------------------------------------------------------------
abstract type AbstractEmber <: DatasetType end

obs_size(::AbstractEmber) = (2381,)

@kwdef struct Ember <: AbstractEmber
    pos_labels::Vector{Int} = [1]
    at_train::Float64 = 0.75
end

parse_type(::Val{:Ember}) = Ember
emberdir(args...) = datasetsdir("Ember", args...)

function load(d::AbstractEmber, reshape_func=flux_shape)
    train = h5open(emberdir("train.h5"), "r") do fid
        return (features = read(fid, "features"), targets = read(fid, "labels"))
    end
    itrain, ivalid = splitobs(1:numobs(train); at=d.at_train, shuffle=true)
    test = h5open(emberdir("test.h5"), "r") do fid
        return (features = read(fid, "features"), targets = read(fid, "labels"))
    end

    pos = d.pos_labels

    x_train = obsview(train.features, itrain)
    y_train = obsview(train.targets, itrain)
    x_valid = obsview(train.features, ivalid)
    y_valid = obsview(train.targets, ivalid)

    return (
        ArrayDataset(reshape_func(x_train), reshape_func(binarize(y_train, pos))),
        ArrayDataset(reshape_func(x_valid), reshape_func(binarize(y_valid, pos))),
        ArrayDataset(reshape_func(test.features), reshape_func(binarize(test.targets, pos))),
    )
end
