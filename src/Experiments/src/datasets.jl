function load_hdf5(path::AbstractString)
    @info """
    Loading file: $(path)
    (it may take several minutes to load data into memory)
    """
    return h5open(path, "r") do fid
        read(fid, "data")
    end
end

abstract type DatasetType end

@option "Nsf5_01" struct Nsf5_01 <: DatasetType
    ratio::Float64 = 1
end

@option "Nsf5_02" struct Nsf5_02 <: DatasetType
    ratio::Float64 = 1
end

@option "Nsf5_05" struct Nsf5_05 <: DatasetType
    ratio::Float64 = 1
end

@option struct DataConfig
    dataset::Union{Nsf5_01,Nsf5_02,Nsf5_05}
end

function split_in_half(x, ratio)
    n = size(x, 2)
    k = ratio < 1 ? round(Int64, ratio * n) : n
    inds = randperm(n)
    return inds[1:k÷2], inds[(k÷2+1):k]
end

get_path(d::DataConfig) = get_path(d.dataset)
get_path(::Nsf5_01) = datadir("dataset", "nsf5_0.1_jrm.h5")
get_path(::Nsf5_02) = datadir("dataset", "nsf5_0.2_jrm.h5")
get_path(::Nsf5_05) = datadir("dataset", "nsf5_0.5_jrm.h5")

function load(d::D) where {D<:DataConfig}
    x_cover = load_hdf5(datadir("dataset", "cover_jrm.h5"))
    x_stego = load_hdf5(get_path(d))

    i1_train, i1_test = split_in_half(x_cover, 1)
    i2_train, i2_test = split_in_half(x_stego, d.dataset.ratio)

    xtrain = hcat(x_cover[:, i1_train], x_stego[:, i2_train])
    ytrain = reshape(1:size(xtrain, 2) .> length(i1_train), 1, :)
    xtest = hcat(x_cover[:, i1_test], x_stego[:, i2_test])
    ytest = reshape(1:size(xtest, 2) .> length(i1_test), 1, :)
    return (xtrain, ytrain), (xtest, ytest)
end
