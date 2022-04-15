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

@option struct DataConfig
    type::String = "Nsf5_01"
    full_dataset::Bool = false
    at_train::Float32 = 0.45
    at_valid::Float32 = 0.05
    cover_ratio::Float32 = 1
    stego_ratio::Float32 = 1
end

function Base.string(d::DataConfig)
    vals = dir_string.(
        (d.full_dataset, d.at_train, d.at_valid, d.cover_ratio, d.stego_ratio)
    )
    return "$(string(d.type))($(join(vals, ", ")))"
end

function split_data(n, at, ratio)
    train, valid, test = splitobs(shuffleobs(1:n); at)
    if 0 < ratio < 1
        train = train[1:round(Int64, ratio * length(train))]
    end
    return train, valid, test
end

function get_path(d::DataConfig)
    return get_path(Val(Symbol(d.type)), ifelse(d.full_dataset, "full","partial"))
end
get_path(::Val{:Nsf5_01}, dir) = datadir("dataset", dir, "nsf5_0.1_jrm.h5")
get_path(::Val{:Nsf5_02}, dir) = datadir("dataset", dir, "nsf5_0.2_jrm.h5")
get_path(::Val{:Nsf5_05}, dir) = datadir("dataset", dir, "nsf5_0.5_jrm.h5")

function load(d::DataConfig)
    x_cover = load_hdf5(datadir(
        "dataset",
        ifelse(d.full_dataset, "full","partial"),
        "cover_jrm.h5")
    )
    x_stego = load_hdf5(get_path(d))

    at = (d.at_train, d.at_valid)
    i1_train, i1_valid, i1_test = split_data(size(x_cover, 2), at, d.cover_ratio)
    i2_train, i2_valid, i2_test = split_data(size(x_stego, 2), at, d.stego_ratio)

    train = (
        hcat(x_cover[:, i1_train], x_stego[:, i2_train]),
        reshape(1:(length(i1_train)+length(i2_train)) .> length(i1_train), 1, :),
    )
    valid = (
        hcat(x_cover[:, i1_valid], x_stego[:, i2_valid]),
        reshape(1:(length(i1_valid)+length(i2_valid)) .> length(i1_valid), 1, :),
    )
    test = (
        hcat(x_cover[:, i1_test], x_stego[:, i2_test]),
        reshape(1:(length(i1_test)+length(i2_test)) .> length(i1_test), 1, :),
    )
    @info """
    Dataset: all (cover/stego)
    ⋅ Train: $(length(train[2])) ($(sum(.~train[2]))/$(sum(train[2])))
    ⋅ Valid: $(length(valid[2])) ($(sum(.~valid[2]))/$(sum(valid[2])))
    ⋅ Test: $(length(test[2])) ($(sum(.~test[2]))/$(sum(test[2])))
    """
    return train, valid, test
end