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
    at_train::Float32 = 0.45
    at_valid::Float32 = 0.05
    cover_ratio::Float32 = 1
    stego_ratio::Float32 = 1
end

function Base.string(d::DataConfig)
    vals = string.([d.at_train, d.at_valid, d.cover_ratio, d.stego_ratio])
    return "$(string(d.type))($(join(vals, ", ")))"
end

function split_data(n, at, ratio)
    train, valid, test = splitobs(shuffleobs(1:n); at)
    if 0 < ratio < 1
        train = train[1:round(Int64, ratio * length(train))]
    end
    return train, valid, test
end

get_path(d::DataConfig) = get_path(Val(Symbol(d.type)))
get_path(::Val{:Nsf5_01}) = datadir("dataset", "nsf5_0.1_jrm.h5")
get_path(::Val{:Nsf5_02}) = datadir("dataset", "nsf5_0.2_jrm.h5")
get_path(::Val{:Nsf5_05}) = datadir("dataset", "nsf5_0.5_jrm.h5")

function load(d::D) where {D<:DataConfig}
    x_cover = load_hdf5(datadir("dataset", "cover_jrm.h5"))
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
    @info"""
    Dataset:
    ⋅ File: $(size(train[1]))
    ⋅ Valid: $(size(valid[1]))
    ⋅ Test: $(size(test[1]))
    """
    return train, valid, test
end