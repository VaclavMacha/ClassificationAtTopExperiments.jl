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

function hdf5_size(path::AbstractString)
    return h5open(path, "r") do fid
        size(fid["data"])
    end
end

@option struct DataConfig
    type::String = "Nsf5_01"
    full_dataset::Bool = false
    at_train::Float64 = 0.45
    at_valid::Float64 = 0.05
    cover_ratio::Float64 = 1
    stego_ratio::Float64 = 1
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

function get_path(d::DataConfig; cover::Bool = false)
    type = cover ? :Cover : Symbol(d.type)
    return get_path(Val(type), ifelse(d.full_dataset, "full","partial"))
end
get_path(::Val{:Cover}, dir) = datadir("dataset", dir, "cover_jrm.h5")
get_path(::Val{:Nsf5_01}, dir) = datadir("dataset", dir, "nsf5_0.1_jrm.h5")
get_path(::Val{:Nsf5_02}, dir) = datadir("dataset", dir, "nsf5_0.2_jrm.h5")
get_path(::Val{:Nsf5_05}, dir) = datadir("dataset", dir, "nsf5_0.5_jrm.h5")

function load(d::DataConfig)
    file_cover = get_path(d; cover=true)
    file_stego = get_path(d)

    s_cover = hdf5_size(file_cover)
    s_stego = hdf5_size(file_stego)
    k = s_cover[1]
    n = s_cover[2] + s_stego[2]

    x = Array{Float32}(undef, k, n)
    x[:, 1:s_cover[2]] = load_hdf5(file_cover)
    x[:, (s_cover[2]+1):end] = load_hdf5(file_stego)
    y = reshape((1:n) .> s_cover[2], 1, :)

    # split
    at = (d.at_train, d.at_valid)
    i1_train, i1_valid, i1_test = split_data(s_cover[2], at, d.cover_ratio)
    i2_train, i2_valid, i2_test = split_data(s_stego[2], at, d.stego_ratio)

    i_train = vcat(i1_train, i2_train .+ s_stego[2])
    i_valid = vcat(i1_valid, i2_valid .+ s_stego[2])
    i_test = vcat(i1_test, i2_test .+ s_stego[2])

    train = @views (x[:, i_train], y[:, i_train])
    valid = @views (x[:, i_valid], y[:, i_valid])
    test = @views (x[:, i_test], y[:, i_test])

    @info """
    Dataset: all (cover/stego)
    ⋅ Train: $(length(train[2])) ($(sum(.~train[2]))/$(sum(train[2])))
    ⋅ Valid: $(length(valid[2])) ($(sum(.~valid[2]))/$(sum(valid[2])))
    ⋅ Test: $(length(test[2])) ($(sum(.~test[2]))/$(sum(test[2])))
    """
    return train, valid, test
end