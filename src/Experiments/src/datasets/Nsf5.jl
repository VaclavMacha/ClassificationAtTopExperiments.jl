abstract type AbstractNsf5 end

@option "Nsf5" struct Nsf5 <: AbstractNsf5
    payload::Float64 = 0.2
    ratio::Float64 = 0.1
    at_train::Float64 = 0.45
    at_valid::Float64 = 0.05
end

@option "Nsf5Small" struct Nsf5Small <: AbstractNsf5
    payload::Float64 = 0.2
    ratio::Float64 = 0.1
    at_train::Float64 = 0.45
    at_valid::Float64 = 0.05
end

function Base.string(d::D) where {D<:AbstractNsf5}
    vals = dir_string.((d.payload, d.ratio, d.at_train, d.at_valid))
    return "$(D.name.name)($(join(vals, ", ")))"
end

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

function split_data(n, at, ratio = 1)
    train, valid, test = splitobs(shuffleobs(1:n); at)
    if 0 < ratio < 1
        train = train[1:round(Int64, ratio * length(train))]
    end
    return train, valid, test
end

nsf5dir(args...) = datadir("dataset", args...)

get_dir(::Nsf5) = "full"
get_dir(::Nsf5Small) = "partial"

function get_path(d::AbstractNsf5; cover::Bool = false)
    dir = get_dir(d)
    if cover
        return nsf5dir(dir, "cover_jrm.h5")
    else
        return nsf5dir(dir, "nsf5_$(d.payload)_jrm.h5")
    end
end

function load(d::AbstractNsf5)
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
    i1_train, i1_valid, i1_test = split_data(s_cover[2], at)
    i2_train, i2_valid, i2_test = split_data(s_stego[2], at, d.ratio)

    i_train = vcat(i1_train, i2_train .+ s_cover[2])
    i_valid = vcat(i1_valid, i2_valid .+ s_cover[2])
    i_test = vcat(i1_test, i2_test .+ s_cover[2])

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
