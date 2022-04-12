function load_hdf5(path::AbstractString)
    @info """
    Loading file: $(path)
    (it may take several minutes to load data into memory)
    """
    return h5open(path, "r") do fid
        read(fid, "data")
    end
end

@option "dataset" struct Dataset
    cover_file::String = "cover_jrm.h5"
    stego_file::String = "nsf5_0.1_jrm_partial.h5"
    cover_ratio::Float64 = 1
    stego_ratio::Float64 = 1
end

function split_in_half(x, ratio)
    n = size(x, 2)
    k = ratio < 1 ? round(Int64, ratio * n) : n
    inds = randperm(n)
    return inds[1:k÷2], inds[(k÷2+1):k]
end

function load(d::Dataset)
    x_cover = load_hdf5(datadir("dataset", d.cover_file))
    x_stego = load_hdf5(datadir("dataset", d.stego_file))

    i1_train, i1_test = split_in_half(x_cover, d.cover_ratio)
    i2_train, i2_test = split_in_half(x_stego, d.stego_ratio)

    xtrain = hcat(x_cover[:, i1_train], x_stego[:, i2_train])
    ytrain = reshape(1:size(xtrain, 2) .> length(i1_train), 1, :)
    xtest = hcat(x_cover[:, i1_test], x_stego[:, i2_test])
    ytest = reshape(1:size(xtest, 2) .> length(i1_test), 1, :)
    return (xtrain, ytrain), (xtest, ytest)
end
