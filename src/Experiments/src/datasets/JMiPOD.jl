abstract type AbstractJMiPOD end

@option "JMiPOD" struct JMiPOD <: AbstractJMiPOD
    payload::Float64 = 0.4
    ratio::Float64 = 0.1
    at_train::Float64 = 0.45
    at_valid::Float64 = 0.05
end

@option "JMiPODSmall" struct JMiPODSmall <: AbstractJMiPOD
    payload::Float64 = 0.4
    ratio::Float64 = 0.1
    at_train::Float64 = 0.45
    at_valid::Float64 = 0.05
end

function Base.string(d::D) where {D<:AbstractJMiPOD}
    vals = dir_string.((d.payload, d.ratio, d.at_train, d.at_valid))
    return "$(D.name.name)($(join(vals, ", ")))"
end

jmipoddir(args...) = datadir("dataset", "jmipod", args...)
actordir(id::Int, args...) = jmipoddir("actor$(lpad(id, 5, "0"))", args...)

get_ids(::JMiPOD) = 1:47807
get_ids(::JMiPODSmall) = setdiff(1:111, 6:10:116)

list_jpgs(dir) = filter(file -> endswith(file, ".jpg"), readdir(dir; join = true))

function list_files(d::AbstractJMiPOD)
    ids = get_ids(d)

    files_cover = String[]
    files_stego = String[]
    for id in ids
        cover = list_jpgs(actordir(id))
        append!(files_cover, cover)

        type = lpad(round(Int, 100*d.payload), 3, "0")
        stego = list_jpgs(actordir(id, "stego-$(type)"))
        append!(files_stego, stego)
    end
    return files_cover, files_stego
end

function load_image(path)
    img = Float32.(channelview(jpeg_decode(path)))
    return permutedims(img, (3, 2, 1))
end

function load(d::AbstractJMiPOD)
    x_cover, x_stego = list_files(d)
    n = length(x_cover) + length(x_stego)

    x = vcat(x_cover, x_stego)
    y = reshape((1:n) .> length(x_cover), 1, :)

    # split
    at = (d.at_train, d.at_valid)
    i1_train, i1_valid, i1_test = split_data(length(x_cover), at)
    i2_train, i2_valid, i2_test = split_data(length(x_stego), at, d.ratio)

    i_train = vcat(i1_train, i2_train .+ length(x_cover))
    i_valid = vcat(i1_valid, i2_valid .+ length(x_cover))
    i_test = vcat(i1_test, i2_test .+ length(x_cover))

    train = (FileDataset(load_image, x[i_train]), y[:, i_train])
    valid = (FileDataset(load_image, x[i_valid]), y[:, i_valid])
    test = (FileDataset(load_image, x[i_test]), y[:, i_test])

    @info """
    Dataset: all (cover/stego)
    ⋅ Train: $(length(train[2])) ($(sum(.~train[2]))/$(sum(train[2])))
    ⋅ Valid: $(length(valid[2])) ($(sum(.~valid[2]))/$(sum(valid[2])))
    ⋅ Test: $(length(test[2])) ($(sum(.~test[2]))/$(sum(test[2])))
    """
    return train, valid, test
end
