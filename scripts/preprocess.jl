using HDF5
using ProgressMeter

"""
    load_features(path)

Reads features from .bin file.
"""
function load_features(path)
    l = div(stat(path).size, (22510 * 8))
    x = zeros(22510, l)
    open(io -> read!(io, x), path, "r")
    return x
end

nsamples(path) = div(stat(path).size, (22510 * 8))
filedir(dir, id::Int) = joinpath(dir, "actor$(lpad(id, 6, "0"))")

function save_hdf5(
    dir::AbstractString,
    ids,
    type::AbstractString;
    savedir=pwd()
)

    # summary
    files = String[]
    for fdir in filedir.(dir, ids)
        fls = filter(file -> contains(file, type), readdir(fdir; join=true))
        if isempty(fls)
            @warn "File not found in $(fdir)"
        elseif length(fls) != 1
            @warn "Multiple files in $(fdir)"
        else
            append!(files, fls)
        end
    end

    n = mapreduce(nsamples, +, files)
    @info "# files: $(length(files))"
    @info "# samples: $(n)"

    # save as h5
    mkpath(savedir)
    h5open(joinpath(savedir, "$(type).h5"), "w") do fid
        data = create_dataset(fid, "data", datatype(Float32), dataspace(22510, n))

        col = 0
        @showprogress for file in files
            x = load_features(file)
            cols = (col+1):(col+size(x, 2))
            col = cols[end]
            data[:, cols] = x
        end
    end
    return
end

dir = "/mnt/stego/flcc80-all-fea/"
savedir = "/mnt/stego/alaska/"

# partial dataset
ids = setdiff(1:111, 6:10:116)
save_hdf5(dir, ids, "nsf5_0.1_jrm"; savedir=joinpath(savedir, "partial"))
save_hdf5(dir, ids, "nsf5_0.2_jrm"; savedir=joinpath(savedir, "partial"))
save_hdf5(dir, ids, "nsf5_0.5_jrm"; savedir=joinpath(savedir, "partial"))
save_hdf5(dir, ids, "cover_jrm"; savedir=joinpath(savedir, "partial"))

# full dataset
ids = 1:47807
save_hdf5(dir, ids, "nsf5_0.1_jrm"; savedir=joinpath(savedir, "full"))
save_hdf5(dir, ids, "nsf5_0.2_jrm"; savedir=joinpath(savedir, "full"))
save_hdf5(dir, ids, "nsf5_0.5_jrm"; savedir=joinpath(savedir, "full"))
save_hdf5(dir, ids, "cover_jrm"; savedir=joinpath(savedir, "full"))

@info "Finished"