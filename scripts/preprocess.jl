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
filename(dir, id::Int, type) = joinpath(dir, "actor$(lpad(id, 6, "0"))", "$(type).bin")

function save_hdf5(
    dir::AbstractString,
    ids,
    type::AbstractString;
    savedir=pwd()
)

    # summary
    files = filename.(dir, ids, type)
    for file in files
        isfile(file) && continue
        @warn "file does not exist: $(file)"
    end
    filter!(isfile, files)

    n = mapreduce(nsamples, +, files)
    @info "# files: $(length(files))"
    @info "# samples: $(n)"

    # save as h5
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
ids = 1:100

save_hdf5(dir, ids, "cover_jrm"; savedir)
save_hdf5(dir, ids, "nsf5_0.1_jrm_partial"; savedir)
save_hdf5(dir, ids, "nsf5_0.2_jrm_partial"; savedir)
save_hdf5(dir, ids, "nsf5_0.5_jrm_partial"; savedir)

@info "Finished"