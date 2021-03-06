module SLURM

export sbatch, sbatch_array

slurm_cmd(key::Symbol) = Symbol(replace(string(key), "_" => "-"))

slurm_val(key) = string(key)
slurm_val(key::AbstractArray) = join(slurm_val.(key), ",")
slurm_val(key::Tuple) = join(slurm_val.(key), ",")

parse_arg(arg::Pair) = "--$(slurm_cmd(arg[1]))=$(slurm_val(arg[2]))"
parse_arg(arg) = slurm_val(arg)
parse_args(args) = [parse_arg(arg) for arg in args]

function sbatch(
    file::AbstractString,
    args...;
    logdir::AbstractString=dirname(file),
    output::AbstractString=joinpath(logdir, "%A.log"),
    kwargs...
)
    sbatch_args = parse_args(kwargs)
    file_args = parse_args(args)

    mkpath(dirname(output))

    return `sbatch --output=$(output) $(sbatch_args) $(file) $(file_args)`
end

function sbatch_array(
    file::AbstractString,
    dir::AbstractString;
    mpi_jobs::Int = 1,
    zero_indexing::Bool=false,
    logdir::AbstractString=dirname(file),
    output::AbstractString=joinpath(logdir, "%A-%a.log"),
    kwargs...
)
    sbatch_args = parse_args(kwargs)
    k = length(readdir(dir))
    if mpi_jobs > 1
        k = ceil(Int, k/mpi_jobs)
    end
    array = zero_indexing ? "--array=0-$(k-1)" : "--array=1-$(k)"

    mkpath(dirname(output))

    return `sbatch $(array) --output=$(output) $(sbatch_args) $(file) $(dir)`
end

end # module
