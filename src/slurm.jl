module SLURM

export sbatch, sbatch_array

slurm_cmd(key::Symbol) = Symbol(replace(string(key), "_" => "-"))

parse_arg(arg::Pair) = "--$(slurm_cmd(arg[1]))=$(arg[2])"
parse_arg(arg) = "$(arg)"
parse_args(args) = [parse_arg(arg) for arg in args]

function sbatch(
    file::AbstractString,
    args...;
    logdir::AbstractString = dirname(file),
    output::AbstractString = joinpath(logdir, "%A.log"),
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
    zero_indexing::Bool = false,
    logdir::AbstractString = dirname(file),
    output::AbstractString = joinpath(logdir, "%A-%a.log"),
    kwargs...
)
    sbatch_args = parse_args(kwargs)
    k = length(readdir(dir))
    array = zero_indexing ? "--array=0-$(k-1)" : "--array=1-$(k)"

    mkpath(dirname(output))

    return `sbatch $(array) --output=$(output) $(sbatch_args) $(file) $(dir)`
end

end # module