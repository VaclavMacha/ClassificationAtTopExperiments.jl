#!/usr/bin/env sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#=

module load fosscuda
module load cuDNN/8.0.5.39-CUDA-11.1.1
module load --ignore-cache Julia/1.7.3-linux-x86_64

export OMPI_MCA_mpi_warn_on_fork=0
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_CUDA_USE_BINARYBUILDER=false
export DATADEPS_ALWAYS_ACCEPT=true

mpirun -np 4 julia --color=no --startup-file=no --threads=auto "${BASH_SOURCE[0]}" "$@"
exit
=#

# parse args
id_slurm = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
id_mpi = parse(Int, ENV["OMPI_COMM_WORLD_RANK"])
id = 4 * (id_slurm - 1) + (id_mpi + 1)
path = readdir(ARGS[1]; join=true, sort=true)[id]

if !isfile(path)
    @warn "config file does not exists: $path"
else
    # precompilation
    s = rand(1:60)
    @info """
    Configuration:
    ⋅ File: $(basename(path))
    ⋅ Dir: $(dirname(path))
    ⋅ Sleep: $(s)s
    """
    sleep(s)

    using Pkg
    Pkg.add("DrWatson")

    using DrWatson
    quickactivate(
        "/home/machava2/projects/ClassificationAtTopExperiments.jl",
        "ClassificationAtTopExperiments.jl",
    )

    Pkg.instantiate()
    using Experiments

    function Experiments.datasetsdir(args...)
        return joinpath("/mnt/personal/machava2/datasets", args...)
    end

    try
        load_or_run(path)
    catch e
        @warn string(e)
        @warn "training failed: $path"
    end
end
