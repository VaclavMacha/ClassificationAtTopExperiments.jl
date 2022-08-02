using DrWatson
@quickactivate("ClassificationAtTopExperiments.jl")

using Pkg
Pkg.instantiate()
Pkg.precompile()

include(srcdir("slurm.jl"))
using .SLURM

configs_dir(args...) = projectdir("configs", args...)
log_dir(args...) = projectdir("logs", args...)

for folder in joinpath.("Primal", readdir(configs_dir("Primal")))
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(folder);
        logdir=log_dir(folder),
        partition="amd",
        cpus_per_task=2,
        mem="100G"
    ) |> run
end

for folder in joinpath.("PrimalNN", readdir(configs_dir("PrimalNN")))
    sbatch_array(
        scriptsdir("run_model_MPI.jl"),
        configs_dir(folder);
        mpi_jobs=4,
        logdir=log_dir(folder),
        partition="amdgpufast",
        gres="gpu:1",
        cpus_per_task=4,
        mem="200G"
    ) |> run
end

for folder in joinpath.("Dual", readdir(configs_dir("Dual")))
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(folder);
        logdir=log_dir(folder),
        partition="amdlong",
        cpus_per_task=2,
        mem="100G"
    ) |> run
end
