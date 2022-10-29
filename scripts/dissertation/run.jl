using DrWatson
@quickactivate("ClassificationAtTopExperiments.jl")

using Pkg
Pkg.instantiate()
Pkg.precompile()

include(srcdir("slurm.jl"))
using .SLURM

configs_dir(args...) = projectdir("configs", args...)
log_dir(args...) = projectdir("logs", args...)

primal = "dissertation/primal"
for folder in joinpath.(primal, readdir(configs_dir(primal)))
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(folder);
        logdir=log_dir(folder),
        partition="amd",
        cpus_per_task=2,
        mem="50G"
    ) |> run
end

primal = "dissertation/primalFull"
for folder in joinpath.(primal, readdir(configs_dir(primal)))
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(folder);
        logdir=log_dir(folder),
        partition="amdfast",
        cpus_per_task=2,
        mem="50G"
    ) |> run
end

primalnn = "dissertation/primalNN"
for folder in joinpath.(primalnn, readdir(configs_dir(primalnn)))
    sbatch_array(
        scriptsdir("run_model_MPI.jl"),
        configs_dir(folder);
        mpi_jobs=4,
        logdir=log_dir(folder),
        partition="amdgpu",
        gres="gpu:1",
        cpus_per_task=4,
        mem="200G"
    ) |> run
end

dual = "dissertation/dual"
for folder in joinpath.(dual, readdir(configs_dir(dual)))
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(folder);
        logdir=log_dir(folder),
        partition="amdlong",
        cpus_per_task=2,
        mem="100G"
    ) |> run
end
