using DrWatson
@quickactivate("ClassificationAtTopExperiments.jl")

using Pkg
Pkg.instantiate()
Pkg.precompile()

include(srcdir("slurm.jl"))
using .SLURM

configs_dir(args...) = projectdir("configs", args...)
log_dir(args...) = projectdir("logs", args...)

for folder in joinpath.("Nsf5Small", readdir(configs_dir("Nsf5Small")))
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(folder);
        logdir=log_dir(folder),
        partition="amd",
        cpus_per_task=2,
        mem="100G"
    ) |> run
end

for folder in joinpath.("Nsf5", readdir(configs_dir("Nsf5")))
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(folder);
        logdir=log_dir(folder),
        partition="amd",
        cpus_per_task=2,
        mem="1000G"
    ) |> run
end

for folder in joinpath.("JMiPODSmall", readdir(configs_dir("JMiPODSmall")))
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(folder);
        logdir=log_dir(folder),
        partition="amdgpu",
        gres="gpu:1",
        cpus_per_task=16,
        mem="50G"
    ) |> run
end

for folder in joinpath.("JMiPOD", readdir(configs_dir("JMiPOD")))
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(folder);
        logdir=log_dir(folder),
        partition="amdgpuextralong",
        gres="gpu:1",
        cpus_per_task=16,
        mem="100G"
    ) |> run
end
