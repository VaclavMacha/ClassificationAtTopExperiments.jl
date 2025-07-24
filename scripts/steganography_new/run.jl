using DrWatson
@quickactivate("ClassificationAtTopExperiments.jl")

using Pkg
Pkg.instantiate()
Pkg.precompile()

include(srcdir("slurm.jl"))
using .SLURM

configs_dir(args...) = projectdir("configs", "new", args...)
log_dir(args...) = projectdir("logs", "new", args...)

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

for folder in joinpath.("Nsf5SmallSGD", readdir(configs_dir("Nsf5SmallSGD")))
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(folder);
        logdir=log_dir(folder),
        partition="amd",
        cpus_per_task=2,
        mem="100G"
    ) |> run
end
