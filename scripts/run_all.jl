using DrWatson
@quickactivate("SteganographyExperiments.jl")

using Pkg
Pkg.instantiate()
Pkg.precompile()

include(srcdir("slurm.jl"))
using .SLURM

configs_dir(args...) = projectdir("configs", args...)
log_dir(args...) = projectdir("logs", args...)

sbatch_array(
    scriptsdir("run_model.jl"),
    configs_dir("partial");
    logdir=log_dir(),
    partition="cpu",
    mem="100G"
) |> run

sbatch_array(
    scriptsdir("run_model.jl"),
    configs_dir("full");
    logdir=log_dir(),
    partition="amd",
    mem="1000G"
) |> run
