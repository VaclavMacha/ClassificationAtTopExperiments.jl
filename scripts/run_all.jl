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
    configs_dir("Nsf5Small");
    logdir=log_dir("Nsf5Small"),
    partition=("cpu", "amd"),
    mem="100G"
) |> run

sbatch_array(
    scriptsdir("run_model.jl"),
    configs_dir("Nsf5");
    logdir=log_dir("Nsf5"),
    partition="amdextralong",
    mem="1000G"
) |> run

sbatch_array(
    scriptsdir("run_model.jl"),
    configs_dir("JMiPODSmall");
    logdir=log_dir("JMiPODSmall"),
    partition="amdgpu"
    mem="100G"
) |> run

sbatch_array(
    scriptsdir("run_model.jl"),
    configs_dir("JMiPOD");
    logdir=log_dir("JMiPOD"),
    partition="amdgpuextralong"
    mem="100G"
) |> run
