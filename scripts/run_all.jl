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
    configs_dir();
    logdir=log_dir(),
    partition="amd",
    mem="100G"
) |> run
