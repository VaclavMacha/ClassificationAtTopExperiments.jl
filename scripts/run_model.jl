#!/usr/bin/env sh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#=
export JULIA_NUM_THREADS=20

srun --unbuffer julia --color=no --startup-file=no "${BASH_SOURCE[0]}" "$@"
exit
=#

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# parse args
id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
path = readdir(ARGS[1]; join=true, sort=true)[id]

# precompilation
s = rand(1:120)
@info """
Configuration:
⋅ File: $(basename(path))
⋅ Dir: $(dirname(path))
⋅ Sleep: $(s)s
"""
sleep(s)

using DrWatson
quickactivate(
    "/home/machava2/projects/SteganographyExperiments.jl/scripts/run_model.jl", "SteganographyExperiments.jl",
)

using Experiments

run_experiments(path)