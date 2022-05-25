#!/usr/bin/env sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#=

export JULIA_NUM_THREADS=32
export DATADEPS_ALWAYS_ACCEPT= "true"

srun --unbuffer julia --color=no --startup-file=no "${BASH_SOURCE[0]}" "$@"
exit
=#

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
    "/home/machava2/projects/SteganographyExperiments.jl",
    "SteganographyExperiments.jl",
)

using Experiments

load_or_run(path)
