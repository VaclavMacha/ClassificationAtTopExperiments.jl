#!/usr/bin/env sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#=

module load --ignore-cache Julia/1.7.2-linux-x86_64

export DATADEPS_ALWAYS_ACCEPT= "true"

srun --unbuffer julia --color=no --startup-file=no --threads=auto "${BASH_SOURCE[0]}" "$@"
exit
=#

# parse args
id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
path = readdir(ARGS[1]; join=true, sort=true)[id]

# precompilation
s = rand(1:60)
@info """
Configuration:
⋅ File: $(basename(path))
⋅ Dir: $(dirname(path))
⋅ Sleep: $(s)s
"""
sleep(s)

using DrWatson
quickactivate(
    "/home/machava2/projects/ClassificationAtTopExperiments.jl",
    "ClassificationAtTopExperiments.jl",
)

using Experiments

load_or_run(path)
