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
    partition="amd",
    cpus_per_task=2,
    mem="100G"
) |> run

datasets = (
    "DeepTopPush_MNIST",
    "DeepTopPush_FashionMNIST",
    "DeepTopPush_CIFAR10",
    "DeepTopPush_CIFAR20",
    "DeepTopPush_CIFAR100",
    "DeepTopPush_SVHN2",
    "DeepTopPush_SVHN2Extra",
)

for dataset in datasets
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(dataset);
        logdir=log_dir(dataset),
        partition="amdgpu",
        gres="gpu:1",
        cpus_per_task=2,
        mem="100G"
    ) |> run
end

# duals
datasets = (
    "Dual_MNIST",
    "Dual_FashionMNIST",
    "Dual_CIFAR10",
    "Dual_CIFAR20",
    "Dual_CIFAR100",
    "Dual_SVHN2",
)

for dataset in datasets
    sbatch_array(
        scriptsdir("run_model.jl"),
        configs_dir(dataset);
        logdir=log_dir(dataset),
        partition="amd",
        cpus_per_task=2,
        mem="100G"
    ) |> run
end
