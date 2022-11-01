using DrWatson
quickactivate("ClassificationAtTopExperiments.jl")

using Experiments

# utilities
configsdir(args...) = projectdir("configs", args...)

function generate_configs(
    dir_name,
    datasets,
    models,
    objectives,
    optimisers,
    trains;
    limit::Int=1000
)

    i = 0
    j = 1
    for train in trains, optimiser in optimisers
        for dataset in datasets, model in models, objective in objectives
            i += 1
            path = configsdir(string(dir_name, "_$(j)"), "config_$(lpad(i, 4, "0")).toml")
            write_config(path, dataset, model, objective, optimiser, train)

            if i == limit
                i = 0
                j += 1
            end
        end
    end
end

function generate_configs(dir_name, datasets, models, objectives, trains; limit=1000)
    i = 0
    j = 1
    for train in trains
        for dataset in datasets, model in models, objective in objectives
            i += 1
            path = configsdir(string(dir_name, "_$(j)"), "config_$(lpad(i, 4, "0")).toml")
            write_config(path, dataset, model, objective, train)

            if i == limit
                i = 0
                j += 1
            end
        end
    end
end

function select_not_solved(maindir, dirname="to_solve"; limit=1000)
    i = 0
    j = 1
    for dir in readdir(maindir; join=true)
        for file in readdir(dir; join=true)
            if is_solved(file)
                rm(file)
            else
                i += 1
                savedir = joinpath(maindir, string(dirname, "_$(j)"))
                mkpath(savedir)
                mv(file, joinpath(savedir, "config_$(lpad(i, 4, "0")).toml"))
            end
            if i == limit
                i = 0
                j += 1
            end
        end
        rm(dir)
    end
end

#-------------------------------------------------------------------------------------------
# Primal formulation
#-------------------------------------------------------------------------------------------
λs = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
ϑs = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

optimisers = (
    OptADAM(eta=1e-2),
)

datasets = (
    MNIST(),
    FashionMNIST(),
    CIFAR10(),
    CIFAR20(),
    CIFAR100(),
    SVHN2(),
    SVHN2Extra(),
    Nsf5Small(),
)

objectives = (
    (CrossEntropy(; λ) for λ in λs)...,
    (TopPush(; λ, surrogate="Hinge") for λ in λs)...,
    (DeepTopPush(; λ, surrogate="Hinge") for λ in λs)...,
    (TopPushK(; λ, K=5, surrogate="Hinge") for λ in λs)...,
    (TopPushK(; λ, K=10, surrogate="Hinge") for λ in λs)...,
    (PatMatNP(; λ=0.001, τ=0.01, ϑ, surrogate="Hinge") for ϑ in ϑs)...,
    (PatMatNP(; λ=0.001, τ=0.05, ϑ, surrogate="Hinge") for ϑ in ϑs)...,
    (TauFPL(; λ, τ=0.01, surrogate="Hinge") for λ in λs)...,
    (TauFPL(; λ, τ=0.05, surrogate="Hinge") for λ in λs)...,
    (GrillNP(; λ, τ=0.01, surrogate="Hinge") for λ in λs)...,
    (GrillNP(; λ, τ=0.05, surrogate="Hinge") for λ in λs)...,
)

# Linear model
trains = (
    TrainConfig(;
        seed=seed,
        epoch_max=100,
        checkpoint_every=5,
        batch_neg=256,
        batch_pos=256,
        device="CPU",
        save_dir="dissertation/primal"
    )
    for seed in 1:10
)

for dataset in datasets
    dir = string("dissertation/primal/", typeof(dataset).name.name)
    generate_configs(dir, (dataset,), (Linear(),), objectives, optimisers, trains)
end

# Neural Networks
trains = (
    TrainConfig(;
        seed=seed,
        epoch_max=100,
        checkpoint_every=5,
        batch_neg=256,
        batch_pos=256,
        device="GPU",
        save_dir="dissertation/primalNN"
    )
    for seed in 1:10
)

for dataset in datasets
    dir = string("dissertation/primalNN/", typeof(dataset).name.name)
    generate_configs(dir, (dataset,), (SimpleConv(),), objectives, optimisers, trains)
end

#-------------------------------------------------------------------------------------------
# Dual formulation
#-------------------------------------------------------------------------------------------
objectives = (
    (SVM(; λ) for λ in λs)...,
    (TopPush(; λ, surrogate="Hinge") for λ in λs)...,
    (TopPushK(; λ, K=5, surrogate="Hinge") for λ in λs)...,
    (TopPushK(; λ, K=10, surrogate="Hinge") for λ in λs)...,
    (PatMatNP(; λ=0.001, τ=0.01, ϑ, surrogate="Hinge") for ϑ in ϑs)...,
    (PatMatNP(; λ=0.001, τ=0.05, ϑ, surrogate="Hinge") for ϑ in ϑs)...,
    (TauFPL(; λ, τ=0.01, surrogate="Hinge") for λ in λs)...,
    (TauFPL(; λ, τ=0.05, surrogate="Hinge") for λ in λs)...,
)

datasets = (
    MNIST(),
    FashionMNIST(),
    CIFAR10(),
    CIFAR20(),
    CIFAR100(),
    SVHN2(),
)

models = (
    Linear(),
    Gaussian(),
)

trains = (
    TrainConfigDual(;
        seed=seed,
        epoch_max=20,
        checkpoint_every=5,
        loss_every=100,
        p_update=0.9,
        ε=1e-4,
        save_dir="dissertation/dual"
    )
    for seed in 1:10
)

for dataset in datasets
    dir = string("dissertation/dual/", typeof(dataset).name.name)
    generate_configs(dir, (dataset,), models, objectives, trains)
end
