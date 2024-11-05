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
    limit::Int=1500
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

function generate_configs(dir_name, datasets, models, objectives, trains; limit=1500)
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

function select_not_solved(maindir, dirname="to_solve"; limit=1500)
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
# Formulations
#-------------------------------------------------------------------------------------------
objectives = (
    DeepTopPush(surrogate="Hinge"),
    PatMatNP(τ=1e-3, surrogate="Hinge"),
    PatMatNP(τ=1e-4, surrogate="Hinge"),
    PatMatNP(τ=1e-5, surrogate="Hinge"),
    CrossEntropy(ϵ=0.5),
    CrossEntropy(ϵ=0.9),
    CrossEntropy(ϵ=0.99),
    CrossEntropy(ϵ=0.999),
)

#-------------------------------------------------------------------------------------------
# Nsf5Small
#-------------------------------------------------------------------------------------------
# optimisers
optimisers = (
    OptADAM(eta=1e-2, decay_every=50),
)

# SGD
datasets = (
    Nsf5Small(payload=0.2, ratio=1),
    Nsf5Small(payload=0.2, ratio=0.5),
    Nsf5Small(payload=0.2, ratio=0.1),
)

trains = (
    TrainConfig(;
        seed=seed,
        epoch_max=1000,
        checkpoint_every=50,
        device="CPU",
        save_dir="steganography"
    )
    for seed in 1:10
)

for dataset in datasets
    dir = string("Nsf5Small/", string(typeof(dataset).name.name, "-", dataset.ratio))
    generate_configs(dir, (dataset,), (Linear(),), objectives, optimisers, trains)
end

#-------------------------------------------------------------------------------------------
# Nsf5Small SGD
#-------------------------------------------------------------------------------------------
# optimisers
optimisers = (
    OptADAM(eta=1e-2, decay_every=50),
)

# SGD
datasets = (
    Nsf5Small(payload=0.2, ratio=1),
    Nsf5Small(payload=0.2, ratio=0.5),
    Nsf5Small(payload=0.2, ratio=0.1),
)

trains = (
    TrainConfig(;
        seed=seed,
        epoch_max=1000,
        checkpoint_every=50,
        batch_neg=batch_size,
        batch_pos=batch_size,
        device="CPU",
        save_dir="steganography"
    )
    for seed in 1:10
    for batch_size in [16, 32, 128]
)

for dataset in datasets
    dir = string("Nsf5Small/", string(typeof(dataset).name.name, "-", dataset.ratio))
    generate_configs(dir, (dataset,), (Linear(),), objectives, optimisers, trains)
end

#-------------------------------------------------------------------------------------------
# Nsf5
#-------------------------------------------------------------------------------------------
# optimisers
optimisers = (
    OptADAM(eta=1e-2, decay_every=50),
)

# SGD
datasets = (
    Nsf5(payload=0.2, ratio=1),
    Nsf5(payload=0.2, ratio=0.5),
    Nsf5(payload=0.2, ratio=0.1),
)

trains = (
    TrainConfig(;
        seed=seed,
        epoch_max=500,
        checkpoint_every=10,
        device="CPU",
        save_dir="steganography",
    )
    for seed in 1:10
)

for dataset in datasets
    dir = string("Nsf5/", string(typeof(dataset).name.name, "-", dataset.ratio))
    generate_configs(dir, (dataset,), (Linear(),), objectives, optimisers, trains)
end

#-------------------------------------------------------------------------------------------
# JMiPODSmall
#-------------------------------------------------------------------------------------------
# optimisers
optimisers = (
    OptDescent(eta=1e-2, decay_every=5),
)

# SGD
datasets = (
    JMiPODSmall(payload=0.1, ratio=1),
    JMiPODSmall(payload=0.1, ratio=0.5),
    JMiPODSmall(payload=0.1, ratio=0.1),
)

trains = (
    TrainConfig(;
        seed=seed,
        epoch_max=30,
        checkpoint_every=10,
        eval_all=true,
        batch_neg=batch_size,
        batch_pos=batch_size,
        device="GPU",
        save_dir="steganography",
    )
    for seed in 1:10
    for batch_size in [16, 32, 128]
)

for dataset in datasets
    dir = string("JMiPODSmall/", string(typeof(dataset).name.name, "-", dataset.ratio))
    generate_configs(dir, (dataset,), (EfficientNetB0(true),), objectives, optimisers, trains)
end

#-------------------------------------------------------------------------------------------
# JMiPOD
#-------------------------------------------------------------------------------------------
# optimisers
optimisers = (
    OptDescent(eta=1e-2, decay_every=5),
)

# SGD
datasets = (
    JMiPOD(payload=0.1, ratio=1),
    JMiPOD(payload=0.1, ratio=0.5),
    JMiPOD(payload=0.1, ratio=0.1),
)

trains = (
    TrainConfig(;
        seed=seed,
        epoch_max=30,
        checkpoint_every=10,
        eval_all=false,
        batch_neg=128,
        batch_pos=128,
        device="GPU",
        save_dir="steganography",
    )
    for seed in 1:10
)

for dataset in datasets
    dir = string("JMiPOD/", typeof(dataset).name.name)
    generate_configs(dir, (dataset,), (EfficientNetB0(true),), objectives, optimisers, trains)
end
