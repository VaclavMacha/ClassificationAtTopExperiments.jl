using DrWatson
quickactivate("ClassificationAtTopExperiments.jl")

using Experiments

# utilities
configsdir(args...) = projectdir("configs", "new", args...)

function generate_configs(
    dir_name,
    datasets,
    models,
    objectives,
    optimisers,
    trains;
    limit::Int=500
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

function generate_configs(dir_name, datasets, models, objectives, trains; limit=500)
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

function select_not_solved(maindir, dirname="to_solve"; limit=500)
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
SAVE_DIR = "steganography_new"

objectives = (
    # hinge
    DeepTopPush(λ=0,surrogate="Hinge"),
    PatMatNP(λ=0, τ=1e-3, surrogate="Hinge"),
    PatMatNP(λ=0, τ=1e-4, surrogate="Hinge"),
    PatMatNP(λ=0, τ=1e-5, surrogate="Hinge"),
    TauFPL(λ=0, τ=1e-3, surrogate="Hinge"),
    TauFPL(λ=0, τ=1e-4, surrogate="Hinge"),
    TauFPL(λ=0, τ=1e-5, surrogate="Hinge"),
    # softplus
    DeepTopPush(λ=0,surrogate="Softplus"),
    PatMatNP(λ=0, τ=1e-3, surrogate="Softplus"),
    PatMatNP(λ=0, τ=1e-4, surrogate="Softplus"),
    PatMatNP(λ=0, τ=1e-5, surrogate="Softplus"),
    TauFPL(λ=0, τ=1e-3, surrogate="Softplus"),
    TauFPL(λ=0, τ=1e-4, surrogate="Softplus"),
    TauFPL(λ=0, τ=1e-5, surrogate="Softplus"),
    # mode
    MODE(fpr=1e-3),
    MODE(fpr=1e-4),
    MODE(fpr=1e-5),
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
    # Nsf5Small(payload=0.2, ratio=0.5),
    # Nsf5Small(payload=0.2, ratio=0.1),
)

trains = (
    TrainConfig(;
        seed=seed,
        epoch_max=epoch_max,
        checkpoint_every=50,
        device="CPU",
        save_dir=SAVE_DIR,
        force=false,
    )
    for seed in 1:10
    for epoch_max in [3000]
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
    # Nsf5Small(payload=0.2, ratio=0.5),
    # Nsf5Small(payload=0.2, ratio=0.1),
)

trains = (
    TrainConfig(;
        seed=seed,
        epoch_max=epoch_max,
        checkpoint_every=50,
        batch_neg=batch_size,
        batch_pos=batch_size,
        device="CPU",
        save_dir=SAVE_DIR,
        force=false,
    )
    for seed in 1:10
    for epoch_max in [3000]
    for batch_size in [128]
)

for dataset in datasets
    dir = string("Nsf5SmallSGD/", string(typeof(dataset).name.name, "-", dataset.ratio))
    generate_configs(dir, (dataset,), (Linear(),), objectives, optimisers, trains)
end