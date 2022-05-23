using DrWatson
quickactivate("SteganographyExperiments.jl")

using Experiments

configsdir(args...) = projectdir("configs", args...)

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
optimiser = OptADAM(eta=1e-2)

#-------------------------------------------------------------------------------------------
# Nsf5 datasets
#-------------------------------------------------------------------------------------------
model = Linear()
train_config = TrainConfig(epoch_max=500, checkpoint_every=50)

# Small data
datasets = (
    Nsf5Small(payload=0.2, ratio=1),
    Nsf5Small(payload=0.2, ratio=0.5),
    Nsf5Small(payload=0.2, ratio=0.1),
)

i = 0
for dataset in datasets, loss_type in objectives
    i += 1
    path = configsdir("Nsf5Small", "config_$(lpad(i, 4, "0")).toml")
    write_config(path, dataset, model, loss_type, optimiser, train_config)
end

# Full data
datasets = (
    Nsf5(payload=0.2, ratio=1),
    Nsf5(payload=0.2, ratio=0.5),
    Nsf5(payload=0.2, ratio=0.1),
)

i = 0
for dataset in datasets, loss_type in objectives
    i += 1
    path = configsdir("Nsf5", "config_$(lpad(i, 4, "0")).toml")
    write_config(path, dataset, model, loss_type, optimiser, train_config)
end

#-------------------------------------------------------------------------------------------
# JMiPOD datasets
#-------------------------------------------------------------------------------------------
model = EfficientNetB0()
train_config = TrainConfig(
    epoch_max=100,
    checkpoint_every=10,
    batch_neg=32,
    batch_pos=32,
    device="GPU",
)

# Small data
datasets = (
    JMiPODSmall(payload=0.4, ratio=1),
    JMiPODSmall(payload=0.4, ratio=0.5),
    JMiPODSmall(payload=0.4, ratio=0.1),
)

i = 0
for dataset in datasets, loss_type in objectives
    i += 1
    path = configsdir("JMiPODSmall", "config_$(lpad(i, 4, "0")).toml")
    write_config(path, dataset, model, loss_type, optimiser, train_config)
end

# Full data
datasets = (
    JMiPOD(payload=0.4, ratio=1),
    JMiPOD(payload=0.4, ratio=0.5),
    JMiPOD(payload=0.4, ratio=0.1),
)

i = 0
for dataset in datasets, loss_type in objectives
    i += 1
    path = configsdir("JMiPOD", "config_$(lpad(i, 4, "0")).toml")
    write_config(path, dataset, model, loss_type, optimiser, train_config)
end
