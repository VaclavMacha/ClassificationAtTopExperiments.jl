using DrWatson
quickactivate("SteganographyExperiments.jl")

using Experiments

configsdir(args...) = projectdir("configs", args...)

Lconfigs = LossConfig.([
    DeepTopPush(surrogate="Hinge"),
    PatMatNP(τ=1e-3, surrogate="Hinge"),
    PatMatNP(τ=1e-4, surrogate="Hinge"),
    PatMatNP(τ=1e-5, surrogate="Hinge"),
    CrossEntropy(ϵ=0.5),
    CrossEntropy(ϵ=0.9),
    CrossEntropy(ϵ=0.99),
    CrossEntropy(ϵ=0.999),
])
Oconfig = OptConfig(type="ADAM", eta=1e-2)

#-------------------------------------------------------------------------------------------
# Nsf5 datasets
#-------------------------------------------------------------------------------------------
Mconfig = ModelConfig(Linear())
Tconfig = TrainConfig(epochs=500, checkpoint_every=50, eval_every=50)

# Small data
Dconfigs = DataConfig.((
    Nsf5Small(payload=0.2, ratio=1),
    Nsf5Small(payload=0.2, ratio=0.5),
    Nsf5Small(payload=0.2, ratio=0.1),
))

i = 0
for Dconfig in Dconfigs, Lconfig in Lconfigs
    i += 1
    d = make_dict(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig)
    path = configsdir("Nsf5Small", "config_$(lpad(i, 4, "0")).yaml")
    mkpath(dirname(path))
    save_config(path, d)
end

# Full data
Dconfigs = DataConfig.((
    Nsf5(payload=0.2, ratio=1),
    Nsf5(payload=0.2, ratio=0.5),
    Nsf5(payload=0.2, ratio=0.1),
))

i = 0
for Dconfig in Dconfigs, Lconfig in Lconfigs
    i += 1
    d = make_dict(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig)
    path = configsdir("Nsf5", "config_$(lpad(i, 4, "0")).yaml")
    mkpath(dirname(path))
    save_config(path, d)
end

#-------------------------------------------------------------------------------------------
# JMiPOD datasets
#-------------------------------------------------------------------------------------------
Mconfig = ModelConfig(Efficientnet(pretrained = false))
Tconfig = TrainConfig(
    epochs=100,
    checkpoint_every=10,
    eval_every=10,
    buffer=true,
    batch_size=64,
    batch_neg=32,
    batch_pos=32,
    device="GPU",
)

# Small data
Dconfigs = DataConfig.((
    JMiPODSmall(payload=0.4, ratio=1),
    JMiPODSmall(payload=0.4, ratio=0.5),
    JMiPODSmall(payload=0.4, ratio=0.1),
))

i = 0
for Dconfig in Dconfigs, Lconfig in Lconfigs
    i += 1
    d = make_dict(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig)
    path = configsdir("JMiPODSmall", "config_$(lpad(i, 4, "0")).yaml")
    mkpath(dirname(path))
    save_config(path, d)
end

# Full data
Dconfigs = DataConfig.((
    JMiPOD(payload=0.4, ratio=1),
    JMiPOD(payload=0.4, ratio=0.5),
    JMiPOD(payload=0.4, ratio=0.1),
))

i = 0
for Dconfig in Dconfigs, Lconfig in Lconfigs
    i += 1
    d = make_dict(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig)
    path = configsdir("JMiPOD", "config_$(lpad(i, 4, "0")).yaml")
    mkpath(dirname(path))
    save_config(path, d)
end
