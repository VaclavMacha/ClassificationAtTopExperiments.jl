using DrWatson
quickactivate("SteganographyExperiments.jl")

using Experiments

configsdir(args...) = projectdir("configs", args...)

Lconfigs = LossConfig.([
    AATP(threshold=PatMatType(1e-3), surrogate=Hinge()),
    AATP(threshold=PatMatType(1e-4), surrogate=Hinge()),
    AATP(threshold=PatMatType(1e-5), surrogate=Hinge()),
    CrossEntropy(ϵ=0.5),
    CrossEntropy(ϵ=0.9),
    CrossEntropy(ϵ=0.99),
    CrossEntropy(ϵ=0.999),
])
Oconfigs = [
    OptConfig(type="ADAM", eta=1e-2),
]
Mconfig = ModelConfig(Linear())

# Partial data
Dconfigs = [
    DataConfig(; type="Nsf5_02", full_dataset = false, stego_ratio=1),
    DataConfig(; type="Nsf5_02", full_dataset = false, stego_ratio=0.5),
    DataConfig(; type="Nsf5_02", full_dataset = false, stego_ratio=0.1),
]
Tconfig_partial = TrainConfig(iters=1000, checkpoint_every=100)

i = 0
for Dconfig in Dconfigs, Lconfig in Lconfigs, Oconfig in Oconfigs
    i += 1
    d = make_dict(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig_partial)
    path = configsdir("partial", "config_$(lpad(i, 4, "0")).yaml")
    mkpath(dirname(path))
    save_config(path, d)
end

# Full data
Dconfigs = [
    DataConfig(; type="Nsf5_02", full_dataset = true, stego_ratio=1),
    DataConfig(; type="Nsf5_02", full_dataset = true, stego_ratio=0.5),
    DataConfig(; type="Nsf5_02", full_dataset = true, stego_ratio=0.1),
]
Tconfig_full = TrainConfig(iters=500, checkpoint_every=10)

i = 0
for Dconfig in Dconfigs, Lconfig in Lconfigs, Oconfig in Oconfigs
    i += 1
    d = make_dict(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig_full)
    path = configsdir("full", "config_$(lpad(i, 4, "0")).yaml")
    mkpath(dirname(path))
    save_config(path, d)
end