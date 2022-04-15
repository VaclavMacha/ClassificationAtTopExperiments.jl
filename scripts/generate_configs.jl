using DrWatson
quickactivate("SteganographyExperiments.jl")

using Experiments

configsdir(args...) = projectdir("configs", args...)

full_dataset = true

Dconfigs = [
    DataConfig(; type="Nsf5_02", full_dataset, stego_ratio=1),
    DataConfig(; type="Nsf5_02", full_dataset, stego_ratio=0.5),
    DataConfig(; type="Nsf5_02", full_dataset, stego_ratio=0.1),
]

Lconfigs = LossConfig.([
    CrossEntropy(ϵ=0.5),
    CrossEntropy(ϵ=0.9),
    CrossEntropy(ϵ=0.99),
    CrossEntropy(ϵ=0.999),
    AATP(threshold=PatMatType(1e-3), surrogate=Hinge()),
    AATP(threshold=PatMatType(1e-4), surrogate=Hinge()),
    AATP(threshold=PatMatType(1e-5), surrogate=Hinge()),
])
Oconfigs = [
    OptConfig(type="ADAM", eta=1e-2),
]
Mconfig = ModelConfig(Linear())
Tconfig = TrainConfig(iters=1000, checkpoint_every=100)

i = 0
for Dconfig in Dconfigs, Lconfig in Lconfigs, Oconfig in Oconfigs
    i += 1
    d = make_dict(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig)
    path = configsdir(ifelse(full_dataset, "full", "partial"), "config_$(lpad(i, 4, "0")).yaml")
    mkpath(dirname(path))
    save_config(path, d)
end