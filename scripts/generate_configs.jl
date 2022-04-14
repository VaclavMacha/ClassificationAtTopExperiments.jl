using DrWatson
quickactivate("SteganographyExperiments.jl")

using Experiments

configsdir(args...) = projectdir("configs", args...)

Dconfigs = DataConfig.([
    Nsf5_01(),
    Nsf5_02(),
    Nsf5_05(),
])

Lconfigs = LossConfig.([
    CrossEntropy(ϵ=0.5),
    CrossEntropy(ϵ=0.1),
    CrossEntropy(ϵ=0.01),
    CrossEntropy(ϵ=0.001),
    AATP(threshold=PatMatType(1e-3), surrogate=Hinge()),
    AATP(threshold=PatMatType(1e-4), surrogate=Hinge()),
    AATP(threshold=PatMatType(1e-5), surrogate=Hinge()),
    AATP(threshold=PatMatType(1e-3), surrogate=Quadratic()),
    AATP(threshold=PatMatType(1e-4), surrogate=Quadratic()),
    AATP(threshold=PatMatType(1e-5), surrogate=Quadratic()),
])
Oconfigs = [
    OptConfig(type="ADAM", eta=1e-2),
    OptConfig(type="ADAM", eta=1e-3),
    OptConfig(type="ADAM", eta=1e-4),
]
Mconfig = ModelConfig(Linear())
Tconfig = TrainConfig(iters = 1000, checkpoint_every = 100)

i = 0
for Dconfig in Dconfigs, Lconfig in Lconfigs, Oconfig in Oconfigs
    i += 1
    d = make_dict(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig)
    path = configsdir("config_$(lpad(i, 4, "0")).yaml")
    mkpath(dirname(path))
    save_config(path, d)
end