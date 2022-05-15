@option struct DataConfig
    dataset::Union{Nsf5, Nsf5Small, JMiPOD, JMiPODSmall}
end

Base.string(d::DataConfig) = string(d.dataset)
load(d::DataConfig) = load(d.dataset)
