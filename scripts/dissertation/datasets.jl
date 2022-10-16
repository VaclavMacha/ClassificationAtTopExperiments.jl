using DrWatson
quickactivate("ClassificationAtTopExperiments.jl")

using Experiments
using Evaluation
using Evaluation.CSV

function Experiments.datasetsdir(args...)
    return joinpath("/mnt/personal/machava2/datasets", args...)
end

datasets = (
    MNIST(),
    FashionMNIST(),
    CIFAR10(),
    CIFAR20(),
    CIFAR100(),
    SVHN2(),
    SVHN2Extra(),
    Nsf5Small(),
    JMiPODSmall(),
)

df = Experiments.summary_dataset(datasets...)
CSV.write(datadir("dissertation", "datasets_summary.csv"), df)
