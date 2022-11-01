using DrWatson
@quickactivate("ClassificationAtTopExperiments.jl")

using Evaluation

#-------------------------------------------------------------------------------------------
# Utilities
#-------------------------------------------------------------------------------------------
const LOSS = Dict(
    "CrossEntropy" => ("\\BaseLine", 1, :auc),
    "SVM" => ("\\SVM", 2, :auc),
    "TopPush" => ("\\TopPush", 3, :tpr_at_k1),
    "DeepTopPush" => ("\\DeepTopPush", 4, :tpr_at_k1),
    "TopPushK-5" => ("\\TopPushK(5)", 5, :tpr_at_k5),
    "TopPushK-10" => ("\\TopPushK(10)", 6, :tpr_at_k10),
    "GrillNP-0.01" => ("\\GrillNP(0.01)", 7, :tpr_at_fpr1),
    "GrillNP-0.05" => ("\\GrillNP(0.05)", 8, :tpr_at_fpr5),
    "TauFPL-0.01" => ("\\tauFPL(0.01)", 9, :tpr_at_fpr1),
    "TauFPL-0.05" => ("\\tauFPL(0.05)", 10, :tpr_at_fpr5),
    "PatMatNP-0.01" => ("\\PatMatNP(0.01)", 11, :tpr_at_fpr1),
    "PatMatNP-0.05" => ("\\PatMatNP(0.05)", 12, :tpr_at_fpr5),
)

map_loss(loss) = LOSS[loss][1]
order_loss(loss) = LOSS[loss][2]
map_default_metric(loss) = LOSS[loss][3]

metrics = [
    :tpr_at_k1 => (y, s) -> round_perc(tpr_at_k(y, s, 1)),
    :tpr_at_k5 => (y, s) -> round_perc(tpr_at_k(y, s, 5)),
    :tpr_at_k10 => (y, s) -> round_perc(tpr_at_k(y, s, 10)),
    :tpr_at_fpr1 => (y, s) -> round_perc(tpr_at_fpr(y, s, 0.01)),
    :tpr_at_fpr5 => (y, s) -> round_perc(tpr_at_fpr(y, s, 0.05)),
    :auc => (y, s) -> round_perc(roc_auc(y, s)),
]

const METRIC = Dict(
    :tpr_at_k1 => "True-positive rate at \$K = 1\$",
    :tpr_at_k5 => "True-positive rate at \$K = 5\$",
    :tpr_at_k10 => "True-positive rate at \$K = 10\$",
    :tpr_at_fpr1 => "True-positive rate at False-positive rate \$0.01\$",
    :tpr_at_fpr5 => "True-positive rate at False-positive rate \$0.05\$",
    :auc => "ROC AUC",
)

const DATADIR = "/home/machava2/projects/ClassificationAtTopExperiments.jl/data/dissertation/"
const RESULTSDIR = "/home/machava2/projects/ClassificationAtTopExperiments.jl/data/results"

const DATASETS = [
    "MNIST",
    "FashionMNIST",
    "CIFAR10",
    "CIFAR20",
    "CIFAR100",
    "SVHN2",
    "SVHN2Extra",
    "Nsf5Small",
    "JMiPODSmall",
]

#-------------------------------------------------------------------------------------------
# Loading DataFrames
#-------------------------------------------------------------------------------------------
force = false
file_primal = joinpath(DATADIR, "primal", "metrics.csv")
if !isfile(file_primal) || force
    df_primal = evaluation(joinpath(DATADIR, "primal"); metrics)
    CSV.write(file_primal, df_primal)
else
    df_primal = CSV.read(file_primal, DataFrame)
end

file_dual = joinpath(DATADIR, "dual", "metrics.csv")
if !isfile(file_dual) || force
    df_dual = evaluation(joinpath(DATADIR, "dual"); metrics)
    CSV.write(file_dual, df_dual)
else
    df_dual = CSV.read(file_dual, DataFrame)
end

file_primalNN = joinpath(DATADIR, "primalNN", "metrics.csv")
if !isfile(file_primalNN) || force
    df_primalNN = evaluation(joinpath(DATADIR, "primalNN"); metrics)
    CSV.write(file_primalNN, df_primalNN)
else
    df_primalNN = CSV.read(file_primalNN, DataFrame)
end

λs = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
ϑs = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

function isvalidrow_primal(row)
    return all([
        row.optimiser == "OptADAM",
        ismissing(row.λ) || in(row.λ, λs),
        ismissing(row.ϑ) || in(row.ϑ, ϑs),
        row.batch_neg == row.batch_pos == 256,
    ])
end

function isvalidrow_dual(row)
    return all([
        row.model == "Gaussian",
        ismissing(row.λ) || in(row.λ, λs),
        ismissing(row.ϑ) || in(row.ϑ, ϑs),
    ])
end

dfs = (
    "primal" => filter(isvalidrow_primal, df_primal),
    "dual" => filter(isvalidrow_dual, df_dual),
    "primalnn" => filter(isvalidrow_primal, df_primalNN),
)

#-------------------------------------------------------------------------------------------
# critical diagrams
#-------------------------------------------------------------------------------------------
to_join = [
    :dataset => [:dataset, :seed],
    :loss => [:loss, :τ, :K],
]

include_cols = [
    :id,
    :split,
    :dataset,
    :loss,
]

α = 0.05
y_step = 3.5
tikz_only = true
use_default_metric = true

for (name, df) in dfs
    df_joined = join_cols(df; metrics, to_join, include_cols)
    df_joined[:, :default_metric] .= map_default_metric.(df_joined.loss)
    df_ranks = rank_table(df_joined, first.(metrics); use_default_metric)

    path = joinpath(RESULTSDIR, "critical_diagrams_$(name).tex")
    mkpath(dirname(path))

    open(path, "w") do io
        ymin = 0
        write(io, "\\begin{tikzpicture}")
        for metric in reverse(first.(metrics))
            loss = map_loss.(df_ranks.loss)
            ranks = df_ranks[:, metric]
            cv = nemenyi_cd(length(loss), df_ranks.n_datasets[1]; α)

            title = "$(METRIC[metric])"
            write(io, critical_diagram(loss, ranks, cv; ymin, tikz_only, title))
            ymin += y_step
            write(io, "\n\n")
        end
        write(io, "\\end{tikzpicture}")
    end
end

#-------------------------------------------------------------------------------------------
# mean metrics
#-------------------------------------------------------------------------------------------
to_join = [
    :loss => [:loss, :τ, :K],
    :optimiser => [:optimiser, :eta],
]

include_cols = [
    :id,
    :split,
    :dataset,
    :seed,
    :loss,
]

use_default_metric = true

for (name, df) in dfs
    df_joined = join_cols(df; metrics, to_join, include_cols)
    df_joined[:, :default_metric] .= map_default_metric.(df_joined.loss)

    path = joinpath(RESULTSDIR, "mean_metrics_$(name).tex")
    path_csv = joinpath(RESULTSDIR, "mean_metrics_$(name).csv")
    mkpath(dirname(path))
    dfs_best = []

    open(path, "w") do io
        for metric in first.(metrics)
            df_best = best_table(df_joined, metric; split=:test, use_default_metric)
            perm = sortperm(order_loss.(df_best.loss))
            df_best.loss = map_loss.(df_best.loss)

            cols = ["loss", intersect(DATASETS, names(df_best)[2:end])...,]
            df_best = df_best[perm, cols]

            highlight_best = []
            for j in 2:size(df_best, 2)
                valmax = maximum(skipmissing(df_best[:, j]))
                for i in findall(==(valmax), skipmissing(df_best[:, j]))
                    push!(highlight_best, (i, j))
                end
            end

            highlight_worst = []
            for j in 2:size(df_best, 2)
                valmax = minimum(skipmissing(df_best[:, j]))
                for i in findall(==(valmax), skipmissing(df_best[:, j]))
                    push!(highlight_worst, (i, j))
                end
            end

            write(io, nice_table(
                df_best;
                caption="$(METRIC[metric])",
                label="$name $(metric)",
                highlight_best = [highlight_best...,],
                highlight_worst = [highlight_worst...,],
            ))
            write(io, "\n\n")
            insertcols!(df_best, 1, :metric => fill(metric, size(df_best, 1)))
            push!(dfs_best, df_best)
        end
    end
    CSV.write(path_csv, reduce(vcat, dfs_best))
end
