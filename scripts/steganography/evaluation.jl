using DrWatson
@quickactivate("ClassificationAtTopExperiments.jl")

using Evaluation

#-------------------------------------------------------------------------------------------
# Utilities
#-------------------------------------------------------------------------------------------
const LOSS = Dict(
    "CrossEntropy-0.5" => ("CrossEntropy(0.5)", 1),
    "CrossEntropy-0.9" => ("CrossEntropy(0.9)", 2),
    "CrossEntropy-0.99" => ("CrossEntropy(0.99)", 3),
    "CrossEntropy-0.999" => ("CrossEntropy(0.999)", 4),
    "CrossEntropy-0.1" => ("CrossEntropy(0.1)", 5),
    "CrossEntropy-0.01" => ("CrossEntropy(0.01)", 6),
    "CrossEntropy-0.001" => ("CrossEntropy(0.001)", 7),
    "TopPush" => ("TopPush", 8),
    "DeepTopPush" => ("DeepTopPush", 9),
    "PatMatNP-0.001" => ("PatMatNP(\$10^{-3}\$)", 10),
    "PatMatNP-0.0001" => ("PatMatNP(\$10^{-4}\$)", 11),
    "PatMatNP-1.0e-5" => ("PatMatNP(\$10^{-5}\$)", 12),
)

map_loss(loss) = LOSS[loss][1]
order_loss(loss) = LOSS[loss][2]

metrics = [
    :tpr_at_k1 => (y, s) -> round_perc(tpr_at_k(y, s, 1)),
    :tpr_at_k5 => (y, s) -> round_perc(tpr_at_k(y, s, 5)),
    :tpr_at_k10 => (y, s) -> round_perc(tpr_at_k(y, s, 10)),
    :tpr_at_fpr00001 => (y, s) -> round_perc(tpr_at_fpr(y, s, 0.00001)),
    :tpr_at_fpr0001 => (y, s) -> round_perc(tpr_at_fpr(y, s, 0.0001)),
    :tpr_at_fpr001 => (y, s) -> round_perc(tpr_at_fpr(y, s, 0.001)),
    :auc => (y, s) -> round_perc(roc_auc(y, s)),
]

const METRIC = Dict(
    :tpr_at_k1 => "True-positive rate at \$K = 1\$",
    :tpr_at_k5 => "True-positive rate at \$K = 5\$",
    :tpr_at_k10 => "True-positive rate at \$K = 10\$",
    :tpr_at_fpr00001 => "True-positive rate at False-positive rate \$10^{-5}\$",
    :tpr_at_fpr0001 => "True-positive rate at False-positive rate \$10^{-4}\$",
    :tpr_at_fpr001 => "True-positive rate at False-positive rate \$10^{-3}\$",
    :auc => "ROC AUC",
)

const DATADIR = "/home/machava2/projects/ClassificationAtTopExperiments.jl/data/steganography"

const DATASETS = [
    "Nsf5Small-1.0",
    "Nsf5Small-0.5",
    "Nsf5Small-0.1",
    "Nsf5-1.0",
    "Nsf5-0.5",
    "Nsf5-0.1",
    "JMiPODSmall-1.0",
    "JMiPODSmall-0.5",
    "JMiPODSmall-0.1",
    "JMiPOD-1.0",
    "JMiPOD-0.5",
    "JMiPOD-0.1",
]

#-------------------------------------------------------------------------------------------
# Loading DataFrames
#-------------------------------------------------------------------------------------------
force = true
file = joinpath(DATADIR, "metrics.csv")
if !isfile(file) || force
    df = evaluation(DATADIR; metrics)
    CSV.write(file, df)
else
    df = CSV.read(file, DataFrame)
end

#-------------------------------------------------------------------------------------------
# critical diagrams
#-------------------------------------------------------------------------------------------
to_join = [
    :dataset => [:dataset, :ratio, :seed],
    :loss => [:loss, :τ, :K, :ϵ],
    :optimiser => [:optimiser, :eta],
]

include_cols = [
    :id,
    :split,
    :dataset,
    :loss,
]

α = 0.05
y_step = 3.5


df_joined = join_cols(df; metrics, to_join, include_cols)
df_ranks = rank_table(df_joined, first.(metrics))

path = joinpath(DATADIR, "results", "critical_diagrams.tex")
mkpath(dirname(path))

open(path, "w") do io
    ymin = 0
    for metric in reverse(first.(metrics))
        loss = map_loss.(df_ranks.loss)
        ranks = df_ranks[:, metric]
        cv = nemenyi_cd(length(loss), df_ranks.n_datasets[1]; α)

        write(io, critical_diagram(loss, ranks, cv; ymin, title="$(METRIC[metric])"))
        ymin += y_step
    end
end

#-------------------------------------------------------------------------------------------
# mean metrics
#-------------------------------------------------------------------------------------------
to_join = [
    :dataset => [:dataset, :ratio],
    :loss => [:loss, :τ, :K, :ϵ],
    :optimiser => [:optimiser, :eta],
]

include_cols = [
    :id,
    :split,
    :dataset,
    :seed,
    :loss,
]

df_joined = join_cols(df; metrics, to_join, include_cols)

path = joinpath(DATADIR, "results", "mean_metrics.tex")
mkpath(dirname(path))

open(path, "w") do io
    for metric in first.(metrics)
        df_best = best_table(df_joined, metric; split=:test)
        perm = sortperm(order_loss.(df_best.loss))
        df_best.loss = map_loss.(df_best.loss)

        cols = ["loss", intersect(DATASETS, names(df_best)[2:end])...,]
        df_best = df_best[perm, cols]

        highlight = Vector{NTuple{2,Int}}(undef, size(df_best, 2) - 1)
        for j in 2:size(df_best, 2)
            highlight[j-1] = (argmax(skipmissing(df_best[:, j])), j)
        end

        write(io, nice_table(
            df_best;
            caption="$(METRIC[metric])",
            label="$(metric)",
            highlight
        ))
        write(io, "\n\n")
    end
end
