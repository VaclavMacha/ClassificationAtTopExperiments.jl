using DrWatson
@quickactivate("ClassificationAtTopExperiments.jl")

using Evaluation
using Evaluation.Plots

using Evaluation: plot_roc, get_roc

#-------------------------------------------------------------------------------------------
# Utilities
#-------------------------------------------------------------------------------------------
const LOSS = Dict(
    "CrossEntropy-0.5" => ("CrossEntropy(0.5)", 1, :auc),
    "CrossEntropy-0.9" => ("CrossEntropy(0.9)", 2, :auc),
    "CrossEntropy-0.99" => ("CrossEntropy(0.99)", 3, :auc),
    "CrossEntropy-0.999" => ("CrossEntropy(0.999)", 4, :auc),
    "CrossEntropy-0.1" => ("CrossEntropy(0.1)", 5, :auc),
    "CrossEntropy-0.01" => ("CrossEntropy(0.01)", 6, :auc),
    "CrossEntropy-0.001" => ("CrossEntropy(0.001)", 7, :auc),
    "TopPush" => ("TopPush", 8, :tpr_at_k1),
    "DeepTopPush" => ("DeepTopPush", 9, :tpr_at_k1),
    "PatMatNP-1.0e-5" => ("PatMatNP(\$10^{-5}\$)", 10, :tpr_at_fpr00001),
    "PatMatNP-0.0001" => ("PatMatNP(\$10^{-4}\$)", 11, :tpr_at_fpr0001),
    "PatMatNP-0.001" => ("PatMatNP(\$10^{-3}\$)", 12, :tpr_at_fpr001),
)

map_loss(loss) = LOSS[loss][1]
order_loss(loss) = LOSS[loss][2]
map_default_metric(loss) = LOSS[loss][3]

metrics = [
    :auc => (y, s) -> round_perc(roc_auc(y, s)),
    :tpr_at_k1 => (y, s) -> round_perc(tpr_at_k(y, s, 1)),
    :tpr_at_k5 => (y, s) -> round_perc(tpr_at_k(y, s, 5)),
    :tpr_at_k10 => (y, s) -> round_perc(tpr_at_k(y, s, 10)),
    :tpr_at_fpr00001 => (y, s) -> round_perc(tpr_at_fpr(y, s, 0.00001)),
    :tpr_at_fpr0001 => (y, s) -> round_perc(tpr_at_fpr(y, s, 0.0001)),
    :tpr_at_fpr001 => (y, s) -> round_perc(tpr_at_fpr(y, s, 0.001)),
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
force = false
file = joinpath(DATADIR, "metrics.csv")
if !isfile(file) || force
    df = evaluation(DATADIR; metrics)
    CSV.write(file, df)
else
    df = CSV.read(file, DataFrame)
end


function isvalidrow_jmipod(row)
    return all([
        String(row.dataset) == "JMiPODSmall",
        row.ratio == 1,
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.9),
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.99),
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.999),
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.1),
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.01),
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.001),
    ])
end

function isvalidrow_nsf5(row)
    return all([
        String(row.dataset) == "Nsf5Small",
        row.ratio == 1,
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.9),
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.99),
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.999),
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.1),
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.01),
        !(String(row.loss) == "CrossEntropy" && row.ϵ == 0.001),
    ])
end

data_frames = [
    ("JMiPOD", filter(isvalidrow_jmipod, df)),
    ("NSF5", filter(isvalidrow_nsf5, df)),
]
# #-------------------------------------------------------------------------------------------
# # critical diagrams
# #-------------------------------------------------------------------------------------------
# to_join = [
#     :dataset => [:dataset, :ratio, :seed],
#     :loss => [:loss, :τ, :K, :ϵ],
#     :optimiser => [:optimiser, :eta],
# ]

# include_cols = [
#     :id,
#     :split,
#     :dataset,
#     :loss,
# ]

# α = 0.05
# y_step = 3.5


# df_joined = join_cols(df; metrics, to_join, include_cols)
# df_ranks = rank_table(df_joined, first.(metrics))

# path = joinpath(DATADIR, "results", "critical_diagrams.tex")
# mkpath(dirname(path))

# open(path, "w") do io
#     ymin = 0
#     for metric in reverse(first.(metrics))
#         loss = map_loss.(df_ranks.loss)
#         ranks = df_ranks[:, metric]
#         cv = nemenyi_cd(length(loss), df_ranks.n_datasets[1]; α)

#         write(io, critical_diagram(loss, ranks, cv; ymin, title="$(METRIC[metric])"))
#         ymin += y_step
#     end
# end

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

use_default_metric = true

for (name, df) in data_frames
    df_joined = join_cols(df; metrics, to_join, include_cols)
    path = joinpath(DATADIR, "results", "metrics_$(name).tex")

    mkpath(dirname(path))
    open(path, "w") do io
        dfs = []
        for metric in first.(metrics)
            df_best = best_table(df_joined, metric; split=:test, use_default_metric)
            rename!(df_best, [:Formulation, metric])
            push!(dfs, df_best)
        end

        df_best = innerjoin(dfs...; on=:Formulation)
        perm = sortperm(order_loss.(df_best.Formulation))
        df_best.Formulation = map_loss.(df_best.Formulation)
        df_best = df_best[perm, :]

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
            caption="$(name)",
            label="$(name)",
            highlight_best=[highlight_best...,],
            highlight_worst=[highlight_worst...,]
        ))
        write(io, "\n\n")
    end
end

#-------------------------------------------------------------------------------------------
# Plots
#-------------------------------------------------------------------------------------------
const LOSS_MAP = Dict(
    "CrossEntropy-0.5" => "baseline",
    "DeepTopPush" => "deeptoppush",
    "PatMatNP-1.0e-5" => "patmatnp_1e-5",
    "PatMatNP-0.0001" => "patmatnp_1e-4",
    "PatMatNP-0.001" => "patmatnp_1e-3",
)

const DATASET_MAP = Dict(
    "Nsf5Small-1.0" => "nsf5",
    "JMiPODSmall-1.0" => "jmipod",
)

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

use_default_metric = true


dfss = last.(data_frames)
df_all = vcat(
    dfss[1][dfss[1].seed.==4, :],
    dfss[2][dfss[2].seed.==7, :],
)
df_joined = join_cols(df_all; metrics, to_join, include_cols)
df_joined[:, :default_metric] .= map_default_metric.(df_joined.loss)
df_best = Evaluation.select_best(df_joined, [:dataset, :loss])

function find_best(df, ids, split=:test)
    return findall(row -> row.id in ids && Symbol(row.split) == split, eachrow(df))
end

for row in eachrow(df_best)
    dir = dirname(df_all.file_solution[find_best(df_all, row.id)[1]])
    _, df_roc = get_roc(dir)


    loss = LOSS_MAP[row.loss]
    dataset = DATASET_MAP[row.dataset]
    path_csv = joinpath(DATADIR, "results", "stego_$(dataset)_$(loss).csv")
    mkpath(dirname(path_csv))
    CSV.write(path_csv, df_roc)
end
