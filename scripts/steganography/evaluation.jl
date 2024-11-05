using DrWatson
@quickactivate("ClassificationAtTopExperiments.jl")

using Evaluation
using Evaluation.Plots

using Evaluation: plot_roc, get_roc

#-------------------------------------------------------------------------------------------
# Utilities
#-------------------------------------------------------------------------------------------
const LOSS_MAP = Dict(
    "CrossEntropy-0.5" => Dict(
        "name" => "CrossEntropy(0.5)",
        "latex_name" => "CrossEntropy(0.5)",
        "order" => 1,
        "default_metric" => :auc,
    ),
    "CrossEntropy-0.9" => Dict(
        "name" => "CrossEntropy(0.9)",
        "latex_name" => "CrossEntropy(0.9)",
        "order" => 2,
        "default_metric" => :auc,
    ),
    "CrossEntropy-0.99" => Dict(
        "name" => "CrossEntropy(0.99)",
        "latex_name" => "CrossEntropy(0.99)",
        "order" => 3,
        "default_metric" => :auc,
    ),
    "CrossEntropy-0.999" => Dict(
        "name" => "CrossEntropy(0.999)",
        "latex_name" => "CrossEntropy(0.999)",
        "order" => 4,
        "default_metric" => :auc,
    ),
    "CrossEntropy-0.1" => Dict(
        "name" => "CrossEntropy(0.1)",
        "latex_name" => "CrossEntropy(0.1)",
        "order" => 5,
        "default_metric" => :auc,
    ),
    "CrossEntropy-0.01" => Dict(
        "name" => "CrossEntropy(0.01)",
        "latex_name" => "CrossEntropy(0.01)",
        "order" => 6,
        "default_metric" => :auc,
    ),
    "CrossEntropy-0.001" => Dict(
        "name" => "CrossEntropy(0.001)",
        "latex_name" => "CrossEntropy(0.001)",
        "order" => 7,
        "default_metric" => :auc,
    ),
    "TopPush" => Dict(
        "name" => "TopPush",
        "latex_name" => "TopPush",
        "order" => 8,
        "default_metric" => :tpr_at_k1,
    ),
    "DeepTopPush" => Dict(
        "name" => "DeepTopPush",
        "latex_name" => "DeepTopPush",
        "order" => 9,
        "default_metric" => :tpr_at_k1,
    ),
    "PatMatNP-1.0e-5" => Dict(
        "name" => "PatMat-NP(1e-5)",
        "latex_name" => "PatMatNP(\$10^{-5}\$)",
        "order" => 10,
        "default_metric" => :tpr_at_fpr00001,
    ),
    "PatMatNP-0.0001" => Dict(
        "name" => "PatMat-NP(1e-4)",
        "latex_name" => "PatMatNP(\$10^{-4}\$)",
        "order" => 11,
        "default_metric" => :tpr_at_fpr0001,
    ),
    "PatMatNP-0.001" => Dict(
        "name" => "PatMat-NP(1e-3)",
        "latex_name" => "PatMatNP(\$10^{-3}\$)",
        "order" => 12,
        "default_metric" => :tpr_at_fpr001,
    ),
)

loss_map(loss::AbstractString, key::AbstractString) = LOSS_MAP[loss][key]

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

#-------------------------------------------------------------------------------------------
# Loading DataFrames
#-------------------------------------------------------------------------------------------
results_subdir = "results_tifs_2024-11-04"
force = false
if !isdir(joinpath(DATADIR, results_subdir))
    mkdir(joinpath(DATADIR, results_subdir))
end

file = joinpath(DATADIR, results_subdir, "metrics.csv")
if !isfile(file) || force
    df = evaluation(DATADIR; metrics)
    df[:, "batch_size"] = df[:, "batch_pos"] + df[:, "batch_neg"]

    CSV.write(file, df)
else
    df = CSV.read(file, DataFrame)
end

function isvalidrow(
    row,
    dataset::String;
    ratios::Vector{<:Real}=Float64[],
    batch_sizes::Vector{<:Real}=Float64[]
)
    return all([
        String(row.dataset) == dataset,
        isempty(ratios) || in(row.ratio, ratios),
        isempty(batch_sizes) || in(row.batch_size, batch_sizes),
        !(String(row.loss) == "CrossEntropy" && in(row.ϵ, [0.1, 0.01, 0.001])),
    ])
end

function alias(row)
    dataset = replace(string(row.dataset), "JMiPOD" => "J-MiPOD")
    payload = lpad(Int(100 * row.payload), 3, "0")
    ratio = lpad(Int(100 * row.ratio), 3, "0")
    batch_size = row.batch_size
    return "$(dataset)_payload=$(payload)_ratio=$(ratio)_batchsize=$(batch_size)"
end

ratios = [0.1, 0.5, 1]
batch_sizes = [32, 64, 256]
final_df = vcat(
    filter(row -> isvalidrow(row, "JMiPODSmall"; ratios, batch_sizes=[32, 64, 256]), df),
    filter(row -> isvalidrow(row, "Nsf5Small"; ratios, batch_sizes=[0, 32]), df),
)
final_df[:, "dataset_alias"] = alias.(eachrow(final_df))


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

# path = joinpath(DATADIR, results_subdir, "critical_diagrams.tex")
# mkpath(dirname(path))

# open(path, "w") do io
#     ymin = 0
#     for metric in reverse(first.(metrics))
#         loss = loss_map.(df_ranks.loss, "latex_name")
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
    :loss => [:loss, :τ, :K, :ϵ],
    :optimiser => [:optimiser, :eta],
]

include_cols = [
    :id,
    :dataset_alias,
    :split,
    :seed,
    :loss,
    :file_solution
]

use_default_metric = true

for agg in [mean, median]
    for (keys, grp) in pairs(groupby(final_df, ["dataset_alias"]))
        name = keys["dataset_alias"]

        grp_joined = join_cols(grp; metrics, to_join, include_cols)
        rename!(grp_joined, :dataset_alias => :dataset)
        grp_joined[:, "default_metric"] = loss_map.(grp_joined[:, "loss"], "default_metric")

        path = joinpath(DATADIR, results_subdir, "metrics_$(agg)_$(name).tex")

        mkpath(dirname(path))
        open(path, "w") do io
            dfs = []
            for metric in first.(metrics)
                df_best = best_table(grp_joined, metric; split=:test, use_default_metric, agg)
                rename!(df_best, [:Formulation, metric])
                push!(dfs, df_best)
            end

            df_best = innerjoin(dfs...; on=:Formulation)
            perm = sortperm(loss_map.(df_best.Formulation, "order"))
            df_best.Formulation = loss_map.(df_best.Formulation, "latex_name")
            df_best = df_best[perm, :]

            highlight_best = []
            highlight_worst = []
            for j in axes(df_best, 2)[2:end]
                valmin, valmax = extrema(skipmissing(df_best[:, j]))
                for i in findall(==(valmin), skipmissing(df_best[:, j]))
                    push!(highlight_worst, (i, j))
                end
                for i in findall(==(valmax), skipmissing(df_best[:, j]))
                    push!(highlight_best, (i, j))
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
end

#-------------------------------------------------------------------------------------------
# Plots
#-------------------------------------------------------------------------------------------
function find_best(df)
    id = Evaluation.select_best(DataFrame(df), [:dataset, :loss]).id[1]
    return df[(df[:, "id"].==id).&(df[:, "split"].=="test"), :]
end

for (keys_tmp, grp_tmp) in pairs(groupby(final_df, ["dataset_alias"]))
    dataset_alias = keys_tmp["dataset_alias"]

    grp_joined = join_cols(grp_tmp; metrics, to_join, include_cols)
    rename!(grp_joined, :dataset_alias => :dataset)
    grp_joined[:, "default_metric"] = loss_map.(grp_joined[:, "loss"], "default_metric")

    for (keys, grp) in pairs(groupby(grp_joined, ["loss", "seed"]))
        loss_alias = loss_map(keys["loss"], "name")
        seed = keys["seed"]

        _, df_roc = get_roc(dirname(find_best(grp)[1, "file_solution"]))

        path_csv = joinpath(DATADIR, results_subdir, dataset_alias, string(seed), "$(loss_alias).csv")
        mkpath(dirname(path_csv))
        CSV.write(path_csv, df_roc)
    end
end