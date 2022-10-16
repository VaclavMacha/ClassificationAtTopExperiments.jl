module Evaluation

using Reexport

@reexport using CSV
@reexport using DataFrames
@reexport using Dates
@reexport using Distributions
@reexport using EvalMetrics
@reexport using Measurements
@reexport using Plots
@reexport using Statistics
@reexport using StatsBase
@reexport using Experiments


using ProgressMeter
using TOML

using Experiments: LossType, load_checkpoint, parse_config, _string
using Experiments: solution_path, config_path, timer_path, explog_path, errlog_path

export evaluation, join_cols, round_perc, rank_table
export tpr_at_fpr, tpr_at_k, pos_at_top_k, roc_auc
export summary_dataset
export critical_diagram
export nemenyi_cd, friedman_test_statistic, friedman_critval
export best_table, nice_table

include("utilities.jl")
include("latexutils.jl")

function __init__()
    get!(ENV, "DATADEPS_LOAD_PATH", "/mnt/personal/machava2/datasets")
end

end # module
