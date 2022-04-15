module Experiments

using AccuracyAtTopPrimal
using BSON
using Configurations
using Dates
using DrWatson
using EvalMetrics
using Flux
using HDF5
using Logging
using LoggingExtras
using MLUtils
using Plots
using Random
using Statistics
using YAML

using Flux.Losses: logitbinarycrossentropy
using ProgressMeter: durationstring, speedstring

# exports
export DataConfig

export LossConfig
export CrossEntropy
export AATP
export Hinge, Quadratic, PatMatType, TopPushType

export ModelConfig
export Linear

export OptConfig
export TrainConfig

export run_experiments
export make_dict, save_config, load_config, parse_config
export load_model, save_model

# includes
dir_string(x) = string(x)
dir_string(x::Real) = round(Float64(x); digits=8)
dir_string(x::Int64) = string(x)
dir_string(x::Bool) = string(x)

include("datasets.jl")
include("losses.jl")
include("models.jl")
include("optimisers.jl")
include("logging.jl")
include("train.jl")

end # module
