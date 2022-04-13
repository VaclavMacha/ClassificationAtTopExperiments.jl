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
using Plots
using Random
using Statistics
using YAML

using Flux.Losses: logitbinarycrossentropy
using ProgressMeter: durationstring, speedstring

# exports
export DataConfig
export Nsf5_01
export Nsf5_02
export Nsf5_05

export LossConfig
export CrossEntropy
export AATP
export Hinge, Quadratic, PatMatType, TopPushType

export ModelConfig
export Linear

export OptConfig
export TrainConfig

export run_experiments
export save_config, load_config, parse_config

# includes
include("datasets.jl")
include("losses.jl")
include("models.jl")
include("optimisers.jl")
include("logging.jl")
include("train.jl")

end # module
