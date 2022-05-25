module Experiments

using BSON
using Dates
using Flux
using HDF5
using ImageCore
using JpegTurbo
using JSON3
using Logging
using LoggingExtras
using MLUtils
using Random
using Statistics
using StatsBase
using TimerOutputs
using TOML

import AccuracyAtTopPrimal
import AccuracyAtTop
import EfficientNet
import MLDatasets

using Base: @kwdef
using Flux.Losses: logitbinarycrossentropy
using MLDatasets: SupervisedDataset, FileDataset
using ProgressMeter: durationstring, speedstring

# exports
export Nsf5, Nsf5Small
export JMiPOD, JMiPODSmall
export MNIST, FashionMNIST
export CIFAR10, CIFAR20, CIFAR100
export SVHN2, SVHN2Extra

export CrossEntropy
export PatMat, PatMatNP
export TopPush, TopPushK, TopMeanTau, TauFPL
export DeepTopPush

export Linear
export EfficientNetB0

export OptDescent
export OptADAM

export TrainConfig

export load_or_run
export load_config, write_config
export load_checkpoint, save_checkpoint

# basics
const TO = TimerOutput()
const PROJECT_DIR = get(ENV, "PROJECT_DIR", abspath(joinpath(@__DIR__, "../../../")))

datadir(args...) = joinpath(PROJECT_DIR, "data", args...)

const DATASETS_DIR = get(ENV, "DATASETS_DIR", datadir("datasets"))

datasetsdir(args...) = joinpath(DATASETS_DIR, args...)

@kwdef struct TrainConfig
    seed::Int = 1234
    epoch_max::Int = 1000
    checkpoint_every::Int = 100
    batch_pos::Int = 0
    batch_neg::Int = 0
    force::Bool = false
    device::String = "CPU"
end

parse_type(::Val{:TrainConfig}) = TrainConfig
_exclude(::Type{TrainConfig}) = (:checkpoint_every, :device, :force)

# includes
include("datasets.jl")
include("batchloader.jl")
include("losses.jl")
include("models.jl")
include("optimisers.jl")
include("logging.jl")
include("configs.jl")
include("train.jl")

end # module
