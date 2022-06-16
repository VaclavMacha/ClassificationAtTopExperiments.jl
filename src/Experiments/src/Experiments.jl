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
import CUDA
import EfficientNet
import MLDatasets
import Metalhead

using Base: @kwdef
using Flux.Losses: logitbinarycrossentropy
using ProgressMeter: durationstring, speedstring

# exports
export Nsf5, Nsf5Small
export JMiPOD, JMiPODSmall, JMiPODDebug
export MNIST, FashionMNIST
export CIFAR10, CIFAR20, CIFAR100
export SVHN2, SVHN2Extra

export CrossEntropy
export PatMat, PatMatNP
export TopPush, TopPushK, TopMeanTau, TauFPL
export Grill, GrillNP
export DeepTopPush, DeepTopPushCross

export Linear
export EfficientNetB0
export GoogLeNet
export MobileNetv3
export SimpleConv

export OptDescent
export OptADAM

export TrainConfig

export load_or_run
export load_config, write_config
export load_checkpoint, save_checkpoint

# Defaults paths
const TO = TimerOutput()
const PROJECT_DIR = get(ENV, "PROJECT_DIR", abspath(joinpath(@__DIR__, "../../../")))

datadir(args...) = joinpath(PROJECT_DIR, "data", args...)
datasetsdir(args...) = joinpath(PROJECT_DIR, "data", "datasets", args...)
pretraineddir(args...) = joinpath(PROJECT_DIR, "data", "pretrained", args...)

function solution_path(dir::AbstractString, epoch::Int=-1)
    if epoch < 0
        joinpath(dir, "solution.bson")
    else
        joinpath(dir, "checkpoints", "checkpoint_epoch=$(epoch).bson")
    end
end

config_path(dir::AbstractString) = joinpath(dir, "config.toml")
timer_path(dir::AbstractString) = joinpath(dir, "timer.json")
explog_path(dir::AbstractString) = joinpath(dir, "experiment.log")
errlog_path(dir::AbstractString) = joinpath(dir, "error.log")

# Saving and loading of checkpoints
load_checkpoint(path) = BSON.load(path, @__MODULE__)
save_checkpoint(path, model) = BSON.bson(path, model)

# TrainConfig
@kwdef struct TrainConfig
    seed::Int = 1234
    epoch_max::Int = 1000
    checkpoint_every::Int = 100
    batch_pos::Int = 0
    batch_neg::Int = 0
    force::Bool = false
    device::String = "CPU"
    save_dir::String = "results"
end

parse_type(::Val{:TrainConfig}) = TrainConfig
_exclude(::Type{TrainConfig}) = (:checkpoint_every, :device, :force, :save_dir)

# CUDA free memory
free_memory!(x) = nothing
free_memory!(x::CUDA.CuArray) = CUDA.unsafe_free!(x)

# includes
include("loaders.jl")
include("datasets.jl")
include("losses.jl")
include("models.jl")
include("optimisers.jl")
include("logging.jl")
include("configs.jl")
include("train.jl")
include("evaluation.jl")

end # module
