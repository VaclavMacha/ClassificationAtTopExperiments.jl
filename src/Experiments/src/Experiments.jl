module Experiments

using AccuracyAtTopPrimal
using BSON
using Configurations
using DrWatson
using Flux
using HDF5
using Random
using Statistics

using Flux.Losses: logitbinarycrossentropy

# exports
export Dataset
export CrossEntropy
export PatMatObjective
export Model
export Optimiser
export Training

export load
export materialize
export loss
export run_experiments

# abstract types
abstract type Objective end

# includes
include("datasets.jl")
include("objectives.jl")
include("models.jl")
include("optimisers.jl")
include("train.jl")

end # module
