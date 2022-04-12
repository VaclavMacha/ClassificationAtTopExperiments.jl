module Experiments

using AccuracyAtTopPrimal
using Configurations
using DrWatson
using Flux
using HDF5
using Random
using Statistics

using Flux.Losses: logitbinarycrossentropy

export Dataset
export CrossEntropy
export PatMatObjective
export Model

export load
export materialize
export loss

# abstract types
abstract type Objective end

include("datasets.jl")
include("objectives.jl")
include("models.jl")

end # module
