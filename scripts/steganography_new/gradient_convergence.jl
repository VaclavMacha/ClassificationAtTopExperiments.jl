#!/usr/bin/env sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#=

module load fosscuda
module load cuDNN/8.0.5.39-CUDA-11.1.1
module load --ignore-cache Julia/1.7.3-linux-x86_64

export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_CUDA_MEMORY_POOL=none
export DATADEPS_ALWAYS_ACCEPT=true

srun --unbuffer julia --color=no --startup-file=no --threads=auto "${BASH_SOURCE[0]}" "$@"
exit
=#

using DrWatson
using Random

quickactivate("ClassificationAtTopExperiments.jl")

using Experiments
using Experiments.Flux
using Experiments.JSON3
using Experiments.Zygote
using Experiments.StatsBase
using Experiments.AccuracyAtTop
using Statistics: norm, dot
using Evaluation.Plots

ENV["GKSwstype"] = "nul"  # For GR backend
ENV["MPLBACKEND"] = "Agg"  # For PyPlot backend
gr()

function Experiments.datadir(args...)
    return joinpath("/mnt/personal/machava2/experiments", args...)
end

function Experiments.datasetsdir(args...)
    return joinpath("/mnt/personal/machava2/datasets", args...)
end

function Experiments.pretraineddir(args...)
    return joinpath("/mnt/personal/machava2/pretrained", args...)
end

function plot_history(history, save_path = nothing)

    n_iterations = length(history[:sampled_t])
    iterations = 1:n_iterations

    # Create plot with custom styling
    p = plot(layout = (2, 1), size = (900, 700), dpi = 300)

    # Subplot 1: Threshold comparison
    plot!(p[1], iterations, history[:sampled_t],
          label = "Sampled threshold",
          linewidth = 2.5,
          color = :gray,
          linestyle = :dot,
          alpha = 0.8)

    plot!(p[1], iterations, history[:enhanced_t],
          label = "Enhanced threshold",
          linewidth = 2.5,
          color = :steelblue,
          alpha = 0.8)

    plot!(p[1], iterations, history[:actual_t],
          label = "Actual threshold",
          linewidth = 2.5,
          color = :red,
          linestyle = :dash,
          alpha = 0.8)

    # Styling for subplot 1
    plot!(p[1],
          xlabel = "Iteration",
          ylabel = "Threshold",
          grid = true,
          gridwidth = 1,
          gridcolor = :lightgray,
          legend = :topright)

    # Subplot 2: Gradient angles with statistics
    plot!(p[2], iterations, history[:sampled_angle],
          label = "Angle between sampled and actual gradient",
          linewidth = 2.5,
          color = :gray,
          linestyle = :dot,
          alpha = 0.8)

    plot!(p[2], iterations, history[:enhanced_angle],
          label = "Angle between enhanced and actual gradient",
          linewidth = 2.5,
          color = :steelblue,
          alpha = 0.8)

    # Reference lines
    hline!(p[2], [90], linestyle = :dash, color = :red, alpha = 0.7,
           linewidth = 2, label = "90° (Orthogonal)")
    hline!(p[2], [0], linestyle = :dash, color = :red, alpha = 0.7,
           linewidth = 2, label = "0° (Parallel)")

    # Styling for subplot 2
    plot!(p[2],
          xlabel = "Iteration",
          ylabel = "Angle (degrees)",
          grid = true,
          gridwidth = 1,
          gridcolor = :lightgray,
          legend = :topright,
          ylims = (0, 180))

    # Save if path provided
    if save_path !== nothing
        savefig(p, save_path)
        println("Plot saved to: $save_path")
    end
end

# Settings
seed = 1234
iter_max = 1000
batch_neg = 16
batch_pos = 16
eta=0.000001
replace_files= true
surrogate="Hinge"
warm_up = 50

checkpoint_path = nothing
# checkpoint_path = Experiments.datadir("steganography_new/Nsf5Small(0.2, 0.1, 0.375, 0.125)/TrainConfig(1, 3000, 0, 0)/OptADAM(0.01, 0.9, 0.999, 50, 0.8)/Linear()/DeepTopPush(0.0, Hinge)/checkpoints/checkpoint_epoch=3000.bson")

Random.seed!(seed)
settings = (; seed, iter_max, batch_neg, batch_pos, eta, surrogate)
settings_str = replace(string(settings)[2:end-1], ", " => "_", " = " => "=", "\"" => "")
save_dir = "/home/machava2/projects/ClassificationAtTopExperiments.jl/tmp/convergence/$(settings_str)"
mkpath(save_dir)

# experiment
if !isfile("$(save_dir)/history.json") || !replace_files
    # Initialization
    dataset = Nsf5Small(payload=0.2, ratio=1)
    loss_type = Experiments.DeepTopPush(λ=0, surrogate=surrogate)
    opt_type = OptDescent(eta=eta, decay_every = 0)

    if checkpoint_path != nothing
        checkpoint = Experiments.load_checkpoint(checkpoint_path)
        model = checkpoint[:model]
    else
        model_type = Linear()
        model = Experiments.materialize(dataset, model_type)
    end

    pars = Flux.params(model)
    loss = Experiments.materialize(loss_type)
    opt = Experiments.materialize(opt_type, iter_max)

    # data loading
    train, valid, test = Experiments.load(dataset)
    X = train.features
    Y = train.targets

    data = Experiments.DataWrapper(
        X[:, vec(Y .== 0)],
        X[:, vec(Y .== 1)],
        sum(Y .== 0),
        sum(Y .== 1),
        batch_neg,
        batch_pos,
        vec(1:batch_neg),
        vec(1:batch_pos)
    )

    # history
    history = Dict(
        :sampled_loss => Float32[],
        :sampled_t => Float32[],
        :sampled_t_ind => Int[],
        :sampled_angle => Float32[],
        :enhanced_loss => Float32[],
        :enhanced_t => Float32[],
        :enhanced_t_ind => Int[],
        :enhanced_angle => Float32[],
        :actual_loss => Float32[],
        :actual_t => Float32[],
        :actual_t_ind => Int[],
    )

    # train loop
    enhanced_t_ind = 0
    for i in 1:iter_max
        global enhanced_t_ind
        println("$(i)/$(iter_max) ($(round(100*i/iter_max; digits=2))%)")

        # prepare random batch
        Experiments.update_batch(data)

        if i < warm_up
            enhanced_t_ind = argmax(vec(model(data.X_neg)))
        end

        # computation of gradients
        sampled = Experiments.sampled_gradient(model, pars, loss, data)
        enhanced = Experiments.enhanced_gradient(model, pars, loss, data, enhanced_t_ind)
        actual = Experiments.actual_gradient(model, pars, loss, data)

        # update history
        append!(history[:sampled_loss], sampled.loss)
        append!(history[:sampled_t_ind], sampled.threshold_ind)
        append!(history[:sampled_t], sampled.threshold)

        append!(history[:enhanced_loss], enhanced.loss)
        append!(history[:enhanced_t_ind], enhanced.threshold_ind)
        append!(history[:enhanced_t], enhanced.threshold)

        append!(history[:actual_loss], actual.loss)
        append!(history[:actual_t_ind], actual.threshold_ind)
        append!(history[:actual_t], actual.threshold)

        # compute angles between gradients
        append!(history[:sampled_angle], Experiments.gradient_angle(actual.grads, sampled.grads))
        append!(history[:enhanced_angle], Experiments.gradient_angle(actual.grads, enhanced.grads))

        # update model
        Flux.Optimise.update!(opt, pars, enhanced.grads)
        enhanced_t_ind = enhanced.threshold_ind
    end

    open("$(save_dir)/history.json", "w") do file
        JSON3.pretty(file, history)  # 2 spaces indentation
    end
end

if !isfile("$(save_dir)/convergence.png") || !replace_files
    plot_history(history, "$(save_dir)/convergence.png")
end
