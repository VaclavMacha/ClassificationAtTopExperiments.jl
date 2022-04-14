@option "CPU" struct CPU end
@option "GPU" struct GPU end

materialize(::CPU) = Flux.cpu
materialize(::GPU) = Flux.cpu

Base.string(::CPU) = "CPU"
Base.string(::GPU) = "GPU"

@option struct TrainConfig
    seed::Int = 1234
    force::Bool = false
    iters::Int = 1000
    checkpoint_every::Int = 100
    device::Union{CPU,GPU} = CPU()
end

function Base.string(o::TrainConfig)
    vals = string.([o.seed, o.force, o.iters, o.checkpoint_every, o.device])
    return "Train($(join(vals, ", ")))"
end

# Save directory
function dir_name(
    Lconfig::LossConfig,
    Mconfig::ModelConfig,
    Dconfig::DataConfig,
    Oconfig::OptConfig,
    Tconfig::TrainConfig,
)

    return joinpath(
        "results",
        string.([Tconfig, Dconfig, Mconfig, Oconfig, Lconfig])...,
    )
end

# saving and loading model
load_model(path) = BSON.load(path, @__MODULE__)
save_model(path, model) = BSON.bson(path, model)

# experiment
function make_dict(
    Lconfig::LossConfig,
    Mconfig::ModelConfig,
    Dconfig::DataConfig,
    Oconfig::OptConfig,
    Tconfig::TrainConfig,
)

    d = merge(
        to_dict(Lconfig, YAMLStyle),
        to_dict(Mconfig, YAMLStyle),
        to_dict(Dconfig, YAMLStyle),
    )
    d["optimiser"] = to_dict(Oconfig, YAMLStyle)
    d["training"] = to_dict(Tconfig, YAMLStyle)

    return d
end

save_config(path, config) = YAML.write_file(path, config)
load_config(path) = YAML.load_file(path; dicttype=Dict{String,Any})

function parse_config(path)
    d = load_config(path)
    return (
        from_dict(LossConfig, Dict("loss" => d["loss"])),
        from_dict(ModelConfig, Dict("model" => d["model"])),
        from_dict(DataConfig, Dict("dataset" => d["dataset"])),
        from_dict(OptConfig, d["optimiser"]),
        from_dict(TrainConfig, d["training"]),
    )
end

run_experiments(path) = run_experiments(parse_config(path)...)

function run_experiments(
    Lconfig::LossConfig,
    Mconfig::ModelConfig,
    Dconfig::DataConfig,
    Oconfig::OptConfig,
    Tconfig::TrainConfig,
)

    # check if exists
    dir = dir_name(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig)
    if !Tconfig.force && isfile(datadir(dir, "solution.bson"))
        return load_model(datadir(dir, "solution.bson"))
    end

    # logging initialization
    mkpath(datadir(dir))
    mkpath(datadir(dir, "checkpoints"))
    p = Progress(; iter_max=Tconfig.iters)
    logger = generate_logger(datadir(dir))

    # run
    solution = []
    with_logger(logger) do
        @info "Preparing output dir..."
        save_config(
            datadir(dir, "config.yaml"),
            make_dict(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig)
        )

        # initialization
        @info "Experiment in progress..."
        device = materialize(Tconfig.device)
        train, test = load(Dconfig) |> device
        model, pars = materialize(Mconfig; device)
        optimiser = materialize(Oconfig)

        progress!(p; training=false)

        # training loop
        @info "Training in progress..."
        reset_time!(p)
        for iter in 1:Tconfig.iters
            if Oconfig.decay_step != 1 && mod(iter, Oconfig.decay_every) == 0
                optimiser.eta = max(optimiser.eta * Oconfig.decay_step, Oconfig.decay_min)
            end

            # gradient step
            local L_train
            local s_train
            grads = Flux.Zygote.gradient(pars) do
                s_train = model(train[1])
                L_train = loss(Lconfig, train[2], s_train, pars)
                return L_train
            end
            Flux.Optimise.update!(optimiser, pars, grads)
            append!(p.loss_train, L_train)

            # checkpoint
            if mod(iter, Tconfig.checkpoint_every) == 0 || iter == Tconfig.iters
                s_test = model(test[1])
                append!(p.loss_test, loss(Lconfig, test[2], s_test, pars))

                solution = Dict(
                    :iter => iter,
                    :model => deepcopy(cpu(model)),
                    :train => Dict(:y => train[2], :s => cpu(s_train)),
                    :test => Dict(:y => test[2], :s => cpu(s_test)),
                    :loss_train => p.loss_train,
                    :loss_test => p.loss_test,
                )
                save_model(datadir(dir, "checkpoints", "solution_iter=$(iter).bson"), solution)
            end
            progress!(p)
        end
        @info "Saving final solution..."
        if !isempty(solution)
            save_model(datadir(dir, "solution.bson"), solution)
        end
        progress!(p; training=false)
    end
    return solution
end