@option struct TrainConfig
    seed::Int = 1234
    force::Bool = false
    buffer::Bool = false
    epochs::Int = 1000
    checkpoint_every::Int = 100
    batch_size::Int = 0
    batch_pos::Int = 0
    batch_neg::Int = 0
    device::String = "CPU"
end

materialize(t::TrainConfig) = materialize(Val(Symbol(t.device)))
materialize(::Val{:CPU}) = Flux.cpu
materialize(::Val{:GPU}) = Flux.gpu

function Base.string(o::TrainConfig)
    vals = dir_string.((
        o.seed,
        o.buffer,
        o.epochs,
        o.checkpoint_every,
        o.batch_size,
        o.batch_pos,
        o.batch_neg,
    ))
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
        string.((Tconfig, Dconfig, Mconfig, Oconfig, Lconfig))...,
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
    )
    d["dataset"] = to_dict(Dconfig, YAMLStyle)
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
        from_dict(DataConfig, d["dataset"]),
        from_dict(OptConfig, d["optimiser"]),
        from_dict(TrainConfig, d["training"]),
    )
end

function eval_model(
    Lconfig,
    model,
    pars,
    data;
    device = identity,
    batch_size = 1000,
)
    x, y = data
    s = similar(x, 1, length(y))

    for inds in partition(1:length(y), batch_size)
        xi = getobs(x, inds) |> device
        s[1, inds] .= cpu(model(xi))[:]
    end
    return s, loss(Lconfig, y, s, pars)
end

function create_batches(c::TrainConfig, train)
    if c.batch_size == 0
        return (train, )
    end

    loader = BatchLoader(train...; c.buffer, c.batch_neg, c.batch_pos, c.device)
    iters = ceil(Int, (length(c.batch_neg) + length(c.batch_pos)) / c.batch_size)
    return (loader() for _ in 1:iters)
end

run_experiments(path) = run_experiments(parse_config(path)...)

function eval_model(
    p::Progress,
    epoch,
    Lconfig
    model,
    pars,
    train
    valid,
    test;
    device
)
    tm1 = @timed begin
        s_train, L_train = eval_model(Lconfig, model, pars, train; device)
        append!(p.loss_train, L_train)
    end
    tm2 = @timed begin
        s_valid, L_valid = eval_model(Lconfig, model, pars, valid; device)
        append!(p.loss_valid, L_valid)
    end
    tm3 = @timed begin
        s_test, L_test = eval_model(Lconfig, model, pars, test; device)
        append!(p.loss_test, L_test)
    end
    @info """
    Evaluation:
    ⋅ Epoch: $(epoch)
    ⋅ Train: $(durationstring(tm1.time))
    ⋅ Valid: $(durationstring(tm2.time))
    ⋅ Test: $(durationstring(tm3.time))
    """

    return Dict(
        :epoch => epoch,
        :model => deepcopy(cpu(model)),
        :train => Dict(:y => train[2], :s => cpu(s_train)),
        :valid => Dict(:y => valid[2], :s => cpu(s_valid)),
        :test => Dict(:y => test[2], :s => cpu(s_test)),
        :loss_batch => p.loss_batch,
        :loss_train => p.loss_train,
        :loss_valid => p.loss_valid,
        :loss_test => p.loss_test,
    )
end

save_model(datadir(dir, "checkpoints", "solution_epoch=$(epoch).bson"), solution)

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
    logger = generate_logger(datadir(dir))

    # run
    solution = []
    with_logger(logger) do
        @info """
        Initialization:
        ⋅ Dir: $(datadir(dir))
        ⋅ Loss config: $(string(Lconfig))
        ⋅ Model config: $(string(Mconfig))
        ⋅ Dataset config: $(string(Dconfig))
        ⋅ Optimiser config: $(string(Oconfig))
        ⋅ Train config: $(string(Tconfig))
        """
        save_config(
            datadir(dir, "config.yaml"),
            make_dict(Lconfig, Mconfig, Dconfig, Oconfig, Tconfig)
        )

        # initialization
        Random.seed!(Tconfig.seed)
        device = materialize(Tconfig)
        train, valid, test = load(Dconfig)
        batches = create_batches(Tconfig, train)
        model, pars = materialize(Mconfig; device)
        optimiser = materialize(Oconfig)
        p = Progress(;
            epoch_max = Tconfig.epochs*length(batches),
        )

        # initial state
        solution = eval_model(p, 0, Lconfig, model, pars, train, valid, test; device)
        save_model(
            datadir(dir, "checkpoints", "solution_epoch=0.bson"),
            solution,
        )

        # training loop
        progress!(p; training=false, force=true)
        reset_time!(p)
        @info "Training in progress..."
        for epoch in 1:Tconfig.epochs
            if Oconfig.decay_step != 1 && mod(epoch, Oconfig.decay_every) == 0
                optimiser.eta = max(
                    Float32(optimiser.eta * Oconfig.decay_step),
                    Float32(Oconfig.decay_min),
                )
            end

            # gradient step
            for batch in batches
                local L_batch
                grads = Flux.Zygote.gradient(pars) do
                    x, y = device(batch)
                    L_batch = loss(Lconfig, y, model(x), pars)
                    return L_batch
                end
                Flux.Optimise.update!(optimiser, pars, grads)
                append!(p.loss_batch, L_batch)
            end

            # checkpoint
            if mod(epoch, Tconfig.checkpoint_every) == 0 || epoch == Tconfig.epochs
                solution = eval_model(
                    p, epoch Lconfig, model, pars, train, valid, test;
                    device
                )
                save_model(
                    datadir(dir, "checkpoints", "solution_epoch=$(epoch).bson"),
                    solution,
                )
            end
            progress!(p)
        end
        @info "Saving final solution..."
        if !isempty(solution)
            save_model(datadir(dir, "solution.bson"), solution)
        end
        progress!(p; training=false, force=true)
    end
    return solution
end
