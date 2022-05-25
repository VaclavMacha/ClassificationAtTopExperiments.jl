load_checkpoint(path) = BSON.load(path, @__MODULE__)
save_checkpoint(path, model) = BSON.bson(path, model)

load_or_run(path) = load_or_run(load_config(path)...)

function load_or_run(
    dataset::DatasetType,
    model_type::ModelType,
    loss_type::LossType,
    opt_type::OptimiserType,
    train_config::TrainConfig;
)

    # Extract train config
    seed = train_config.seed
    force = train_config.force
    epoch_max = train_config.epoch_max
    checkpoint_every = train_config.checkpoint_every
    batch_pos = train_config.batch_pos
    batch_neg = train_config.batch_neg
    device = train_config.device == "GPU" ? Flux.gpu : Flux.cpu

    # Generate dir
    dir = datadir(
        "results",
        _string(dataset),
        _string(train_config),
        _string(opt_type),
        _string(model_type),
        _string(loss_type),
    )

    if isfile(joinpath(dir, "solution.bson")) && !force
        return load_checkpoint(joinpath(dir, "solution.bson"))
    end
    write_config(
        joinpath(dir, "config.toml"),
        dataset, model_type, loss_type, opt_type, train_config
    )

    # Run
    logger = generate_logger(dir)
    with_logger(logger) do
        @info """
        Initialization:
        ⋅ Dir: $(dir)
        ⋅ Dataset config: $(_string(dataset))
        ⋅ Model config: $(_string(model_type))
        ⋅ Loss config: $(_string(loss_type))
        ⋅ Optimiser config: $(_string(opt_type))
        """

        # Initialization
        @timeit TO "Initialization" begin
            mkpath(joinpath(dir, "checkpoints"))
            Random.seed!(seed)
            train, valid, test = load(dataset)
            model = materialize(dataset, model_type) |> device
            pars = Flux.params(model)
            loss = materialize(loss_type)
            opt = materialize(opt_type)
        end

        # Batch loader
        batch_size = batch_neg + batch_pos
        if batch_size == 0
            batch = Batch(device)
            loader = () -> train
            iter_max = 1
        else
            batch = Batch(device, obs_size(dataset)..., batch_size)
            loader = BatchLoader(
                train;
                buffer=isa(model_type, DeepTopPush) ? () -> Int[] : AccuracyAtTop.buffer_inds,
                batch_neg,
                batch_pos
            )
            iter_max = ceil(Int, length(train) / batch_size)
        end

        # Initial state
        @timeit TO "Evaluation" begin
            history = Dict{Symbol,Any}()
            solution = checkpoint!(
                history,
                batch,
                train,
                valid,
                test,
                model,
                pars,
                loss,
                0,
                joinpath(dir, "checkpoints", "checkpoint_epoch=0.bson"),
            )
        end
        @info """
        TimerOutputs:
        $(TO)
        """

        # Progress logging
        @info "Start training..."
        p = Progress(; epoch_max, iter_max)

        # Training
        for epoch in 1:epoch_max
            for iter in 1:iter_max
                # loading batch
                @timeit TO "Data metarialization" begin
                    x, y = get_batch!(batch, loader())
                end

                # gradient step
                @timeit TO "Gradient step" begin
                    grads = Flux.Zygote.gradient(pars) do
                        loss(y, model(x), pars)
                    end
                    Flux.Optimise.update!(opt, pars, grads)
                    progress!(p, iter, epoch)
                end
            end

            # checkpoint
            if mod(epoch, checkpoint_every) == 0 || epoch == epoch_max
                @timeit TO "Evaluation" begin
                    solution = checkpoint!(
                        history,
                        batch,
                        train,
                        valid,
                        test,
                        model,
                        pars,
                        loss,
                        epoch,
                        joinpath(dir, "checkpoints", "checkpoint_epoch=$(epoch).bson"),
                    )
                end
            end
            @info """
            TimerOutputs:
            $(TO)
            """
        end
        finish!(p)
        save_checkpoint(joinpath(dir, "solution.bson"), solution)
    end
    return solution
end

function checkpoint!(history, batch, train, valid, test, model, pars, loss, epoch, path)
    tm1 = @timed y_train, s_train = eval_model!(batch, train, model)
    @info "Train data evaluation: $(durationstring(tm1.time))"

    tm2 = @timed y_valid, s_valid = eval_model!(batch, valid, model)
    @info "Valid data evaluation: $(durationstring(tm2.time))"

    tm3 = @timed y_test, s_test = eval_model!(batch, test, model)
    @info "Test data evaluation: $(durationstring(tm3.time))"

    loss_train = get!(history, :loss_train, Float32[])
    loss_valid = get!(history, :loss_valid, Float32[])
    loss_test = get!(history, :loss_test, Float32[])

    append!(loss_train, loss(y_train, s_train, pars))
    append!(loss_valid, loss(y_train, s_train, pars))
    append!(loss_test, loss(y_train, s_train, pars))

    solution = Dict(
        :model => deepcopy(cpu(model)),
        :epoch => epoch,
        :train => Dict(:y => cpu(y_train), :s => cpu(s_train)),
        :valid => Dict(:y => cpu(y_valid), :s => cpu(s_valid)),
        :test => Dict(:y => cpu(y_test), :s => cpu(s_test)),
        :loss_train => loss_train,
        :loss_valid => loss_valid,
        :loss_test => loss_test,
    )
    save_checkpoint(path, solution)
    return solution
end
