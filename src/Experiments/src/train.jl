function experiment_dir(
    dataset::DatasetType,
    model_type::ModelType,
    loss_type::LossType,
    opt_type::OptimiserType,
    train_config::TrainConfig;
)
    return datadir(
        train_config.save_dir,
        _string(dataset),
        _string(train_config),
        _string(opt_type),
        _string(model_type),
        _string(loss_type),
    )
end

function experiment_dir(
    dataset::DatasetType,
    model_type::ModelType,
    loss_type::LossType,
    train_config::TrainConfigDual;
)
    return datadir(
        train_config.save_dir,
        _string(dataset),
        _string(train_config),
        _string(model_type),
        _string(loss_type),
    )

end

function is_solved(path::AbstractString, force::Bool=false)
    args = load_config(path; update=false)
    dir = experiment_dir(args...)
    return isfile(solution_path(dir)) && !force
end

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
    eval_all = train_config.eval_all
    device = train_config.device == "GPU" ? Flux.gpu : Flux.cpu

    # Generate dir
    dir = experiment_dir(dataset, model_type, loss_type, opt_type, train_config)
    solution = nothing

    if isfile(solution_path(dir)) && !force
        @info "Loading existing solution"
        return load_checkpoint(solution_path(dir))
    end
    write_config(config_path(dir), dataset, model_type, loss_type, opt_type, train_config)
    reset_timer!(TO)

    # Run
    logger = generate_logger(dir)
    with_logger(logger) do
        @info """
        Initialization:
        ⋅ Dir: $(dir)
        ⋅ Dataset: $(_string(dataset))
        ⋅ Model: $(_string(model_type))
        ⋅ Loss: $(_string(loss_type))
        ⋅ Optimiser: $(_string(opt_type))
        """

        # Initialization
        @timeit TO "Initialization" begin
            Random.seed!(seed)
            @timeit TO "Data Loading" begin
                train, valid, test = load(dataset)
            end
            model = materialize(dataset, model_type) |> device
            pars = Flux.params(model)
            loss = materialize(loss_type)
        end

        # Batch loader
        batch_size = batch_neg + batch_pos
        if batch_size == 0
            loader = (getobs(train),)
        else
            buffer = isa(model_type, DeepTopPush) ? () -> Int[] : AccuracyAtTop.buffer_inds
            loader = BatchLoader(train; buffer, batch_neg, batch_pos)
        end
        iter_max = length(loader)

        # optimiser
        opt = materialize(opt_type, iter_max)

        # Progress logging
        p = Progress(; epoch_max, iter_max)

        # Initial state
        state = Dict{Symbol,Any}(
            :device => device,
            :epoch => 0,
            :dir => dir,
            :train_loader => EvalLoader(train, batch_size),
            :valid_loader => EvalLoader(valid, batch_size),
            :test_loader => EvalLoader(test, batch_size),
        )
        solution = checkpoint!(state, model, pars, loss; eval_all)

        if !any(isempty.((state[:loss_train], state[:loss_valid], state[:loss_test])))
            optionals = () -> (
                "Loss train" => state[:loss_train][end],
                "Loss valid" => state[:loss_valid][end],
                "Loss test" => state[:loss_test][end],
            )
        else
            optionals = () -> tuple()
        end

        # Training
        flag = false
        local training_loss
        start!(p, optionals()...)
        for epoch in 1:epoch_max
            state[:epoch] += 1
            @timeit TO "Epoch" begin
                for (iter, batch) in enumerate(loader)
                    @timeit TO "Loading batch" begin
                        x, y = batch
                        x = device(x)
                    end

                    # gradient step
                    @timeit TO "Gradient step" begin
                        grads = Flux.Zygote.gradient(pars) do
                            s = cpu(model(x))
                            training_loss = loss(y, s, pars)
                            return training_loss
                        end
                        Flux.Optimise.update!(opt, pars, grads)
                        progress!(p, iter, epoch, optionals()...)
                    end
                    free_memory!(x)

                    # break if corrupted solution
                    flag = isnan(training_loss) || isinf(training_loss)
                    flag && break
                end
            end

            # checkpoint
            if mod(epoch, checkpoint_every) == 0 || epoch == epoch_max || flag
                evl = epoch == epoch_max ? true : eval_all
                solution = checkpoint!(state, model, pars, loss; eval_all=evl)
            end
            @timeit TO "Garbage Collector" begin
                GC.gc(true)
            end
            flag && break
        end
        finish!(p, optionals()...)
        save_checkpoint(solution_path(dir), solution)
    end
    return solution
end

function checkpoint!(state, model, pars, loss; eval_all::Bool=true)
    solution = Dict(
        :model => deepcopy(cpu(model)),
        :epoch => state[:epoch],
    )

    loss_train = get!(state, :loss_train, Float32[])
    loss_valid = get!(state, :loss_valid, Float32[])
    loss_test = get!(state, :loss_test, Float32[])

    if eval_all
        @timeit TO "Evaluation" begin
            device = state[:device]
            train = state[:train_loader]
            valid = state[:valid_loader]
            test = state[:test_loader]
            @timeit TO "Train scores" y_train, s_train = eval_model(train, model, device)
            @timeit TO "Valid scores" y_valid, s_valid = eval_model(valid, model, device)
            @timeit TO "Test scores" y_test, s_test = eval_model(test, model, device)

            @timeit TO "Loss" begin
                append!(loss_train, loss(y_train, s_train, pars))
                append!(loss_valid, loss(y_valid, s_valid, pars))
                append!(loss_test, loss(y_test, s_test, pars))
            end

            solution[:train] = Dict(:y => cpu(y_train), :s => cpu(s_train))
            solution[:valid] = Dict(:y => cpu(y_valid), :s => cpu(s_valid))
            solution[:test] = Dict(:y => cpu(y_test), :s => cpu(s_test))
            solution[:loss_train] = loss_train
            solution[:loss_valid] = loss_valid
            solution[:loss_test] = loss_test
        end
    end
    @timeit TO "Saving" begin
        path = solution_path(state[:dir], state[:epoch])
        mkpath(dirname(path))
        save_checkpoint(path, solution)
    end
    return solution
end

function eval_model(
    loader::EvalLoader,
    model,
    device,
)

    if length(loader) == 1
        data, = iterate(loader)
        inds, (x, y) = data
        x = device(x)
        s = cpu(model(x))
        y = cpu(y)
        free_memory!(x)
        return y, s
    else
        n = numobs(loader)
        s = zeros(Float32, 1, n)
        y = similar(loader.data.targets, 1, n)
        k = length(loader)
        K = ceil(Int, k / 10)

        @info "Evaluation:"
        for (i, data) in enumerate(loader)
            if mod(i, K) == 0
                @info "Progress: $(i)/$(k)"
            end
            inds, (xi, yi) = data
            xi = device(xi)
            y[inds] .= yi[:]
            s[inds] .= cpu(model(xi))[:]
            free_memory!(xi)
        end
        return y, s
    end
end

# train for dual problems
function load_or_run(
    dataset::DatasetType,
    model_type::DualModelType,
    loss_type::LossType,
    train_config::TrainConfigDual;
)

    # Extract train config
    seed = train_config.seed
    force = train_config.force

    # Generate dir
    dir = experiment_dir(dataset, model_type, loss_type, train_config)
    solution = nothing

    if isfile(solution_path(dir)) && !force
        @info "Loading existing solution"
        return load_checkpoint(solution_path(dir))
    end
    write_config(config_path(dir), dataset, model_type, loss_type, train_config)
    reset_timer!(TO)

    # Run
    logger = generate_logger(dir)
    with_logger(logger) do
        @info """
        Initialization:
        ⋅ Dir: $(dir)
        ⋅ Dataset: $(_string(dataset))
        ⋅ Kernel: $(_string(model_type))
        ⋅ Loss: $(_string(loss_type))
        """

        # Initialization
        @timeit TO "Initialization" begin
            Random.seed!(seed)
            @timeit TO "Data Loading" begin
                train, valid, test = load(dataset, dual_shape)
                n_pos = sum(train.targets .== 1)
            end
            loss = materialize_dual(loss_type, n_pos)
            kernel = materialize_dual(model_type)
        end

        solution = ClassificationAtTopDual.solve(
            loss,
            kernel,
            getobs(train),
            getobs(valid),
            getobs(test);
            epoch_max=train_config.epoch_max,
            checkpoint_every=train_config.checkpoint_every,
            loss_every=train_config.loss_every,
            p_update=train_config.p_update,
            dir=dir,
            ε=train_config.ε
        )
        save_checkpoint(solution_path(dir), solution)
    end
    return solution
end
