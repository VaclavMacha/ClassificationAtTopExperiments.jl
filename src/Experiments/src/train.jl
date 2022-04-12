@option "training" struct Training
    seed::Int = 1234
    force::Bool = false
    device::Function = cpu
    iters::Int = 100
    checkpoint_every::Int = 5
end

# Save directory
function dir_name(
    loss_config::O,
    model_config::Model,
    data_config::Dataset,
    opt_config::Optimiser,
    config::Training,
) where {O<:Objective}

    return joinpath(
        string.([config, data_config, model_config, loss_config, opt_config])...,
    )
end

# saving and loading model
load_model(path) = BSON.load(path, @__MODULE__)
save_model(path, model) = BSON.bson(path, model)

# experiment
function run_experiments(
    loss_config::O,
    model_config::Model,
    data_config::Dataset,
    opt_config::Optimiser,
    config::Training,
) where {O<:Objective}

    # check if exists
    dir = dir_name(loss_config, model_config, data_config, opt_config, config)
    if !config.force && isfile(datadir(dir, "solution.bson"))
        return load_model(datadir(dir, "solution.bson"))
    end
    mkpath(datadir(dir))
    mkpath(datadir(dir, "checkpoints"))

    # initialization
    @info "Experiment in progress..."
    train, test = load(data_config) |> config.device
    model, pars = materialize(model_config; device=config.device)
    optimiser = materialize(opt_config)

    solution = []
    loss_train = Float32[]
    loss_test = Float32[]

    # training loop
    @info "Training in progress..."
    for iter in 1:config.iters
        if opt_config.decay_step != 1 && mod(iter, opt_config.decay_every) == 0
            optimiser.eta = max(optimiser.eta * opt_config.decay_step, opt_config.decay_min)
        end

        # gradient step
        local L_train
        local s_train
        grads = Flux.Zygote.gradient(pars) do
            s_train = model(train[1])
            L_train = loss(loss_config, train[2], s_train, pars)
            return L_train
        end
        Flux.Optimise.update!(optimiser, pars, grads)
        append!(loss_train, L_train)

        # checkpoint
        if mod(iter, config.checkpoint_every) == 0 || iter == config.iters
            s_test = model(test[1])
            append!(loss_test, loss(loss_config, test[2], s_test, pars))

            solution = Dict(
                :iter => iter,
                :model => deepcopy(cpu(model)),
                :train => Dict(:y => train[2], :s => cpu(s_train)),
                :test => Dict(:y => test[2], :s => cpu(s_test)),
                :loss_train => loss_train,
                :loss_test => loss_test,
            )
            save_model(datadir(dir, "checkpoints", "solution_iter=$(iter).bson"), solution)
        end
    end
    if !isempty(solution)
        save_model(datadir(dir, "solution.bson"), solution)
    end
    return solution
end