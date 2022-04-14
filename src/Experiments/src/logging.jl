Base.@kwdef mutable struct Progress
    t_init::Float64 = time()
    t_last::Float64 = time()
    t_min::Float64 = 60
    iter::Int = 0
    iter_max::Int = 0
    loss_train::Vector{Float32} = Float32[]
    loss_valid::Vector{Float32} = Float32[]
    loss_test::Vector{Float32} = Float32[]
end

function progress!(p::Progress; training::Bool=true, force = false)
    if training
        p.iter += 1
    end
    if time() - p.t_last >= p.t_min || force
        io = IOBuffer()
        write(io, "\n")
    
        log_progress!(io, p)
        log_duration!(io, p)
        log_loss!(io, p)
        @info String(take!(io))
    end
end

function reset_time!(p::Progress)
    p.t_init = time()
    p.t_last = time()
    return
end

function log_progress!(io, p::Progress)
    iter = p.iter
    iter_max = p.iter_max

    if iter == 0
        write(io, "Initialization: \n")
    elseif iter == iter_max
        write(io, "Training finished:  \n")
    else
        perc = round(Int, 100 * iter / iter_max)
        write(io, "Training in progress: $(iter)/$(iter_max) ($(perc))% \n")
    end
    return
end

function log_duration!(io, p::Progress)
    p.t_last = time()
    elapsed = p.t_last - p.t_init
    per_iter = elapsed / p.iter

    write(io, "⋅ Elapsed time: $(durationstring(elapsed)) \n")
    if 0 < p.iter
        write(io, "⋅ Time per iter: $(speedstring(per_iter)) \n")
    end
    if 0 < p.iter < p.iter_max
        eta = per_iter * (p.iter_max - p.iter)
        write(io, "⋅ ETA: $(durationstring(eta)) \n")
    end
    return
end

function log_loss!(io, p::Progress)
    isempty(p.loss_train) || write(io, "⋅ Loss train: $(p.loss_train[end]) \n")
    isempty(p.loss_valid) || write(io, "⋅ Loss valid: $(p.loss_valid[end]) \n")
    isempty(p.loss_test) || write(io, "⋅ Loss test: $(p.loss_test[end]) \n")
    return
end

function generate_logger(dir::AbstractString)
    fmt = "yyyy-mm-dd HH:MM:SS"
    return TeeLogger(
        EarlyFilteredLogger(
            log -> log.level == Logging.Info,
            TransformerLogger(FileLogger(joinpath(dir, "experiment.log"))) do log
                merge(log, (; message="$(Dates.format(now(), fmt))\n $(log.message)"))
            end,
        ),
        MinLevelLogger(FileLogger(joinpath(dir, "error.log")), Logging.Warn),
        global_logger()
    )
end