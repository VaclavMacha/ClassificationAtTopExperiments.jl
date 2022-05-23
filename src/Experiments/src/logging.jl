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

# progress
@kwdef mutable struct Progress
    t_init::Float64 = time()
    t_last::Float64 = time()
    t_min::Float64 = 60
    epoch_max::Int = 0
    iter_max::Int = 0
end

function progress!(p::Progress, iter, epoch)
    time() - p.t_last >= p.t_min || return

    all_iter = p.epoch_max * p.iter_max
    finished_iter = iter + (epoch - 1) * p.iter_max

    # generate log message
    io = IOBuffer()
    perc = round(Int, 100 * finished_iter / all_iter)
    write(io, "Training in progress: $(perc)% \n")
    write(io, "⋅ Epoch: $(epoch)/$(p.epoch_max) \n")
    write(io, "⋅ Iteration: $(iter)/$(p.iter_max) \n")

    # duration
    p.t_last = time()
    elapsed = p.t_last - p.t_init
    per_epoch = elapsed / (epoch - 1)
    per_iter = elapsed / finished_iter
    eta = per_iter * (all_iter - finished_iter)

    write(io, "⋅ Elapsed time: $(durationstring(elapsed)) \n")
    write(io, "⋅ Time per epoch: $(speedstring(per_epoch)) \n")
    write(io, "⋅ Time per iter: $(speedstring(per_iter)) \n")
    write(io, "⋅ ETA: $(durationstring(eta)) \n")

    # print to logger
    @info String(take!(io))
    return
end

function finish!(p::Progress)
    p.t_last = time()
    elapsed = p.t_last - p.t_init

    # generate log message
    io = IOBuffer()
    write(io, "Training finished:  \n")
    write(io, "⋅ Elapsed time: $(durationstring(elapsed)) \n")

    # print to logger
    @info String(take!(io))
    return
end
