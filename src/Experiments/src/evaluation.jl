# ------------------------------------------------------------------------------------------
# DataFrame conversion
# ------------------------------------------------------------------------------------------
function _dataframe(x::T) where {T}
    ks = [key for key in fieldnames(T)]
    return [Symbol(key) => getproperty(x, key) for key in ks]
end

function _dataframe(x::T, type::String) where {T}
    return [Symbol(type) => T.name.name, _dataframe(x)...]
end

_dataframe(path::AbstractString) = _dataframe(TOML.parsefile(path))

function _dataframe(dict::AbstractDict)
    return DataFrame([
        _dataframe(parse_config(dict["dataset"]), "dataset")...,
        _dataframe(parse_config(dict["model"]), "model")...,
        _dataframe(parse_config(dict["loss"]), "loss")...,
        _dataframe(parse_config(dict["optimiser"]), "optimiser")...,
        _dataframe(parse_config(dict["training"]))...,
    ])
end

# ------------------------------------------------------------------------------------------
# Metrics and utils
# ------------------------------------------------------------------------------------------
function extract_scores(solution::Dict, split::Symbol)
    return (
        targets=vec(solution[split][:y]),
        scores=vec(solution[split][:s]),
    )
end

function pos_at_top_k(y, s, k::Int=1)
    t = partialsort(s[y.==0], k; rev=true)
    top = mean(s[y.==1] .> t)
    return top
end

function tpr_at_fpr(y, s, rate::Real)
    t = threshold_at_fpr(y, s, rate)
    return true_positive_rate(y, s, t)
end

function tpr_at_k(y, s, k::Int=1)
    t = partialsort(s[y.==0], k; rev=true)
    return true_positive_rate(y, s, t)
end

round_perc(val, digits=2) = round(100 * val; digits)

# ------------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------------
function evaluation(
    dir::AbstractString,
    id::Int,
    metrics::Pair...;
    split::Symbol=:test,
    epoch::Int=-1
)

    file_solution = solution_path(dir, epoch)
    file_config = config_path(dir)

    if !isfile(file_solution)
        @warn "Missing solution file: $(file_solution)"
        return nothing
    end
    if !isfile(file_config)
        @warn "Missing configuration file: $(file_config)"
        return nothing
    end
    df = _dataframe(file_config)
    insertcols!(df, 1, :id => id, :split => split)
    epoch = epoch < 0 ? df.epoch_max : epoch
    insertcols!(df, findfirst(==(:epoch_max), propertynames(df)), :epoch => epoch)
    y, s = extract_scores(load_checkpoint(file_solution), split)

    for (metric_name, metric_func) in metrics
        df[!, metric_name] .= metric_func(y, s)
    end
    return df
end

function evaluation(
    path::AbstractString;
    metrics::Vector{<:Pair}=Pair[],
    level::Int=4,
    id_start::Int=0,
    kwargs...
)

    if isdir(path)
        dirs = list_subdirs(path, level)
        isempty(dirs) && return nothing
        dfs = map(enumerate(dirs)) do (id, dir)
            return vcat(
                evaluation(dir, id_start + id, metrics...; split=:train, kwargs...),
                evaluation(dir, id_start + id, metrics...; split=:valid, kwargs...),
                evaluation(dir, id_start + id, metrics...; split=:test, kwargs...),
            )
        end
        filter!(df -> isa(df, AbstractDataFrame), dfs)
        df = vcat(dfs...; cols=:union)
    else
        df = CSV.read(path, DataFrame)
    end
    return df
end

function join_cols(
    df_in::AbstractDataFrame;
    metrics::Vector{<:Pair}=Pair[],
    to_join::Vector{<:Pair}=Pair[],
    include_cols::Vector{Symbol}=Symbol[],
    kwargs...
)

    # merge data frames
    df = copy(df_in)
    _join_cols!(df, to_join...)
    if isempty(include_cols)
        return df
    else
        include_cols = unique(vcat(include_cols, first.(metrics)))
        return select(df, include_cols)
    end
end

function _join_cols(cols...)
    return map(zip(cols...)) do vals
        join(string.(skipmissing(vals)), "-")
    end
end

function _join_cols!(df, tojoin::Pair...)
    for (col_new, cols) in tojoin
        cols = intersect(cols, propertynames(df))
        isempty(cols) && continue

        if col_new in setdiff(propertynames(df), cols)
            @warn "Skipping creation of column $(col_new) since the original DataFrame already contains a column with the same name."
            continue
        end

        ind = findfirst(in(cols), propertynames(df))
        new_column = select(df, cols => _join_cols)[:, 1]
        select!(df, Not(cols))
        insertcols!(df, ind, col_new => new_column)
    end
end

function _list_subdirs(dir::AbstractString)
    return filter(isdir, readdir(dir; join=true))
end

function _list_subdirs(dirs::Vector{String})
    return reduce(vcat, _list_subdirs.(dirs))
end

function list_subdirs(dirs, level::Int=4)
    return if level == 0
        _list_subdirs(dirs)
    else
        list_subdirs(_list_subdirs(dirs), level - 1)
    end
end

function _select_best(df, col::Symbol; by=argmax)
    inds = findall(df.split .== :valid)
    ind = by(df[inds, col])
    id = df.id[inds[ind]]
    return df[df.id.==id, :]
end

function select_best(
    df_in::DataFrame,
    metric::Symbol;
    split::Symbol=:test,
    wide::Bool=true,
    rank::Bool=false,
    rank_func::Function = x -> tiedrank(x; rev=true)
)

    df = select(df_in, [:id, :dataset, :loss, :split, metric])

    # select best parameters for loss
    df_best = combine(
        groupby(df, [:dataset, :loss]),
        sdf -> _select_best(sdf, metric)
    )
    df_best = df_best[df_best.split.==split, :]
    select!(df_best, Not([:split, :id]))

    # convert to ranks
    if rank
        df_best = transform(
            groupby(df_best, :dataset),
            :loss,
            metric => rank_func => metric,
        )
    end
    return wide ? DataFrames.unstack(df_best, :dataset, :loss, metric) : df_best
end

function rank_table(
    df::DataFrame,
    metrics::Vector{Symbol}=Symbol[];
    split::Symbol=:test,
    kwargs...
)
    dfs = []
    for (i, metric) in enumerate(metrics)
        df_best = select_best(df, metric; split, wide=true, rank=true, kwargs...)
        dropmissing!(df_best)

        tmp = DataFrame(Dict(
            :loss => names(df_best)[2:end],
            metric => vec(mean(Array(df_best[:, 2:end]); dims=1)),
        ))
        if i == 1
            insertcols!(
                tmp,
                2,
                :n_datasets => size(df_best, 1),
                :n_loss => size(df_best, 2) - 1,
            )
        end
        push!(dfs, tmp)
    end
    return innerjoin(dfs..., on=:loss)
end

# ------------------------------------------------------------------------------------------
# Critical diagrams
# ------------------------------------------------------------------------------------------
"""
    friedman_test_statistic(R::Vector, n::Int, k::Int)

Value of the Friedman test statistic.

# Arguments
- `R::Vector{<:Real}` vector of mean ranks
- `k::Int` number of models
- `n::Int` number of datasets
"""
function friedman_test_statistic(R::Vector, n::Int, k::Int)
    return 12 * n / (k * (k + 1)) * (sum(R .^ 2) - k * (k + 1)^2 / 4)
end

crit_chisq(α::Real, df::Int) = quantile(Chisq(df), 1 - α)

"""
    friedman_critval(k::Int; α::Real = 0.05)

Critical value of the Friedman test at level α.

# Arguments
- `k::Int` number of models
"""
friedman_critval(k::Int; α::Real=0.05) = crit_chisq(α / 2, k - 1)

function crit_srd(α::Real, k::Real, df::Real)
    if isnan(k) || isnan(df)
        NaN
    else
        quantile(StudentizedRange(df, k), 1 - α)
    end
end

"""
    nemenyi_cd(k::Int, n::Int; α::Real = 0.05)

Critical difference value for the Nemenyi paired test. If the difference between the average rank of two models is larger than this, their performance is different with a given statistical significance α.

# Arguments
- `k::Int` number of models
- `n::Int` number of datasets
"""
function nemenyi_cd(k::Int, n::Int; α::Real=0.05)
    return sqrt(k * (k + 1) / (6 * n)) * crit_srd(α, k, Inf) / sqrt(2)
end

# ------------------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------------------
function _load_solution(
    dir::AbstractString;
    epoch::Int=-1
)

    file_solution = solution_path(dir, epoch)
    file_config = config_path(dir)

    if !isfile(file_solution)
        @warn "Missing solution file: $(file_solution)"
        return nothing
    end
    if !isfile(file_config)
        @warn "Missing configuration file: $(file_config)"
        return nothing
    end
    _, _, loss, _, _ = load_config(file_config)
    return load_checkpoint(file_solution), loss
end

plot_roc(d::Dict, key; kwargs...) = plot_roc!(plot(), d, key; kwargs...)
plot_roc!(d::Dict, key; kwargs...) = plot_roc!(current(), d, key; kwargs...)

function plot_roc!(plt, d::Dict, key; kwargs...)
    y, s = extract_scores(d, key)
    t = sort(unique(s[y.==0]))
    return rocplot!(plt, y, s, t; kwargs...)
end

function plot_roc(
    maindir::String,
    exclude::LossType...;
    key::Symbol=:test,
    epoch::Int=-1,
    kwargs...
)

    plt = plot()
    for dir in readdir(maindir; join=true)
        isdir(dir) || continue
        out = _load_solution(dir; epoch)
        if !isnothing(out)
            sol, loss = out
            in(loss, exclude) && continue
            plot_roc!(plt, sol, key; label=_string(loss), kwargs...)
        end
    end
    return plt
end
