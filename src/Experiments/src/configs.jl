function parse_config(T, dict, exclude::Symbol...)
    vals = [Symbol(key) => val for (key, val) in dict if !in(Symbol(key), exclude)]
    return T(; NamedTuple(vals)...)
end

function parse_config(dict)
    T = parse_type(dict["type"])
    return parse_config(T, dict, :type)
end

parse_type(type::String) = parse_type(Val(Symbol(type)))

function write_config(path, dict)
    open(path, "w") do io
        TOML.print(io, dict)
    end
end

function write_config(path, dataset, model_type, loss_type, opt_type, train_config)
    dict = Dict(
        "dataset" => _dict(dataset),
        "model" => _dict(model_type),
        "loss" => _dict(loss_type),
        "optimiser" => _dict(opt_type),
        "training" => _dict(train_config),
    )
    mkpath(dirname(path))
    write_config(path, dict)
end

function load_config(path; update::Bool=false)
    dict = TOML.parsefile(path)

    dataset = parse_config(dict["dataset"])
    model_type = parse_config(dict["model"])
    loss_type = parse_config(dict["loss"])
    opt_type = parse_config(dict["optimiser"])
    train_config = parse_config(dict["training"])

    if update
        write_config(path, dataset, model_type, loss_type, opt_type, train_config)
    end
    return dataset, model_type, loss_type, opt_type, train_config
end

# directory strings
function _string(x::T) where {T}
    exclude = _exclude(T)
    vals = [getproperty(x, key) for key in fieldnames(T) if !in(key, exclude)]
    return string(name(x), "(", join(string.(vals), ", "), ")")
end

name(::T) where {T} = string(T.name.name)
_exclude(::T) where {T} = tuple()
_exclude_toml(::T) where {T} = tuple()

# conversion to dicts
function _dict(x::T) where {T}
    exclude = _exclude_toml(T)
    ks = [key for key in fieldnames(T) if !in(key, exclude)]
    vals = [string(key) => toml_compat(getproperty(x, key)) for key in ks]
    return Dict("type" => name(x), vals...)
end

toml_compat(x) = x
toml_compat(x::Tuple) = [x...,]
