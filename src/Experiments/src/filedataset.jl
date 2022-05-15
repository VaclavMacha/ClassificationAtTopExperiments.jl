struct FileDataset{F, T<:AbstractString} <: AbstractDataContainer
    loadfn::F
    paths::Vector{T}
end

Base.getindex(dataset::FileDataset, i::Integer) = dataset.loadfn(dataset.paths[i])
Base.getindex(dataset::FileDataset, is::AbstractVector) = map(Base.Fix1(getobs, dataset), is)
Base.length(dataset::FileDataset) = length(dataset.paths)
