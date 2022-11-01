# ------------------------------------------------------------------------------------------
# critical diagrams
# ------------------------------------------------------------------------------------------
left_pad(n) = string(repeat(" ", n))

function critical_diagram(
    models,
    ranks,
    cv::Real;
    title::AbstractString="",
    l_pad::Int=2,
    v_pad::Real=0.4,
    v_pad_cv::Real=0.2,
    ymin::Real=0,
    full_range::Bool=true,
    digits::Int=2,
    tikz_only::Bool = false,
)

    io = IOBuffer()
    if !tikz_only
        write(
            io,
            """
            \\documentclass{standalone}
            \\usepackage{tikz}
            \\tikzstyle{line_node} = [line width=1pt, rounded corners, color=black, ->]
            \\tikzstyle{line_cv} = [line width=3pt, color=gray, line cap=round]

            \\begin{document}
            \\begin{tikzpicture}
            """
        )
    end

    # axis
    xmin = full_range ? 1 : floor(Int, minimum(ranks))
    xmax = full_range ? length(models) : ceil(Int, maximum(ranks))
    tmin = floor(ymin - 0.1; digits)
    tmax = ceil(ymin + 0.1; digits)
    n_models = length(ranks)
    n_left = ceil(Int, n_models / 2)

    # title
    if !isempty(title)
        x = round((xmin + xmax) / 2; digits=2)
        y = round(ymin + (n_left + 1) * v_pad; digits)
        write(io, left_pad(l_pad))
        write(io, "\\node at ($x,$y) {$title}; \n")
    end

    # x-axis
    write(io, left_pad(l_pad))
    write(io, "\\draw ($(xmin),$(ymin)) -- ($(xmax),$(ymin)); \n")

    # x ticks
    write(io, left_pad(l_pad))
    write(io, "\\foreach \\x in {$xmin,...,$xmax} \\draw (\\x,$(tmax)) -- (\\x,$(tmin)) node[anchor=north]{\$\\x\$}; \n")

    # nodes
    prm = sortperm(ranks)
    for (i, ind) in enumerate(prm)
        x_min = round(ranks[ind]; digits)
        y_min = ymin
        model = models[ind]
        if i <= n_left
            x_max = round(xmin - 0.1; digits)
            y_max = round(ymin + i * v_pad; digits)
            anchor = "east"
        else
            x_max = round(xmax + 0.1; digits)
            y_max = round(ymin + (n_models - i + 1) * v_pad; digits)
            anchor = "west"
        end
        write(io, left_pad(l_pad))
        write(io, "\\draw[line_node] ($x_min,$y_min) -- ($x_min,$y_max) -- ($x_max, $y_max) node[anchor=$anchor] {$(model)}; \n")
    end

    # levels
    coordinates = []
    for i in eachindex(prm)
        j = findlast(abs.(ranks[prm[(i+1):end]] .- ranks[prm[i]]) .<= cv)
        isnothing(j) && continue

        x_min = round(ranks[prm[i]]; digits)
        x_max = round(ranks[prm[j+i]]; digits)
        if isempty(coordinates) || coordinates[end][2] != x_max
            if isempty(coordinates)
                k = 1
            else
                kind = findfirst(getindex.(coordinates, 2) .< x_min)
                if isnothing(kind)
                    k = maximum(getindex.(coordinates, 3)) + 1
                else
                    k = coordinates[kind][3]
                end
            end
            y = round(ymin + k * v_pad_cv; digits)
            push!(coordinates, (x_min, x_max, k, y))
            write(io, left_pad(l_pad))
            write(io, "\\draw[line_cv] ($x_min,$y) -- ($x_max, $y); \n")
        end
    end

    # added end of document
    if !tikz_only
        write(
            io,
            """
            \\end{tikzpicture}
            \\end{document}
            """
        )
    end
    return String(take!(io))
end
