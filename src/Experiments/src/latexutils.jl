# ------------------------------------------------------------------------------------------
# critical diagrams
# ------------------------------------------------------------------------------------------
leftpad(s, n) = string(repeat(" ", n), s)

function critical_diagram(
    models,
    ranks,
    cv::Real;
    pad::Int=2,
    vpad::Real=0.4,
    v_offset::Real=0,
    cv_vpad::Real=0.2,
    cv_offset::Real=0.03,
    title::AbstractString="",
    full_range::Bool=true
)

    # header
    io = IOBuffer()
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

    # axis
    xmin = full_range ? 1 : floor(Int, minimum(ranks))
    xmax = full_range ? length(models) : ceil(Int, maximum(ranks))
    tmin = floor(v_offset - 0.1; digits=2)
    tmax = ceil(v_offset + 0.1; digits=2)
    write(io, leftpad("\\draw ($(xmin),$(v_offset)) -- ($(xmax),$(v_offset)); \n", pad)) # x axis
    write(io, leftpad("\\foreach \\x in {$xmin,...,$xmax} \\draw (\\x,$(tmax)) -- (\\x,$(tmin)) node[anchor=north]{\$\\x\$}; \n", pad)) # x ticks

    # nodes
    prm = sortperm(ranks)
    n = length(ranks)
    n_left = ceil(Int, n / 2)
    level = round(v_offset + cv_vpad; digits=2)
    last_cv = -1
    levels = []

    for (i, i_perm) in enumerate(prm)
        r = round(ranks[i_perm]; digits=2)
        m = models[i_perm]
        if i <= n_left
            h = round(v_offset + i * vpad; digits=2)
            write(io, leftpad("\\draw[line_node] ($(r),$(v_offset)) -- ($r,$h) -- ($(xmin-0.1), $h) node[anchor=east] {$(m)}; \n", pad))
        else
            h = round(v_offset + (n - i + 1) * vpad; digits=2)
            write(io, leftpad("\\draw[line_node] ($(r),$(v_offset)) -- ($r,$h) -- ($(xmax+0.1), $h) node[anchor=west] {$(m)}; \n", pad))
        end

        # levels
        lmin = round(r - cv_offset; digits=2)
        lmax = NaN
        for j_perm in prm[i+1:end]
            ranks[j_perm] - ranks[i_perm] <= cv || break
            lmax = round(ranks[j_perm] + cv_offset; digits=2)
        end
        if !isnan(lmax) && last_cv != lmax
            if last_cv < lmax
                last_cv = lmax
            end
            push!(levels, (lmin, lmax))
        end
    end

    # print levels
    hs = Dict{Int, Any}()
    for (lmin, lmax) in levels
        lvls = sort(collect(keys(hs)))
        if !isempty(hs)
            ind = maximum(lvls) + 1
            for j in lvls
                (kmin, kmax) = hs[j]
                if !any(kmin .<= (lmin, lmax) .<= kmax)
                    hs[j] = (lmin, lmax)
                    ind = j
                    break
                end
            end
        else
            ind = 1
            hs[1] = (lmin, lmax)
        end
        h = round(v_offset + ind * cv_vpad; digits=2)
        write(io, leftpad("\\draw[line_cv] ($lmin,$h) -- ($lmax, $h); \n", pad))
    end

    # title
    if !isempty(title)
        h = round(v_offset + (n_left + 1) * vpad; digits=2)
        c = round((xmin + xmax) / 2; digits=2)
        write(io, leftpad("\\node at ($(c),$(h)) {$title}; \n", pad))
    end

    # end of the document
    write(
        io,
        """
        \\end{tikzpicture}
        \\end{document}
        """
    )
    return String(take!(io))
end
