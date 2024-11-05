using DrWatson
quickactivate("ClassificationAtTopExperiments.jl")

using Evaluation

objectives = [
    # "CrossEntropy(0.5)",
    # "CrossEntropy(0.9)",
    # "CrossEntropy(0.99)",
    # "CrossEntropy(0.999)",
    "DeepTopPush",
    "PatMat-NP(1e-5)",
    "PatMat-NP(1e-4)",
    "PatMat-NP(1e-3)",
]

base_dir = "/home/machava2/projects/ClassificationAtTopExperiments.jl/data/steganography/results_tifs_2024-11-04"

for dataset_path in filter(isdir, readdir(base_dir, join=true))
    dataset = basename(dataset_path)

    for seed_path in filter(isdir, readdir(dataset_path, join=true))
        seed = basename(seed_path)

        plt1 = plot(
            xlims=(0, 1),
            ylims=(0, 1),
            legend=:bottomright,
            title=dataset
        )
        plt2 = plot(
            xaxis=:log,
            xlims=(1e-6, 1),
            ylims=(0, 1),
            # legend=false,
            legend=:bottomright,
            title=dataset,
        )

        for objective_path in filter(isfile, readdir(seed_path, join=true))
            if !endswith(objective_path, ".csv")
                continue
            end

            objective = replace(basename(objective_path), ".csv" => "")

            if !in(objective, objectives)
                continue
            end
            df = CSV.read(objective_path, DataFrame)

            plot!(plt1, df[!, "fpr_test"], df[!, "tpr_test"], label=objective, linewidth=2)
            plot!(plt2, df[!, "fpr_test"], df[!, "tpr_test"], label=objective, linewidth=2)
        end
        plt = plot(plt2, size=(600, 600))
        savefig("$(seed_path)/roc_curve.png")
    end
end