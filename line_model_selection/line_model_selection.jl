using Printf
using Turing
using CairoMakie
using ElectronDisplay

# Data generation

N = 100
x = range(0, 1, length=N)
y = @. 0.16x^2 + 4.5x + π
y = @. y * (1 + 0.1 * randn())

scatter(x, y)

# Linear model (y = mx + c)

@model function linear_model(x, y)
    m ~ Normal(0, 1)
    c ~ Normal(0, 1)
    σ ~ Exponential(1)

    N = length(x)
    for n in 1:N
        y[n] ~ Normal(m*x[n] + c, σ)
    end
end

chain_linear = sample(linear_model(x, y), NUTS(0.65), 10000, progress=true)

function y_linear(chain, n, x)
    m = chain[:m][n]
    c = chain[:c][n]
    return @. m * x + c
end

function animate_chain(chain, y_model, x, y; filepath, N_samples=1000, N_history=50, max_alpha=0.5)
    n = Observable(1)

    # Compute all predictions to plot.
    yₚ = []
    for i in 1:N_samples
        push!(yₚ, y_model(chain, i, x))
    end

    fig = Figure()

    title = @lift "Iteration $($n)"
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y"; title)

    # Create N_history lines with different alpha that we can change the data on.
    line_plots = []
    for i in 1:N_history
        line_plot = lines!(x, yₚ[1], color=(:gray, max_alpha * i/N_history))
        push!(line_plots, line_plot)
    end

    # Plot the actual data points.
    scatter!(x, y)

    record(fig, filepath, 1:N_samples, framerate=30) do chain_iter
        @info "Animating $filepath: frame $chain_iter/$N_samples"
        n[] = chain_iter

        # Change the data of the line plots to show the last N_history predictions.
        for (i, j) in enumerate(chain_iter:-1:max(1, chain_iter-N_history+1))
            line_plots[i][2] = yₚ[j]
        end
    end

    return
end

animate_chain(chain_linear, y_linear, x, y, filepath="linear_model.mp4")

function plot_chains_and_densities(chain, vars; filepath)
    fig = Figure(resolution=(800, 200*length(vars)))

    for (n, var) in enumerate(vars)
        ax1 = Axis(fig[n, 1], ylabel="$var", xlabel="iteration")
        lines!(ax1, chain[var][:] |> skipmissing |> collect)

        ax2 = Axis(fig[n, 2], ylabel="P($var)", xlabel="$var", yticklabelsvisible=false)
        density!(ax2, chain[var][:] |> skipmissing |> collect)
    end

    save(filepath, fig, px_per_unit=2)

    return
end

plot_chains_and_densities(chain_linear, [:m, :c, :σ], filepath="linear_model_posteriors.png")

# Quadratic model (y = ax² + bx + c)

@model function quadratic_model(x, y)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    c ~ Normal(0, 1)
    σ ~ Exponential(1)

    N = length(x)
    for n in 1:N
        y[n] ~ Normal(a*x[n]^2 + b*x[n] + c, σ)
    end
end

chain_quadratic = sample(quadratic_model(x, y), NUTS(0.65), 10000, progress=true)

function y_quadratic(chain, n, x)
    a = chain[:a][n]
    b = chain[:b][n]
    c = chain[:c][n]
    return @. a*x^2 + b*x + c
end

animate_chain(chain_quadratic, y_quadratic, x, y, filepath="quadratic_model.mp4")

plot_chains_and_densities(chain_quadratic, [:a, :b, :c, :σ], filepath="quadratic_model_posteriors.png")

# Exponential model [y = α + β * exp(γ * x)]

@model function exponential_model(x, y)
    α ~ Normal(0, 1)
    β ~ Normal(0, 1)
    γ ~ Normal(0, 1)
    σ ~ Exponential(1)

    N = length(x)
    for n in 1:N
        y[n] ~ Normal(α + β * exp(γ*x[n]), σ)
    end
end

chain_exponential = sample(exponential_model(x, y), NUTS(0.65), 10000, progress=true)

function y_exponential(chain, n, x)
    α = chain[:α][n]
    β = chain[:β][n]
    γ = chain[:γ][n]
    return @. α + β * exp(γ * x)
end

animate_chain(chain_exponential, y_exponential, x, y, filepath="exponential_model.mp4")

plot_chains_and_densities(chain_exponential, [:α, :β, :γ, :σ], filepath="exponential_model_posteriors.png")

# Categorical model (picks between linear, quadratic, and exponential)

@model function categorical_model1(x, y)
    N = length(x)

    pl ~ Beta(1, 1)
    M ~ Categorical([pl, 1-pl])

    if M == 1
        m ~ Normal(0, 1)
        k ~ Normal(0, 1)
        σl ~ Exponential(1)

        for n in 1:N
            y[n] ~ Normal(m*x[n] + k, σl)
        end
    else
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        σq ~ Exponential(1)

        for n in 1:N
            y[n] ~ Normal(a*x[n]^2 + b*x[n] + c, σq)
        end
    end
end

prior_chain_categorical1 = sample(categorical_model1(x, y), Prior(), 10000)
plot_chains_and_densities(prior_chain_categorical1, [:pl, :M, :a, :b, :c, :m, :k], filepath="categorical_model1_priors.png")

# Errors since variables defined in if statements are missing during inference?
sampler = Gibbs(PG(100, :M), NUTS(100, 0.65, :pl, :m, :k, :σl, :a, :b, :c, :σq))
chain_categorical1 = sample(categorical_model1(x, y), sampler, 1000, progress=true)

@model function categorical_model2(x, y)
    N = length(x)

    pl ~ Beta(1, 1)
    M ~ Categorical([pl, 1-pl])

    m ~ Normal(0, 1)
    k ~ Normal(0, 1)
    σl ~ Exponential(1)

    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    c ~ Normal(0, 1)
    σq ~ Exponential(1)

    if M == 1
        for n in 1:N
            y[n] ~ Normal(m*x[n] + k, σl)
        end
    else
        for n in 1:N
            y[n] ~ Normal(a*x[n]^2 + b*x[n] + c, σq)
        end
    end
end

prior_chain_categorical2 = sample(categorical_model2(x, y), Prior(), 10000)
plot_chains_and_densities(prior_chain_categorical2, [:pl, :M, :a, :b, :c, :m, :k], filepath="categorical_model2_priors.png")

# Inference fails? since it always picks M=1 or M=2 and doesn't sample from both models.
sampler = Gibbs(PG(10, :M), NUTS(100, 0.65, :pl, :m, :k, :σl, :a, :b, :c, :σq))
chain_categorical2 = sample(categorical_model2(x, y), sampler, 1000, progress=true)
plot_chains_and_densities(chain_categorical2, [:pl, :M, :m, :k, :σl, :a, :b, :c, :σq], filepath="categorical_model2_posteriors.png")

@model function categorical_model3(x, y)
    N = length(x)

    m ~ Normal(0, 1)
    k ~ Normal(0, 1)
    σl ~ Exponential(1)

    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    c ~ Normal(0, 1)
    σq ~ Exponential(1)

    pl ~ Beta(1, 1)

    if rand() < pl
        for n in 1:N
            y[n] ~ Normal(m*x[n] + k, σl)
        end
    else
        for n in 1:N
            y[n] ~ Normal(a*x[n]^2 + b*x[n] + c, σq)
        end
    end
end

# Inference is bad and chain is very correlated.
chain_categorical3 = sample(categorical_model3(x, y), NUTS(), 1000, progress=true)
plot_chains_and_densities(chain_categorical3, [:pl, :m, :k, :σl, :a, :b, :c, :σq], filepath="categorical_model3_posteriors.png")

function plot_model_comparison(chain, x, y)
    fig = Figure()
    ax = Axis(fig[1, 1])

    pl = mode(chain[:pl])

    m = mode(chain[:m])
    k = mode(chain[:k])

    a = mode(chain[:a])
    b = mode(chain[:b])
    c = mode(chain[:c])

    y_linear = @. m*x + k
    y_quadratic = @. a*x^2 + b*x + c

    lines!(ax, x, y_linear, label=@sprintf("linear (%d%%)", 100pl))
    lines!(ax, x, y_quadratic, label=@sprintf("quadratic (%d%%)", 100*(1-pl)))

    scatter!(x, y)
    axislegend(ax)

    return fig
end

plot_model_comparison(chain_categorical3, x, y)

chain_time(chain) = chain.info.stop_time - chain.info.start_time

chains = (chain_linear, chain_quadratic, chain_exponential, chain_categorical2, chain_categorical3)
chain_names = ("linear", "quadratic", "exponential", "categorical1", "categorical2")

open("inference_wall_clock_times.txt","w") do io
    for (chain, name) in zip(chains, chain_names)
        line = @sprintf("%s: %.1f seconds for %d samples", name, chain_time(chain), size(chain, 1))
        @info line
        println(io, line)
    end
end
