using Turing
using CairoMakie
using ElectronDisplay

N = 100
x = range(0, 1, length=N)
y = @. 0.16x^2 + 4.5x + π
y = @. y * (1 + 0.1 * randn())

scatter(x, y)

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

begin
    N = 1000
    n = Observable(1)

    yₚ = []
    for i in 1:N
        m = chain_linear[:m][i]
        c = chain_linear[:c][i]
        push!(yₚ, @. m * x + c)
    end

    fig = Figure()

    title = @lift "Iteration $($n)"
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y"; title)

    n_history = 50
    line_plots = []
    for i in 1:n_history
        line_plot = lines!(x, yₚ[1], color=(:gray, 0.5 * i/n_history))
        push!(line_plots, line_plot)
    end

    scatter!(x, y)

    chain_iters = 1:N
    record(fig, "linear_model.mp4", chain_iters, framerate=30) do chain_iter
        @info "Animating linear model: frame $chain_iter/$N"
        n[] = chain_iter

        for (i, j) in enumerate(chain_iter:-1:max(1, chain_iter-n_history+1))
            line_plots[i][2] = yₚ[j]
        end
    end
end

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

begin
    fig = Figure()
    ax = Axis(fig[1, 1])
    for _ in 1:100
        a = rand(chain_quadratic[:a])
        b = rand(chain_quadratic[:b])
        c = rand(chain_quadratic[:c])
        y′ = @. a*x^2 + b*x + c
        lines!(x, y′, color=(:gray, 0.25))
    end
    scatter!(x, y)
    fig
end

@model function exponential_model(x, y)
    α ~ Normal(0, 1)
    β ~ Normal(0, 1)
    γ ~ Normal(0, 1)
    δ ~ Normal(0, 1)
    σ ~ Exponential(1)

    N = length(x)
    for n in 1:N
        y[n] ~ Normal(α + β * exp(γ*x[n] + δ), σ)
    end
end

chain_exponential = sample(exponential_model(x, y), NUTS(0.65), 10000, progress=true)

begin
    fig = Figure()
    ax = Axis(fig[1, 1])
    for _ in 1:100
        α = rand(chain_exponential[:α])
        β = rand(chain_exponential[:β])
        γ = rand(chain_exponential[:γ])
        δ = rand(chain_exponential[:δ])
        y′ = @. α + β * exp(γ * x + δ)
        lines!(x, y′, color=(:gray, 0.25))
    end
    scatter!(x, y)
    fig
end

@model function categorical_model(x, y)
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

prior_chain_categorical  = sample(categorical_model(x, y), Prior(), 1000)

function callback(rng, model, sampler, sample, state, iter; kwargs...)
    M = sample.θ.M[1][1]
    @info "Iteration $iter: M=$M"
end

# sampler = Gibbs(PG(100, :M), NUTS(100, 0.65, :pl, :m, :k, :σl, :a, :b, :c, :σq))
chain_categorical = sample(categorical_model(x, y), NUTS(max_depth=15), 10000, progress=true)

begin
    fig = Figure()
    ax = Axis(fig[1, 1])

    pl = mode(chain_categorical[:pl])

    m = mode(chain_categorical[:m])
    k = mode(chain_categorical[:k])

    a = mode(chain_categorical[:a])
    b = mode(chain_categorical[:b])
    c = mode(chain_categorical[:c])

    y_linear = @. m*x + k
    y_quadratic = @. a*x^2 + b*x + c

    lines!(x, y_linear, label=@sprintf("linear (%d%%)", 100pl))
    lines!(x, y_quadratic, label=@sprintf("quadratic (%d%%)", 100*(1-pl)))

    scatter!(x, y)
    fig
end
