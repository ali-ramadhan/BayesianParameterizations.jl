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

# Approximate Bayesian computation

loss(y₁, y₂) = mean((y₂ .- y₁).^2) / length(y₁)

function rand_model(chain, θs)
    N = length(chain)
    n = rand(1:N)
    return NamedTuple(θ => chain[θ][n] for θ in θs)
end

function rand_y_linear(chain, x)
    m, c = rand_model(chain, (:m, :c))
    return @. m*x + c
end

function rand_y_quadratic(chain, x)
    a, b, c = rand_model(chain, [:a, :b, :c])
    return @. a*x^2 + b*x + c
end

function rand_y_exponential(chain, x)
    α, β, γ = rand_model(chain, [:α, :β, :γ])
    return @. α + β*exp(γ*x)
end

@enum LineModel linear=1 quadratic=2 exponential=3

function abc_line_model(x, y, M₀, chain_linear, chain_quadratic, chain_exponential; N_samples, max_loss)
    M = zeros(Int, N_samples)
    samples = 0
    trials = 0

    while samples < N_samples
        model = rand(M₀) |> LineModel

        if model == linear
            l = loss(y, rand_y_linear(chain_linear, x))
        elseif model == quadratic
            l = loss(y, rand_y_quadratic(chain_quadratic, x))
        elseif model == exponential
            l = loss(y, rand_y_exponential(chain_exponential, x))
        end

        if l <= max_loss
            samples += 1
            M[samples] = Int(model)
        end

        trials += 1
        @info @sprintf("Trial %d, sample %d: %s, loss=%.3e (%s)", trials, samples, model, l, l <= max_loss ? "accepted!" : "rejected")
    end

    @info *([@sprintf("P(m=%s)=%.4f%s", LineModel(m), count(M .== m) / N_samples, m < 3 ? ", " : "") for m in 1:3]...)

    return M
end

losses_linear = [loss(y, rand_y_linear(chain_linear, x)) for _ in 1:1000]
losses_quadratic = [loss(y, rand_y_quadratic(chain_quadratic, x)) for _ in 1:1000]
losses_exponential = [loss(y, rand_y_exponential(chain_exponential, x)) for _ in 1:1000]

M₀ = Categorical(3)  # Prior distribution over models?
M = abc_line_model(x, y, M₀, chain_linear, chain_quadratic, chain_exponential, N_samples=10000, max_loss=0.0034)
