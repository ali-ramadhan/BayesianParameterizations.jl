using Statistics
using Printf

using OffsetArrays
using DifferentialEquations
using Turing
using DynamicHMC
using CairoMakie

# Turing.setadbackend(:forwarddiff)

# Exact solution

## Convert x to periodic range in [0, L].
@inline x′(x, t, L, U) = mod(x + L - U * t, L)
@inline ϕ_exact(x, t, L, U) = exp(-100 * (x′(x, t, L, U) - 0.5)^2)

# Numerical solution

@inline advective_flux(i, u, ϕ) = u * (ϕ[i-1] + ϕ[i]) / 2

@inline ∂x_advective_flux(i, Δx, u, ϕ) = (advective_flux(i+1, u, ϕ) - advective_flux(i, u, ϕ)) / Δx

function advection(∂ϕ∂t, ϕ, p, t)
    Nx, Hx, Δx, u = p
    Nx = Int(Nx)
    Hx = Int(Hx)

    # filling periodic halos
    ϕ[-Hx+1:0] .= ϕ[Nx]
    ϕ[Nx+1:Nx+Hx] .= ϕ[1]

    for i in 1:Nx
        ∂ϕ∂t[i] = - ∂x_advective_flux(i, Δx, u, ϕ)
    end
end

Nx = 64
Hx = 1
Lx = 1
Δx = Lx / Nx
u = 0.2
p = [Nx, Hx, Δx, u]

xC = range(Δx/2, Lx - Δx/2, length=Nx)
xF = range(0, Lx, length=Nx+1)

ϕ₀ = OffsetArray(zeros(Nx+2Hx), -Hx+1:Nx+Hx)
@. ϕ₀[1:Nx] = ϕ_exact.(xC, 0, Lx, u)

tspan = (0.0, 10.0)
prob = ODEProblem(advection, ϕ₀, tspan, p, saveat=0.1)
sol = solve(prob, Tsit5())

function animate_advection(ϕ, ϕ_correct, x, t; framerate=30)
    n = Observable(1)
    ϕₙ = @lift ϕ[:, $n]
    ϕₙ_correct = @lift ϕ_correct[:, $n]

    title = @lift "Advection: t = $(round(t[$n], digits=1))"

    fig = Figure()
    ax = Axis(fig[1, 1], title=title, xlabel="x", ylabel="ϕ")
    lines!(ax, x, ϕₙ_correct)
    lines!(ax, x, ϕₙ)

    time_indices = 1:length(t)
    record(fig, "advection_1d.mp4", time_indices; framerate) do time_index
        @info "Animating 1D advection: frame $time_index/$(length(t))"
        n[] = time_index
    end
end

ϕ = Array(sol)[Hx+1:Nx+Hx, :]
ϕ_correct = ϕ_exact.(xC, sol.t', Lx, u)
# ϕ_correct = @. ϕ_correct + 1e-2 * randn()
animate_advection(ϕ, ϕ_correct, xC, sol.t)

function predict_advection(prob, u)
    Nx, Hx, Δx, _ = prob.p
    Nx = Int(Nx)
    Hx = Int(Hx)

    p = [Nx, Hx, Δx, u]
    prob = remake(prob, p=p)
    prediction = solve(prob, Tsit5())

    return prediction
end

@model function probabilistic_advection(data, prob)
    σ ~ truncated(Exponential(0.05), 0, 0.1)
    u ~ truncated(Normal(0.2, 0.05), 0, 0.5)

    prediction = predict_advection(prob, u)

    for i = 1:length(prediction)
        data[:, i] ~ MvNormal(prediction[i][1:Nx], σ)
    end
end

# prior_chain  = sample(probabilistic_advection(ϕ_correct, prob), Prior(), 1000)

function callback(rng, model, sampler, sample, state, iter; loss_history, kwargs...)
    u = sample.θ.u[1][1]
    σ = sample.θ.σ[1][1]

    prediction = Array(predict_advection(prob, u))[1+Hx:Nx+Hx, :]
    loss_history[iter] = loss = mean((prediction .- ϕ_correct).^2)

    @info "Iteration $iter: u=$u, σ=$σ, ℒ=$loss"
end

stat_model = probabilistic_advection(ϕ_correct, prob)

# sampler = MH(:u => x -> Normal(x, 0.05), :σ => x -> Normal(x, 5e-3))
sampler = HMC(1e-3, 512)
# sampler = NUTS(0.65, max_depth=10)
# sampler = DynamicNUTS()
# sampler = HMCDA(10, 0.65, 0.3)

θ₀ = [0.01, 0.2] # Use with the init_params=θ₀ kwarg.

n_samples = 10000
loss_history = zeros(n_samples)
# chain = sample(stat_model, sampler, n_samples, progress=true; callback, loss_history)
# chain = sample(stat_model, sampler, n_samples, progress=true, init_params=θ₀; callback, loss_history)

function callback(rng, model, sampler, sample, state, iter; loss_history, kwargs...)
    n = Threads.threadid()

    u = sample.θ.u[1][1]
    σ = sample.θ.σ[1][1]

    prediction = Array(predict_advection(prob, u))[1+Hx:Nx+Hx, :]
    loss_history[n, iter] = loss = mean((prediction .- ϕ_correct).^2)

    @info "Chain $n, Iteration $iter: u=$u, σ=$σ, ℒ=$loss"
end

n_chains = 8
n_samples = 1000
loss_history = zeros(n_chains, n_samples)
chains = sample(stat_model, sampler, MCMCThreads(), n_samples, n_chains, progress=true; callback, loss_history)

# q = vi(stat_model, ADVI(100, 1000))

# N = 1001
# losses = zeros(N)
# us = range(-1, 1, length=N)
# for (i, ũ) in enumerate(us)
#     prediction = Array(predict_advection(prob, ũ))[1+Hx:Nx+Hx, :]
#     losses[i] = mean((prediction .- ϕ_correct).^2)
# end
# lines(us, maximum(losses) .- losses, axis=(xlabel="u", ylabel="likelihood ~ -loss function"))
