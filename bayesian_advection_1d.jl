using OffsetArrays
using DifferentialEquations
using Turing
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
animate_advection(ϕ, ϕ_correct, xC, sol.t)

@model function probabilistic_advection(data, prob)
    Nx, Hx, Δx, _ = prob.p
    Nx = Int(Nx)
    Hx = Int(Hx)

    σ ~ Uniform(0, 0.01)
    u ~ Uniform(0.18, 0.22)

    p = [Nx, Hx, Δx, u]
    prob = remake(prob, p=p)
    prediction = solve(prob, Tsit5())

    for i = 1:length(prediction)
        data[:, i] ~ MvNormal(prediction[i][1:Nx], σ)
    end
end

stat_model = probabilistic_advection(ϕ_correct, prob)

function callback(rng, model, sampler, sample, state, iter; kwargs...)
    @show iter
end

ϵ, τ = 1e-2, 100
n_samples = 100
chain = sample(stat_model, NUTS(0.65), n_samples, progress=true)

# chain = sample(stat_model, HMC(0.01, 10), MCMCThreads(), 1000, 3; callback)
