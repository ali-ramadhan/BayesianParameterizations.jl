using Statistics
using Printf

using OffsetArrays
using DifferentialEquations
using Turing
using CairoMakie
using ElectronDisplay

# Numerical solution of the 2D advection-diffusion equation

@inline advective_flux_x(i, j, u, ϕ) = u * (ϕ[i-1, j] + ϕ[i, j]) / 2
@inline advective_flux_y(i, j, v, ϕ) = v * (ϕ[i, j-1] + ϕ[i, j]) / 2

@inline ∂x_advective_flux(i, j, Δx, u, ϕ) = (advective_flux_x(i+1, j, u, ϕ) - advective_flux_x(i, j, u, ϕ)) / Δx
@inline ∂y_advective_flux(i, j, Δy, v, ϕ) = (advective_flux_y(i, j+1, v, ϕ) - advective_flux_y(i, j, v, ϕ)) / Δy

@inline div_advective_flux(i, j, Δx, Δy, u, v, ϕ) = ∂x_advective_flux(i, j, Δx, u, ϕ) + ∂y_advective_flux(i, j, Δy, v, ϕ)

@inline ∂xᶠᵃ(i, j, Δx, ϕ) = (ϕ[i, j] - ϕ[i-1, j]) / Δx
@inline ∂yᵃᶠ(i, j, Δy, ϕ) = (ϕ[i, j] - ϕ[i, j-1]) / Δy

@inline diffusive_flux_x(i, j, Δx, κˣ, ϕ) = κˣ * ∂xᶠᵃ(i, j, Δx, ϕ)
@inline diffusive_flux_y(i, j, Δy, κʸ, ϕ) = κʸ * ∂yᵃᶠ(i, j, Δy, ϕ)

@inline ∂x_diffusive_flux(i, j, Δx, κˣ, ϕ) = (diffusive_flux_x(i+1, j, Δx, κˣ, ϕ) - diffusive_flux_x(i, j, Δx, κˣ, ϕ)) / Δx
@inline ∂y_diffusive_flux(i, j, Δy, κʸ, ϕ) = (diffusive_flux_y(i, j+1, Δy, κʸ, ϕ) - diffusive_flux_y(i, j, Δy, κʸ, ϕ)) / Δy

@inline div_diffusive_flux(i, j, Δx, Δy, κˣ, κʸ, ϕ) = ∂x_diffusive_flux(i, j, Δx, κˣ, ϕ) + ∂y_diffusive_flux(i, j, Δy, κʸ, ϕ)

function advection_diffusion_2d(∂ϕ∂t, ϕ, p, t)
    Nx, Ny, Hx, Hy, Δx, Δy, u, v, κˣ, κʸ = p
    Nx, Ny, Hx, Hy = Int.((Nx, Ny, Hx, Hy))

    # filling periodic halos
    ϕ[-Hx+1:0, :] .= ϕ[Nx:Nx, :]
    ϕ[:, -Hy+1:0] .= ϕ[:, Ny:Ny]

    ϕ[Nx+1:Nx+Hx, :] .= ϕ[1:1, :]
    ϕ[:, Ny+1:Ny+Hy] .= ϕ[:, 1:1]

    for i in 1:Nx, j in 1:Ny
        ∂ϕ∂t[i, j] = - div_advective_flux(i, j, Δx, Δy, u, v, ϕ) + div_diffusive_flux(i, j, Δx, Δy, κˣ, κʸ, ϕ)
    end

    return
end

# Initial condition

ϕ_initial_condition(x, y) = exp(-100 * (x - 0.5)^2) * exp(-100 * (y - 0.5)^2)

# Generating a high-resolution solution and animating it

Nx = Ny = 256
Hx = Hy = 1

Lx = Ly = 1
Δx = Lx / Nx
Δy = Ly / Ny

u = 0.25
v = 0.15

κˣ = 1e-3
κʸ = 1e-3

p = [Nx, Ny, Hx, Hy, Δx, Δy, u, v, κˣ, κʸ]

xC = range(Δx/2, Lx - Δx/2, length=Nx)
xF = range(0, Lx, length=Nx+1)

yC = range(Δy/2, Ly - Δy/2, length=Ny)
yF = range(0, Ly, length=Ny+1)

ϕ₀ = OffsetArray(zeros(Nx+2Hx, Ny+2Hy), -Hx+1:Nx+Hx, -Hy+1:Ny+Hy)
@. ϕ₀[1:Nx, 1:Ny] = ϕ_initial_condition(xC, yC')

tspan = (0.0, 10.0)
prob = ODEProblem(advection_diffusion_2d, ϕ₀, tspan, p, saveat=0.1)
sol = solve(prob, Tsit5())

function animate_advection_diffusion_2d(ϕ, x, y, t; filepath, resolution=(1280, 720), colormap=:matter, colorrange=(0, 1), framerate=15)
    Nx, Ny, Nt = size(ϕ)

    n = Observable(1)
    ϕₙ = @lift ϕ[:, :, $n]

    title = @lift @sprintf("2D advection-diffusion: t = %.1f", t[$n])

    fig = Figure(; resolution)
    ax = Axis(fig[1, 1], aspect=1, title=title, xlabel="x", ylabel="y")

    heatmap!(ax, x, y, ϕₙ; colormap, colorrange)
    Colorbar(fig[1, 2], colormap=colormap, limits=colorrange)

    record(fig, filepath, 1:Nt; framerate) do time_index
        @info "Animating 2D advection-diffusion: frame $time_index/$(length(t))"
        n[] = time_index
    end
end

ϕ_correct = Array(sol)[Hx+1:Nx+Hx, Hy+1:Ny+Hy, :]

animate_advection_diffusion_2d(ϕ_correct, xC, yC, sol.t, filepath="advection_diffusion_2d.mp4")

# Coarse graining the high-resolution solution to produce a "true solution" at the coarse resolution.

prob = remake(prob, saveat=1.0)
sol = solve(prob, Tsit5())

ϕ_correct = Array(sol)[Hx+1:Nx+Hx, Hy+1:Ny+Hy, :]

N_thinning = 8
Nx_coarse = Int(Nx / N_thinning)
Ny_coarse = Int(Ny / N_thinning)

Tx_coarse = Nx_coarse + 2Hx
Ty_coarse = Ny_coarse + 2Hy

Δx_coarse = Lx / Nx_coarse
Δy_coarse = Ly / Ny_coarse

Nt = length(sol.t)

ϕ_coarse_correct = zeros(Nx_coarse, Ny_coarse, Nt)

for i in 1:Nx_coarse, j in 1:Ny_coarse
    is = 1+(i-1)*N_thinning:i*N_thinning
    js = 1+(j-1)*N_thinning:j*N_thinning
    ϕ_coarse_correct[i, j, :] .= dropdims(mean(ϕ_correct[is, js, :], dims=(1, 2)), dims=(1, 2))
end

xC_coarse = range(Δx_coarse/2, Lx - Δx_coarse/2, length=Nx_coarse)
yC_coarse = range(Δy_coarse/2, Lx - Δy_coarse/2, length=Ny_coarse)

animate_advection_diffusion_2d(ϕ_coarse_correct, xC_coarse, yC_coarse, sol.t, filepath="advection_diffusion_2d_coarse.mp4", framerate=2)

# Setting up inference

is = -Hx+1:Nx_coarse+Hx
js = -Hy+1:Ny_coarse+Hy
ϕ₀_coarse = OffsetArray(zeros(Tx_coarse, Ty_coarse), is, js)
@. ϕ₀_coarse[1:Nx_coarse, 1:Ny_coarse] = ϕ_initial_condition(xC_coarse, yC_coarse')

p_coarse = [Nx_coarse, Ny_coarse, Hx, Hy, Δx_coarse, Δy_coarse, u, v, κˣ, κʸ]
prob_coarse = ODEProblem(advection_diffusion_2d, ϕ₀_coarse, tspan, p_coarse, saveat=1.0)

function predict_advection_diffusion_2d(prob, u, v, κˣ, κʸ)
    Nx, Ny, Hx, Hy, Δx, Δy, _ = prob.p
    p = [Nx, Ny, Hx, Hy, Δx, Δy, u, v, κˣ, κʸ]

    prob = remake(prob, p=p)
    prediction = solve(prob, Tsit5())

    return prediction
end

@model function probabilistic_advection_diffusion_2d(data, prob)
    σ ~ InverseGamma(3, 0.5)
    u ~ Normal(0, 1)
    v ~ Normal(0, 1)
    κˣ ~ InverseGamma(3, 0.1)
    κʸ ~ InverseGamma(3, 0.1)

    prediction = predict_advection_diffusion_2d(prob, u, v, κˣ, κʸ)

    Nx, Ny, Hx, Hy, _ = prob.p
    Nx, Ny, Hx, Hy = Int.((Nx, Ny, Hx, Hy))

    # for n in 1:length(prediction)
    #     prediction_n = prediction[n][1:Nx, 1:Ny][:]
    #     data[:, n] ~ MvNormal(prediction_n, σ)
    # end

    for n in 1:length(prediction), j in 1:Ny
        data[:, j, n] ~ MvNormal(prediction[n][1:Nx, j], σ)
    end

    # prediction_flattened = Array(prediction)[Hx+1:Nx+Hx, Hy+1:Ny+Hy, :][:]
    # data ~ MvNormal(prediction_flattened, σ)
end

function flatten_solution(ϕ)
    Nx, Ny, Nt = size(ϕ)
    ϕ_flat = zeros(Nx * Ny, Nt)
    for n in 1:Nt
        ϕ_flat[:, n] .= ϕ[1:Nx, 1:Ny, n][:]
    end
    return ϕ_flat
end

function plot_chains_and_densities(chains, vars; filepath)
    n_chains = size(chains, 3)

    fig = Figure(resolution=(800, 200*length(vars)))

    for (n, var) in enumerate(vars)
        ax1 = Axis(fig[n, 1], ylabel="$var", xlabel="iteration")
        for c in 1:n_chains
            lines!(ax1, chains[var].data[:, c])
        end

        ax2 = Axis(fig[n, 2], ylabel="P($var)", xlabel="$var", yticklabelsvisible=false)
        density!(ax2, chains[var][:].data)
    end

    save(filepath, fig, px_per_unit=2)

    return
end

prior_chain = sample(probabilistic_advection_diffusion_2d(ϕ_coarse_correct[:], prob_coarse), Prior(), 100)
plot_chains_and_densities(prior_chain, [:u, :v, :κˣ, :κʸ, :σ], filepath="priors.png")

function callback(rng, model, sampler, sample, state, iter; kwargs...)
    n = Threads.threadid()

    u  = sample.θ.u[1][1]
    v  = sample.θ.v[1][1]
    κˣ = sample.θ.κˣ[1][1]
    κʸ = sample.θ.κʸ[1][1]
    σ  = sample.θ.σ[1][1]

    @info "Chain $n, Iteration $iter: u=$u, v=$v, κˣ=$κˣ, κʸ=$κʸ, σ=$σ"
end

# n_chains = 8
# @assert Threads.nthreads() >= n_chains

stat_model = probabilistic_advection_diffusion_2d(ϕ_coarse_correct, prob_coarse)
# stat_model = probabilistic_advection_diffusion_2d(ϕ_coarse_correct |> flatten_solution, prob_coarse)

# chain = sample(stat_model, MH(), 100, progress=true; callback)
chain = sample(stat_model, NUTS(), 100, progress=true; callback)
# chain = sample(probabilistic_advection_diffusion_2d(ϕ_coarse_correct[:], prob_coarse), MH(), 1000, progress=true; callback)
# chains = sample(probabilistic_advection_diffusion_2d(ϕ_coarse_correct, prob_coarse), sampler, MCMCThreads(), n_samples, n_chains, progress=true, callback=callback_multi_chain)

plot_chains_and_densities(chain, [:u, :v, :κˣ, :κʸ, :σ], filepath="posteriors.png")
