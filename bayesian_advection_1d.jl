using OffsetArrays
using DifferentialEquations
using CairoMakie

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

Nx = 256
Hx = 1
Lx = 1
Δx = Lx / Nx
u = 0.2
p = [Nx, Hx, Δx, u]

xC = range(Δx/2, Lx - Δx/2, length=Nx)
xF = range(0, Lx, length=Nx+1)

ϕ₀ = OffsetArray(zeros(Nx+2Hx), -Hx+1:Nx+Hx)
@. ϕ₀[1:Nx] = exp(-100*(xC-0.5)^2)

tspan = (0.0, 10.0)
prob = ODEProblem(advection, ϕ₀, tspan, p, saveat=0.1)
sol = solve(prob, Tsit5())

function animate_advection(ϕ, x, t; framerate=30)
    n = Observable(1)
    ϕₙ = @lift ϕ[:, $n]

    title = @lift "Advection: t = $(round(t[$n], digits=1))"

    fig = Figure()
    ax = Axis(fig[1, 1], title=title, xlabel="x", ylabel="ϕ")
    lines!(ax, x, ϕₙ)

    time_indices = 1:length(t)
    record(fig, "advection_1d.mp4", time_indices; framerate) do time_index
        @info "Animating 1D advection: frame $time_index/$(length(t))"
        n[] = time_index
    end
end

ϕ = Array(sol)[Hx+1:Nx+Hx, :]
animate_advection(ϕ, xC, sol.t)
