using Test

# include project utilities
include(joinpath(@__DIR__, "..", "choques_utils.jl"))
# complementary_utils.jl is included so its TVD routines are available if you want to call them.

include(joinpath(@__DIR__, "..", "complementary_utils.jl"))

@testset "Burgers + WENO-Z FV reconstruction (u and u_x)" begin
    # Grid setup
    n = 64
    L = 2π
    dx = L / n
    x_centers = ((0:n-1) .* dx) .+ dx/2
    
    # Parameters for evolution
    η = 0.1  # viscosity parameter
    dt = 0.1 * dx  # CFL-based timestep
    t_final = 0.1
    
    # Initial condition: smooth bump
    u0 = @. exp(-(x_centers - L/2)^2 * 4)
    u = copy(u0)
    
    # Arrays for WENO reconstruction
    uL = zeros(n)
    uR = zeros(n)
    duL = zeros(n)
    duR = zeros(n)
    uxL = zeros(n)
    uxR = zeros(n)
    duxL = zeros(n)
    duxR = zeros(n)
    du = zeros(n)
    
    # RHS function for Burgers equation with viscosity
    function burgers_rhs!(du, u, params, t)
        dx, η = params
        
        # Reconstruct u and u_x at interfaces
        WENOZ_FV_reconstruct_from_averages!(uL, uR, duL, duR, u, dx)
        
        # Compute fluxes at interfaces
        fL = @. 0.5 * uL^2 + η * duL
        fR = @. 0.5 * uR^2 + η * duR
        
        # Simple Rusanov flux
        λ = maximum(abs.(u)) + η/dx
        fnum = @. 0.5 * (fL + fR - λ * (uR - uL))
        
        # Update cell averages
        @. du = -(fnum - circshift(fnum, 1)) / dx
        
        return nothing
    end

    function burgers_rhs(u, params, t)
        dx, η = params
        
        # Reconstruct u and u_x at interfaces
        WENOZ_FV_reconstruct_from_averages!(uL, uR, duL, duR, u, dx)
        
        # Compute fluxes at interfaces
        fL = @. 0.5 * uL^2 + η * duL
        fR = @. 0.5 * uR^2 + η * duR
        
        # Simple Rusanov flux
        λ = maximum(abs.(u)) + η/dx
        fnum = @. 0.5 * (fL + fR - λ * (uR - uL))
        
        # Update cell averages
        @. du = -(fnum - circshift(fnum, 1)) / dx
        
        return du[:]
    end

    # Time evolution with TVD3
    t = 0.0
    params = (dx, η)
    nsteps = round(Int, t_final/dt)
    
    # Basic tests before evolution
    @test length(u) == n
    @test maximum(abs.(u .- u0)) < 1e-14
    
    # Evolve
    for i in 1:nsteps
        (f,y0,t0,h,pf)
        TVD3_Step(burgers_rhs, u, t, dt, , params)
        t += dt
    end
    
    # Basic sanity checks after evolution
    @test !any(isnan.(u))  # No NaNs
    @test maximum(abs.(u)) ≤ maximum(abs.(u0)) * 1.1  # Max principle approximately
    @test sum(u) * dx ≈ sum(u0) * dx rtol=1e-3  # Mass conservation
end