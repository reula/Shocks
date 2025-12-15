# Example usage and testing
using Plots

include("choques_utils.jl")

function test_weno_reconstruction()
    println("Testing WENO Reconstruction (FV WENO-Z)")
    n = 200
    x = range(0, 2π, length=n+1)           # interfaces
    dx = x[2] - x[1]
    xc = (x[1:end-1] .+ x[2:end]) ./ 2     # centers
    f(x) = sin(x)
    fp(x) = cos(x)
    F(x) = -cos(x)
    uavg = (F.(xc .+ dx/2) .- F.(xc .- dx/2)) ./ dx
    uL, uR, duL, duR = WENOZ_FV_reconstruct_from_averages(uavg, dx)
    xi = x[2:end]
    exact_u = f.(xi)
    exact_du = fp.(xi)

    p1 = plot(xi, exact_u, label="u exact", lw=2)
    plot!(p1, xi, uL, label="uL", ls=:dash)
    p2 = plot(xi, exact_du, label="du exact", lw=2)
    plot!(p2, xi, duL, label="duL", ls=:dash)
    p = plot(p1, p2, layout=(2,1), size=(800,600))
    return maximum(abs.(duL .- exact_du)), p
end

function test_discontinuous_data()
    println("\n2. Testing with discontinuous data (FV WENO-Z)...")
    n = 400
    x = range(0, 2π, length=n+1)           # interfaces
    dx = x[2] - x[1]
    xc = (x[1:end-1] .+ x[2:end]) ./ 2
    f(x) = x < π ? sin(x) : 0.5*sin(x) + 0.5
    # midpoint average for visualization only
    uavg = f.(xc)
    uL, uR, duL, duR = WENOZ_FV_reconstruct_from_averages(uavg, dx)
    xi = x[2:end]
    p = plot(xi, f.(xi), label="data (midpoint)", lw=1)
    plot!(p, xi, uL, label="uL", ls=:dash)
    plot!(p, xi, uR, label="uR", ls=:dashdot)
    plot!(p, title="FV WENO-Z Reconstruction - Discontinuous", xlabel="x", ylabel="u")
    return p
end

# Convergence test for FV WENO-Z from cell averages to interfaces (integrated in choques_utils.jl)
function convergence_test_fv_wenoz()
    println("Convergence test (cell averages -> interfaces) with WENO-Z (order 5)")
    Ns = [50, 100, 200, 400]
    errs_u = Float64[]
    errs_du = Float64[]

    for n in Ns
        x = range(0, 2π, length=n+1)            # interfaces, periodic
        dx = x[2] - x[1]
        xc = (x[1:end-1] .+ x[2:end]) ./ 2      # cell centers

        f(x) = sin(x) + 0.3*sin(2x)
        fp(x) = cos(x) + 0.6*cos(2x)
        F(x) = -cos(x) - 0.15*cos(2x)           # antiderivative of f

        # exact cell averages
        uavg = (F.(xc .+ dx/2) .- F.(xc .- dx/2)) ./ dx
        uL, uR, duL, duR = WENOZ_FV_reconstruct_from_averages(uavg, dx)

        xi_half = x[2:end]
        u_exact = f.(xi_half)
        du_exact = fp.(xi_half)

        err_u = sqrt(sum((uL .- u_exact).^2) / n)
        err_du = sqrt(sum((duL .- du_exact).^2) / n)

        push!(errs_u, err_u)
        push!(errs_du, err_du)
    end

    rates_u = [log2(errs_u[i-1]/errs_u[i]) for i in 2:length(errs_u)]
    rates_du = [log2(errs_du[i-1]/errs_du[i]) for i in 2:length(errs_du)]

    println("N:       ", Ns)
    println("L2(u):   ", errs_u)
    println("Rate(u): ", rates_u)
    println("L2(du):  ", errs_du)
    println("Rate(du):", rates_du)
end

# Run tests
if true #abspath(PROGRAM_FILE) == @__FILE__
    convergence_test_fv_wenoz()
end