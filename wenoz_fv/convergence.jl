include("wenoz_fv.jl")
import .WENOZFV

function convergence_test()
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

        # exact cell averages: (1/dx) * (F(x+dx/2) - F(x-dx/2))
        uavg = (F.(xc .+ dx/2) .- F.(xc .- dx/2)) ./ dx
        uL, uR, duL, duR = WENOZFV.reconstruct_from_cell_averages(uavg; order=5, dx=dx)

        xi_half = @view x[2:end]                # interfaces i+1/2 aligned with uL index
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

if abspath(PROGRAM_FILE) == @__FILE__
    convergence_test()
end


