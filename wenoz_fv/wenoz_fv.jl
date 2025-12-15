module WENOZFV

export reconstruct_from_cell_averages

"""
    reconstruct_from_cell_averages(uavg; order=5, dx=1.0)

Finite-volume WENO-Z reconstruction (5th order) from cell averages to interface points.
Returns four vectors (length N):
- uL: left-biased values at interfaces i+1/2 (from cell i)
- uR: right-biased values at interfaces i+1/2 (from cell i+1)
- duL: left-biased derivatives at interfaces i+1/2
- duR: right-biased derivatives at interfaces i+1/2

Assumes periodic boundary conditions. Derivatives are scaled by 1/dx.
"""
function reconstruct_from_cell_averages(uavg::AbstractVector{<:Real}; order::Int=5, dx::Real=1.0)
    @assert order == 5 "Only 5th-order WENO-Z is implemented"
    n = length(uavg)
    u = collect(Float64, uavg)
    uL = zeros(n)
    uR = zeros(n)
    duL = zeros(n)
    duR = zeros(n)

    periodic_index(i) = (mod(i-1, n) + 1)

    # Value reconstruction at interfaces using standard FV-WENO5-Z
    for i in 1:n
        im2 = periodic_index(i-2)
        im1 = periodic_index(i-1)
        i0  = periodic_index(i)
        ip1 = periodic_index(i+1)
        ip2 = periodic_index(i+2)

        # Candidates at x_{i+1/2}
        Q0 = (2.0*u[im2] - 7.0*u[im1] + 11.0*u[i0])
        Q1 = (-1.0*u[im1] + 5.0*u[i0] + 2.0*u[ip1])
        Q2 = (2.0*u[i0] + 5.0*u[ip1] - 1.0*u[ip2])

        B1 = 13.0/12.0
        b0 = B1*(u[i0] - 2*u[ip1] + u[ip2])^2 + 0.25*(3*u[i0] - 4*u[ip1] + u[ip2])^2
        b1 = B1*(u[im1] - 2*u[i0] + u[ip1])^2 + 0.25*(u[im1] - u[ip1])^2
        b2 = B1*(u[im2] - 2*u[im1] + u[i0])^2 + 0.25*(u[im2] - 4*u[im1] + 3*u[i0])^2

        eps = 1e-40
        tau5 = abs(b2 - b0)
        a0 = 0.1*(1.0 + (tau5/(b0 + eps))^2)
        a1 = 0.6*(1.0 + (tau5/(b1 + eps))^2)
        a2 = 0.3*(1.0 + (tau5/(b2 + eps))^2)
        an = a0 + a1 + a2
        uL[i] = (a0*Q0 + a1*Q1 + a2*Q2) * (1.0/6.0) / an

        # Right state at i+1/2 (mirror)
        Q0r = (2.0*u[ip2] - 7.0*u[ip1] + 11.0*u[i0])
        Q1r = (-1.0*u[ip1] + 5.0*u[i0] + 2.0*u[im1])
        Q2r = (2.0*u[i0] + 5.0*u[im1] - 1.0*u[im2])

        b0r = B1*(u[i0] - 2*u[im1] + u[im2])^2 + 0.25*(3*u[i0] - 4*u[im1] + u[im2])^2
        b1r = B1*(u[ip1] - 2*u[i0] + u[im1])^2 + 0.25*(u[ip1] - u[im1])^2
        b2r = B1*(u[ip2] - 2*u[ip1] + u[i0])^2 + 0.25*(u[ip2] - 4*u[ip1] + 3*u[i0])^2

        tau5r = abs(b2r - b0r)
        a0r = 0.1*(1.0 + (tau5r/(b0r + eps))^2)
        a1r = 0.6*(1.0 + (tau5r/(b1r + eps))^2)
        a2r = 0.3*(1.0 + (tau5r/(b2r + eps))^2)
        anr = a0r + a1r + a2r
        uR[i] = (a0r*Q0r + a1r*Q1r + a2r*Q2r) * (1.0/6.0) / anr
    end

    # Derivative at centers (Δx assumed 1 here), then WENO-Z to interfaces, then scale by 1/dx
    du_center = zeros(n)
    for i in 1:n
        im2 = periodic_index(i-2)
        im1 = periodic_index(i-1)
        ip1 = periodic_index(i+1)
        ip2 = periodic_index(i+2)
        du_center[i] = (u[im2] - 8.0*u[im1] + 8.0*u[ip1] - u[ip2]) / 12.0
    end

    for i in 1:n
        im2 = periodic_index(i-2)
        im1 = periodic_index(i-1)
        i0  = periodic_index(i)
        ip1 = periodic_index(i+1)
        ip2 = periodic_index(i+2)

        q0 = (2.0*du_center[im2] - 7.0*du_center[im1] + 11.0*du_center[i0])
        q1 = (-1.0*du_center[im1] + 5.0*du_center[i0] + 2.0*du_center[ip1])
        q2 = (2.0*du_center[i0] + 5.0*du_center[ip1] - 1.0*du_center[ip2])

        B1 = 13.0/12.0
        b0 = B1*(du_center[i0] - 2*du_center[ip1] + du_center[ip2])^2 + 0.25*(3*du_center[i0] - 4*du_center[ip1] + du_center[ip2])^2
        b1 = B1*(du_center[im1] - 2*du_center[i0] + du_center[ip1])^2 + 0.25*(du_center[im1] - du_center[ip1])^2
        b2 = B1*(du_center[im2] - 2*du_center[im1] + du_center[i0])^2 + 0.25*(du_center[im2] - 4*du_center[im1] + 3*du_center[i0])^2
        eps = 1e-40
        tau5 = abs(b2 - b0)
        a0 = 0.1*(1.0 + (tau5/(b0 + eps))^2)
        a1 = 0.6*(1.0 + (tau5/(b1 + eps))^2)
        a2 = 0.3*(1.0 + (tau5/(b2 + eps))^2)
        an = a0 + a1 + a2
        duL[i] = (a0*q0 + a1*q1 + a2*q2) * (1.0/6.0) / an

        q0r = (2.0*du_center[ip2] - 7.0*du_center[ip1] + 11.0*du_center[i0])
        q1r = (-1.0*du_center[ip1] + 5.0*du_center[i0] + 2.0*du_center[im1])
        q2r = (2.0*du_center[i0] + 5.0*du_center[im1] - 1.0*du_center[im2])
        b0r = B1*(du_center[i0] - 2*du_center[im1] + du_center[im2])^2 + 0.25*(3*du_center[i0] - 4*du_center[im1] + du_center[im2])^2
        b1r = B1*(du_center[ip1] - 2*du_center[i0] + du_center[im1])^2 + 0.25*(du_center[ip1] - du_center[im1])^2
        b2r = B1*(du_center[ip2] - 2*du_center[ip1] + du_center[i0])^2 + 0.25*(du_center[ip2] - 4*du_center[ip1] + 3*du_center[i0])^2
        tau5r = abs(b2r - b0r)
        a0r = 0.1*(1.0 + (tau5r/(b0r + eps))^2)
        a1r = 0.6*(1.0 + (tau5r/(b1r + eps))^2)
        a2r = 0.3*(1.0 + (tau5r/(b2r + eps))^2)
        anr = a0r + a1r + a2r
        duR[i] = (a0r*q0r + a1r*q1r + a2r*q2r) * (1.0/6.0) / anr
    end

    # scale derivative by 1/dx because du_center assumed Δx=1
    duL .*= (1.0/dx)
    duR .*= (1.0/dx)

    return uL, uR, duL, duR
end

end # module


