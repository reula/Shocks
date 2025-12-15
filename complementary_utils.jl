
function RHS_Flux(u,t,par_RHS_Flux) #version to optimize
    Order, J, Box, dx, du, par_WENOZ = par_RHS_Flux
    wenoz_d_u!(du, u, par_WENOZ, t)
    return du[:]
end

function Burgers_Flux(u,t,par_Burgers_Flux) #version to optimize
    Order, J, Box, dx, du, par_WENOZ = par_Burgers_Flux
    wenoz_d_u!(du, u, par_WENOZ, t)
    return du[:]
end

function RK4_Step!(f,y0,t0,h,p)
    k1 = h*f(y0,t0,p)
    k2 = h*f(y0+0.5*k1, t0+0.5*h,p)
    k3 = h*f(y0+0.5*k2, t0+0.5*h,p)
    k4 = h*f(y0+k3, t0+h,p)
    y0 .= y0 + (k1 + 2k2 + 2k3 + k4)/6
end


function TVD3_Step!(f,y0,t0,h,pf)
  y1 = y0 + h*f(y0,t0,pf) 
  y2 = (3*y0 + y1 + h*f(y1,t0,pf))/4
  y0 .= (y0 + 2*y2 + 2*h*f(y2,t0,pf))/3
end

function burgers_rhs!(du, u, t, params)
        dx, η = params
        n = length(u) - 5  # assuming 3 ghost cells on each side. This n is n+1 counting real averages.
        # Reconstruct u and u_x at interfaces
        uL, uR, duL, duR = WENOZ_FV_reconstruct_from_averages(u, dx)
        duL, duR = WENOZ_derivatives_form_uLR(uL, uR, dx)
        # Compute fluxes at interfaces
        fL = 0.5 * uL.^2 - η * duL
        fR = 0.5 * uR.^2 - η * duR
        
        # Simple Rusanov flux
        #λ = maximum(abs.(u))
        λ = max.(maximum(abs.(uL)), maximum(abs.(uR)))
        #λ = abs.(u)
        fnum =  0.5 * (fL + fR - λ .* (uR - uL))
        
        #@show length(uL)  # Should be n 
        # Update cell averages
        #du[:] = -(fnum - circshift(fnum, 1)) / dx
        du[4:n+2] = -(fnum[2:n] - fnum[1:n-1]) / dx
        return nothing
    end

    function burgers_rhs(u, t, params)
        dx, η, du = params
        du = zeros(length(u))
        burgers_rhs!(du, u, t, (dx, η))
        return du
    end

function burgers_rhs_wgrad!(du, u, t, params)
        dx, η, ζ = params
        n = length(u) - 5  # assuming 3 ghost cells on each side. This n is n+1 counting real averages.
        # Reconstruct u and u_x at interfaces
        uL, uR, duL, duR = WENOZ_FV_reconstruct_from_averages(u, dx)
        duL, duR = WENOZ_derivatives_form_uLR(uL, uR, dx)
        # Compute fluxes at interfaces
        fL = 0.5 * uL.^2 - η * duL + ζ * (duL.^2)
        fR = 0.5 * uR.^2 - η * duR + ζ * (duR.^2)
        
        # Simple Rusanov flux
        #λ = maximum(abs.(u))
        λ = max.(maximum(abs.(uL)), maximum(abs.(uR)))
        #λ = abs.(u)
        fnum =  0.5 * (fL + fR - λ .* (uR - uL))
        
        #@show length(uL)  # Should be n 
        # Update cell averages
        #du[:] = -(fnum - circshift(fnum, 1)) / dx
        du[4:n+2] = -(fnum[2:n] - fnum[1:n-1]) / dx
        return nothing
    end

    function burgers_rhs_wgrad(u, t, params)
        dx, η, ζ, du = params
        du = zeros(length(u))
        burgers_rhs_wgrad!(du, u, t, (dx, η, ζ))
        return du
    end
