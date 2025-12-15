using StaticArrays

@inline function mysign_zero(a)
    return (1.0.*(a .> 0.0) + (-1.0).* (a .< 0.0))
end

@inline function minmod(a, b)
    return 0.5*(mysign_zero(a)+mysign_zero(b))*min(abs(a), abs(b))
end

@inline function minmod(a, b, c)
    sgnbc = (mysign_zero(b)+mysign_zero(c)) #this is 2 if both are positive, -2 if both are negative, 0 otherwise
    sgnac = (mysign_zero(a)+mysign_zero(c)) #this is 2 if both are positive, -2 if both are negative, 0 otherwise
    
    return 0.25*sgnbc*abs(sgnac)*min(abs(a), abs(b),abs(c))
end

@inline function minmod(a,b,c,d)
    return 0.125*(mysign_zero(a)+mysign_zero(b))*(abs((mysign_zero(a)+mysign_zero(c))*(mysign_zero(a)+mysign_zero(d))))*min(abs(a),abs(b),abs(c), abs(d))
end

function MM3(a::AbstractFloat, b::AbstractFloat, c::AbstractFloat, weight::AbstractFloat) #(2*D0,Dp,Dm)
    
    weight = weight * 2.
  
    if (abs(a) <= (weight*abs(b))) 
        return (abs(a) <= (weight*abs(c))) ? abs(a)*.5 : abs(c) 
    else 
        return (abs(b) <= abs(c)) ? abs(b) : abs(c)
    end 
end
#MM3(4.,-1.,4.,2.5)
function MM3N(a::AbstractFloat, b::AbstractFloat, c::AbstractFloat) #(2*D0,Dp,Dm)
    if (abs(a) <= abs(b))
        return (abs(a) <= abs(c)) ? abs(a) : abs(c) 
    else 
        return (abs(b) <= abs(c)) ? abs(b) : abs(c)
    end 
end
#MM3(4.,-1.,4.,2.5)


function DMM(a::AbstractFloat, b::AbstractFloat)
    return 0.5 * (mysign_zero(a) + mysign_zero(b)) * minimum([abs(a),abs(b)])
end
#DMM(2.,2.)

function DM4(a::AbstractFloat, b::AbstractFloat, c::AbstractFloat, d::AbstractFloat)
    return 0.125 * (mysign_zero(a) + mysign_zero(b)) * abs((mysign_zero(a) + mysign_zero(c)) * (mysign_zero(a) + mysign_zero(d))) * minimum([abs(a),abs(b),abs(c),abs(d)])
end
#DM4(1.,2.,3.,0.)

#método de Kurganov-Tadmor

function KT!(dfields, fields, par, t)
    #Los parámetros son h, θ, funciones auxiliares y vectores auxiliares
    eqpars, h::Float64, θ::Float64, Fx!, MaxSpeed, N::Int64, N_FIELDS::Int32, auxvectors = par
    Dm, D, Dp, u_mm, u_mp, u_pm, u_pp, F_mm, F_mp, F_pm, F_pp, H_m, H_p = auxvectors

    for idx in 1:N
        idxll = mod(((idx-2) - 1),N) + 1
        idxl = mod(((idx-1) - 1),N) + 1
        idxr = mod(((idx+1) - 1),N) + 1
        idxrr = mod(((idx+2) - 1),N) + 1
        
        fll = @view fields[idxll,:]
        fl = @view fields[idxl,:]
        f = @view fields[idx,:]
        fr = @view fields[idxr,:]
        frr = @view fields[idxrr,:]
        
        @. Dm = minmod(0.5 *(f - fll), θ*(f-fl), θ*(fl-fll)) # * h
        @. D  = minmod(0.5 *(fr - fl), θ*(fr-f), θ*(f-fl)) # * h
        @. Dp = minmod(0.5*(frr-f), θ*(frr-fr), θ*(fr-f)) # * h
        @. u_mm = fl + 0.5*Dm  #/h
        @. u_mp = f - 0.5*D #/h
        @. u_pm = f + 0.5*D #/h
        @. u_pp = fr - 0.5*Dp #/h
        
        Fx!(F_mm, u_mm, eqpars)
        Fx!(F_mp, u_mp, eqpars)
        Fx!(F_pm, u_pm, eqpars)
        Fx!(F_pp, u_pp, eqpars)
        
        
        a_m::Float64 = max(MaxSpeed(u_mm, eqpars), MaxSpeed(u_mp, eqpars))       
        a_p::Float64 = max(MaxSpeed(u_pm, eqpars), MaxSpeed(u_pp, eqpars))        

        
        @. H_m = 0.5 * (F_mp + F_mm) - 0.5 * a_m * (u_mp - u_mm)
        @. H_p = 0.5 * (F_pp + F_pm) - 0.5 * a_p * (u_pp - u_pm)
        
        @. dfields[idx, :] = -h*(H_p - H_m)
    end
end

function createKTauxvectors(N_FIELDS)
    D = Array{Float64}(undef, N_FIELDS)
    Dm = copy(D)
    Dp = copy(D)
    umm = copy(D)
    ump = copy(D)
    upm = copy(D)
    upp = copy(D)
    Fmm = copy(D)
    Fmp = copy(D)
    Fpm = copy(D)
    Fpp = copy(D)
    Hm = copy(D)
    Hp = copy(D)
    return (Dm, D, Dp, umm, ump, upm, upp, Fmm, Fmp, Fpm, Fpp, Hm, Hp)
end



#==================================MP5=====================================#

#Reconstrucción
function MP5reconstruction!(Vl, Vjmm, Vjm, Vj, Vjp, Vjpp, N_Fields)
    B1 = 0.0166666666666666667  #1/60
    B2 = 1.3333333333333333333  #4/3
    eps = 1e-10
    ALPHA = 4.0
    #=Vjmm = V[1]
    Vjm = V[2]
    Vj = V[3]
    Vjp = V[4]
    Vjpp = V[5]=#
    for i in 1:N_Fields
        Vor = B1*(2.0*Vjmm[i] - 13.0*Vjm[i] + 47.0*Vj[i] + 27*Vjp[i] - 3.0*Vjpp[i]) #=This is the original interpolation.
                                                                       All that follows is the application of 
                                                                       limiters to treat shocks=#
        Vmp = Vj[i] + minmod(Vjp[i]-Vj[i], ALPHA*(Vj[i]-Vjm[i]))  #mp = monotonicity preserving. It's the median between v_j, v_(j+1)
                                                  #and an upper limit v^UL = v_j+ALPHA(v_j-v_(j-1))
        if ((Vor-Vj[i])*(Vor-Vmp)) < eps             #this condition is equivalent to asking vl in [vj, v^{MP}]
            Vl[i] = Vor #vl = v^{L}_{j+1/2}
        else
            djm1 = Vjmm[i] - 2.0*Vjm[i] + Vj[i]
            dj = Vjm[i] - 2*Vj[i] + Vjp[i]
            djp1 = Vj[i] - 2.0*Vjp[i] + Vjpp[i]
            dm4jph = minmod(4*dj - djp1, 4*djp1-dj, dj, djp1)  #ph = plus half (+1/2)
            dm4jmh = minmod(4*dj - djm1, 4*djm1-dj, dj, djm1)  #mh = minus half (-1/2)
            #d^{M4}_{j+1/2} = \minmod(4d_{j}-d_{j+1},4d_{j+1}-d_{j}, d_{j}, d_{j+1})
            Vul = Vj[i] + ALPHA*(Vj[i] - Vjm[i])   #upper limit
            Vav = 0.5*(Vj[i] + Vjp[i])          #average
            Vmd = Vav - 0.5*dm4jph        #Vmedian
            Vlc = Vj[i] + 0.5*(Vj[i]-Vjm[i]) + B2*dm4jmh
            Vmin = max(min(Vj[i], Vjp[i], Vmd), min(Vj[i], Vul, Vlc));
            Vmax = min(max(Vj[i], Vjp[i], Vmd), max(Vj[i], Vul, Vlc));
            Vl[i] = Vor + minmod(Vmin-Vor, Vmax-Vor) #this places Vor between Vmin and Vmax
        end
    end
end 

#Implementación de MP5 con el Flux Splitting de Lax

function mp5!(dfields, fields, par, t) # j is the grid position
    #asumimos u unidimensional por ahora
    par_eq, h, N, N_Fields, Fx!, Speed_max, auxvectors = par
    F_Mm3, F_Mm2, F_Mm1, F_M, F_Mp1, F_Mp2, F_Mp3, F_Pm3, F_Pm2, F_Pm1, F_P, F_Pp1, F_Pp2, F_Pp3, F_LP, F_LM, F_RP, F_RM, H_m, H_p = auxvectors
    

    #nota: f minuscula o u se usa para hablar de campos, F mayúscula para hablar de Flujos.
    
    for idx in 1:N
        #first we defined shifted indices
        idxm3 = mod(((idx-3) - 1),N) + 1
        idxm2 = mod(((idx-2) - 1),N) + 1
        idxm1 = mod(((idx-1) - 1),N) + 1
        idxp1 = mod(((idx+1) - 1),N) + 1
        idxp2 = mod(((idx+2) - 1),N) + 1
        idxp3 = mod(((idx+3) - 1),N) + 1
        
    
        um3 = @view fields[idxm3,:]
        um2 = @view fields[idxm2,:]
        um1 = @view fields[idxm1,:]
        u   = @view fields[idx,:]
        up1 = @view fields[idxp1,:]
        up2 = @view fields[idxp2,:]
        up3 = @view fields[idxp3,:]
        
        S_MAX = max(Speed_max(up3, par_eq), Speed_max(um3, par_eq), 
            Speed_max(up2, par_eq), Speed_max(um2, par_eq), Speed_max(up1, par_eq), 
            Speed_max(um1, par_eq), Speed_max(u, par_eq)) #maximum speed
        
        Fx!(F_Pm3, um3, par_eq)
        Fx!(F_Pm2, um2, par_eq)
        Fx!(F_Pm1, um1, par_eq)
        Fx!(F_P, u, par_eq)
        Fx!(F_Pp1, up1, par_eq)
        Fx!(F_Pp2, up2, par_eq)
        Fx!(F_Pp3, up3, par_eq)
        
        
        @. F_Mm3 = 0.5 * (F_Pm3 - S_MAX * um3)
        @. F_Mm2 = 0.5 * (F_Pm2 - S_MAX * um2)
        @. F_Mm1 = 0.5 * (F_Pm1 - S_MAX * um1)
        @. F_M   = 0.5 * (F_P   - S_MAX * u)
        @. F_Mp1 = 0.5 * (F_Pp1 - S_MAX * up1)
        @. F_Mp2 = 0.5 * (F_Pp2 - S_MAX * up2)
        @. F_Mp3 = 0.5 * (F_Pp3 - S_MAX * up3)
        @. F_Pm3 = 0.5 * (F_Pm3 + S_MAX * um3)
        @. F_Pm2 = 0.5 * (F_Pm2 + S_MAX * um2)
        @. F_Pm1 = 0.5 * (F_Pm1 + S_MAX * um1)
        @. F_P   = 0.5 * (F_P   + S_MAX * u)
        @. F_Pp1 = 0.5 * (F_Pp1 + S_MAX * up1)
        @. F_Pp2 = 0.5 * (F_Pp2 + S_MAX * up2)
        @. F_Pp3 = 0.5 * (F_Pp3 + S_MAX * up3)
    
        MP5reconstruction!(F_RM, F_Mp2, F_Mp1,  F_M,  F_Mm1, F_Mm2, N_Fields)
        MP5reconstruction!(F_LM, F_Pm3, F_Pm2, F_Pm1, F_P,  F_Pp1, N_Fields)
        MP5reconstruction!(F_LP, F_Pm2, F_Pm1,  F_P,  F_Pp1, F_Pp2, N_Fields)
        MP5reconstruction!(F_RP, F_Mp3, F_Mp2, F_Mp1, F_M,  F_Mm1, N_Fields)
        
        @. H_p = F_LP + F_RP
        @. H_m = F_LM + F_RM
        
        @. dfields[idx, :] = -h*(H_p - H_m)
        
    end
    
end

function createMP5auxvectors(N_FIELDS)
    F_P = Array{Float64}(undef, N_FIELDS)
    F_P = copy(F_P)
    F_M = copy(F_P)
    F_Pm3 = copy(F_P)
    F_Pm2 = copy(F_P)
    F_Pm1 = copy(F_P)
    F_Pp1 = copy(F_P)
    F_Pp2 = copy(F_P)
    F_Pp3 = copy(F_P)
    F_Mm3 = copy(F_P)
    F_Mm2 = copy(F_P)
    F_Mm1 = copy(F_P)
    F_Mp1 = copy(F_P)
    F_Mp2 = copy(F_P)
    F_Mp3 = copy(F_P)

    F_LP = copy(F_P)
    F_LM = copy(F_P)
    F_RM = copy(F_P)
    F_RP = copy(F_P)
    H_m = copy(F_P)
    H_p = copy(F_P)
    return (F_Mm3, F_Mm2, F_Mm1, F_M, F_Mp1, F_Mp2, F_Mp3, F_Pm3, F_Pm2, F_Pm1, F_P, F_Pp1, F_Pp2, F_Pp3, F_LP, F_LM, F_RP, F_RM, H_m, H_p)
end

#====================WENOZ====================#

function createWENOZvectors(N_FIELDS)
    F_Mm3 = Array{Float64}(undef, N_FIELDS)
    F_Mm2 = copy(F_Mm3)
    F_Mm1 = copy(F_Mm3)
    F_M   = copy(F_Mm3)
    F_Mp1 = copy(F_Mm3)
    F_Mp2 = copy(F_Mm3)
    F_Mp3 = copy(F_Mm3)
    F_Pm3 = copy(F_Mm3)
    F_Pm2 = copy(F_Mm3)
    F_Pm1 = copy(F_Mm3)
    F_P   = copy(F_Mm3)
    F_Pp1 = copy(F_Mm3)
    F_Pp2 = copy(F_Mm3)
    F_Pp3 = copy(F_Mm3)
    F_LP  = copy(F_Mm3)
    F_LM  = copy(F_Mm3)
    F_RP  = copy(F_Mm3)
    F_RM  = copy(F_Mm3)
    H_m   = copy(F_Mm3)
    H_p   = copy(F_Mm3)
    sourcevec = copy(F_Mm3)
    
    return (F_Mm3, F_Mm2, F_Mm1, F_M, F_Mp1, F_Mp2, F_Mp3, F_Pm3, F_Pm2, F_Pm1, F_P, F_Pp1, F_Pp2, F_Pp3, F_LP, F_LM, F_RP, F_RM, H_m, H_p, sourcevec)
end

function WENOZreconstruction!(Vl, Vjmm, Vjm, Vj, Vjp, Vjpp, N_Fields)
    B1 = 1.0833333333333333333  #13/12
    B2 = 0.1666666666666666666  #1/6
    
    eps = 1e-40

    for i in 1:N_Fields
        Q0 = 2.0*Vj[i] +5.0*Vjp[i] - 1.0*Vjpp[i]
        Q1 = -Vjm[i] + 5.0*Vj[i] + 2.0*Vjp[i]
        Q2 = 2.0*Vjmm[i] - 7.0*Vjm[i] + 11* Vj[i]


        β0 =  B1*(Vj[i] - 2*Vjp[i] + Vjpp[i])^2 + 0.25*(3*Vj[i] - 4*Vjp[i]+ Vjpp[i])^2
        β1 =  B1*(Vjm[i] - 2*Vj[i] + Vjp[i])^2 + 0.25*(Vjm[i] - Vjp[i])^2
        β2 =  B1*(Vjmm[i] - 2*Vjm[i] + Vj[i])^2 + 0.25*(Vjmm[i] - 4*Vjm[i]+ 3*Vj[i])^2


        τ5 = abs(β2 - β0)

        α0 = 0.3*(1.0 + (τ5/(β0 + eps))^2)
        α1 = 0.6*(1.0 + (τ5/(β1 + eps))^2)
        α2 = 0.1*(1.0 + (τ5/(β2 + eps))^2)

        alphasum = (α0 + α1 + α2)

        Vl[i] = (α0*Q0 + α1*Q1 + α2*Q2)*B2/alphasum
    end
end 


function wenoz!(du_fields, u_fields, par, t) # j is the grid position
    #asumimos u unidimensional por ahora
    #en esta version primero están los campos y luego los puntos del espacio.
    par_eq, hh, J, N_Fields, F!, Speed_max, auxvecs = par
    F_Mm3, F_Mm2, F_Mm1, F_M, F_Mp1, F_Mp2, F_Mp3, F_Pm3, F_Pm2, F_Pm1, F_P, F_Pp1, F_Pp2, F_Pp3, F_LP, F_LM, F_RP, F_RM, H_m, H_p, sourcevec = auxvecs
    
    fields = reshape(u_fields, N_Fields, J...)
    dfields = reshape(du_fields, N_Fields, J...)


    #nota: f minuscula o u se usa para hablar de campos, F mayúscula para hablar de Flujos.
    if length(J) == 1
        Fx! = F!
        N = J[1]
        h = hh
        for idx in 1:N
        
            #first we defined shifted indices. The mod function make the indices periodic.
            #Primero definimos los indices desplazados. La función mod hace que los índices sean periódicos.
            idxm3 = mod(((idx-3) - 1),N) + 1
            idxm2 = mod(((idx-2) - 1),N) + 1
            idxm1 = mod(((idx-1) - 1),N) + 1
            idxp1 = mod(((idx+1) - 1),N) + 1
            idxp2 = mod(((idx+2) - 1),N) + 1
            idxp3 = mod(((idx+3) - 1),N) + 1
        
        
            #We give a name to the fields on the stencil points (idx-3, idx-2, idx-1, idx, idx+1, idx+2 and idx+3)
            #Le damos un nombre a los campos en los puntos del stencil (idx-3, idx-2, idx-1, idx, idx+1, idx+2 y idx+3)
            um3 = @view fields[:,idxm3]
            um2 = @view fields[:,idxm2]
            um1 = @view fields[:,idxm1]
            u   = @view fields[:,idx]
            up1 = @view fields[:,idxp1]
            up2 = @view fields[:,idxp2]
            up3 = @view fields[:,idxp3]
        
            #We calculate the maximum propagation speed inside the stencil
            #Calculamos la velocidad de propagación máxima dentro del stencil
            S_MAX = max(Speed_max(up3, par_eq), Speed_max(um3, par_eq), 
                Speed_max(up2, par_eq), Speed_max(um2, par_eq), Speed_max(up1, par_eq), 
                Speed_max(um1, par_eq), Speed_max(u, par_eq)) #maximum speed
        
            #We calculate the fluxes on each of the points.
            #Calculamos los flujos en los puntos
            Fx!(F_Pm3, um3, par_eq)
            Fx!(F_Pm2, um2, par_eq)
            Fx!(F_Pm1, um1, par_eq)
            Fx!(F_P, u, par_eq)
            Fx!(F_Pp1, up1, par_eq)
            Fx!(F_Pp2, up2, par_eq)
            Fx!(F_Pp3, up3, par_eq)
        
            #We make the Lax-Friedrichs Flux-Splitting
            #Hacemos la separación de flujos de Lax-Friedrichs
            @. F_Mm3 = 0.5 * (F_Pm3 - S_MAX * um3)
            @. F_Mm2 = 0.5 * (F_Pm2 - S_MAX * um2)
            @. F_Mm1 = 0.5 * (F_Pm1 - S_MAX * um1)
            @. F_M   = 0.5 * (F_P   - S_MAX * u)
            @. F_Mp1 = 0.5 * (F_Pp1 - S_MAX * up1)
            @. F_Mp2 = 0.5 * (F_Pp2 - S_MAX * up2)
            @. F_Mp3 = 0.5 * (F_Pp3 - S_MAX * up3)
            @. F_Pm3 = 0.5 * (F_Pm3 + S_MAX * um3)
            @. F_Pm2 = 0.5 * (F_Pm2 + S_MAX * um2)
            @. F_Pm1 = 0.5 * (F_Pm1 + S_MAX * um1)
            @. F_P   = 0.5 * (F_P   + S_MAX * u)
            @. F_Pp1 = 0.5 * (F_Pp1 + S_MAX * up1)
            @. F_Pp2 = 0.5 * (F_Pp2 + S_MAX * up2)
            @. F_Pp3 = 0.5 * (F_Pp3 + S_MAX * up3)
            #We reconstruct the fluxes in i-1/2 and i+1/2 for the split fluxes
            #Hacemos la reconstrucción de flujos en i-1/2 y i+1/2 para los dos flujos separados
            WENOZreconstruction!(F_RM, F_Mp2, F_Mp1,  F_M,  F_Mm1, F_Mm2, N_Fields)
            WENOZreconstruction!(F_LM, F_Pm3, F_Pm2, F_Pm1, F_P,  F_Pp1, N_Fields)
            WENOZreconstruction!(F_LP, F_Pm2, F_Pm1,  F_P,  F_Pp1, F_Pp2, N_Fields)
            WENOZreconstruction!(F_RP, F_Mp3, F_Mp2, F_Mp1, F_M,  F_Mm1, N_Fields)
        
            #We add the reconstructed split fluxes.
            @. H_p = F_LP + F_RP
            @. H_m = F_LM + F_RM

        
            #We calculate the temporal derivatives
            #Calculamos la derivada temporal
            @. dfields[:,idx] = -h*(H_p - H_m)
        end
    elseif length(J) == 2
        (Fx!,Fy!) = F!
        h = hh[1]
        N = J[1]
        for idy in 1:J[2]
            for  idx in 1:J[1]
                #first we defined shifted indices. The mod function make the indices periodic.
                #Primero definimos los indices desplazados. La función mod hace que los índices sean periódicos.
                idxm3 = mod(((idx-3) - 1),N) + 1
                idxm2 = mod(((idx-2) - 1),N) + 1
                idxm1 = mod(((idx-1) - 1),N) + 1
                idxp1 = mod(((idx+1) - 1),N) + 1
                idxp2 = mod(((idx+2) - 1),N) + 1
                idxp3 = mod(((idx+3) - 1),N) + 1
        
        
                #We give a name to the fields on the stencil points (idx-3, idx-2, idx-1, idx, idx+1, idx+2 and idx+3)
                #Le damos un nombre a los campos en los puntos del stencil (idx-3, idx-2, idx-1, idx, idx+1, idx+2 y idx+3)
                um3 = @view fields[:,idxm3,idy]
                um2 = @view fields[:,idxm2,idy]
                um1 = @view fields[:,idxm1,idy]
                u   = @view fields[:,idx,idy]
                up1 = @view fields[:,idxp1,idy]
                up2 = @view fields[:,idxp2,idy]
                up3 = @view fields[:,idxp3,idy]
        
                #We calculate the maximum propagation speed inside the stencil
                #Calculamos la velocidad de propagación máxima dentro del stencil
                S_MAX = max(Speed_max(up3, par_eq), Speed_max(um3, par_eq), 
                    Speed_max(up2, par_eq), Speed_max(um2, par_eq), Speed_max(up1, par_eq), 
                    Speed_max(um1, par_eq), Speed_max(u, par_eq)) #maximum speed
        
                #We calculate the fluxes on each of the points.
                #Calculamos los flujos en los puntos
                Fx!(F_Pm3, um3, par_eq)
                Fx!(F_Pm2, um2, par_eq)
                Fx!(F_Pm1, um1, par_eq)
                Fx!(F_P, u, par_eq)
                Fx!(F_Pp1, up1, par_eq)
                Fx!(F_Pp2, up2, par_eq)
                Fx!(F_Pp3, up3, par_eq)
        
                #We make the Lax-Friedrichs Flux-Splitting
                #Hacemos la separación de flujos de Lax-Friedrichs
                @. F_Mm3 = 0.5 * (F_Pm3 - S_MAX * um3)
                @. F_Mm2 = 0.5 * (F_Pm2 - S_MAX * um2)
                @. F_Mm1 = 0.5 * (F_Pm1 - S_MAX * um1)
                @. F_M   = 0.5 * (F_P   - S_MAX * u)
                @. F_Mp1 = 0.5 * (F_Pp1 - S_MAX * up1)
                @. F_Mp2 = 0.5 * (F_Pp2 - S_MAX * up2)
                @. F_Mp3 = 0.5 * (F_Pp3 - S_MAX * up3)
                @. F_Pm3 = 0.5 * (F_Pm3 + S_MAX * um3)
                @. F_Pm2 = 0.5 * (F_Pm2 + S_MAX * um2)
                @. F_Pm1 = 0.5 * (F_Pm1 + S_MAX * um1)
                @. F_P   = 0.5 * (F_P   + S_MAX * u)
                @. F_Pp1 = 0.5 * (F_Pp1 + S_MAX * up1)
                @. F_Pp2 = 0.5 * (F_Pp2 + S_MAX * up2)
                @. F_Pp3 = 0.5 * (F_Pp3 + S_MAX * up3)
                #We reconstruct the fluxes in i-1/2 and i+1/2 for the split fluxes
                #Hacemos la reconstrucción de flujos en i-1/2 y i+1/2 para los dos flujos separados
                WENOZreconstruction!(F_RM, F_Mp2, F_Mp1,  F_M,  F_Mm1, F_Mm2, N_Fields)
                WENOZreconstruction!(F_LM, F_Pm3, F_Pm2, F_Pm1, F_P,  F_Pp1, N_Fields)
                WENOZreconstruction!(F_LP, F_Pm2, F_Pm1,  F_P,  F_Pp1, F_Pp2, N_Fields)
                WENOZreconstruction!(F_RP, F_Mp3, F_Mp2, F_Mp1, F_M,  F_Mm1, N_Fields)
        
                #We add the reconstructed split fluxes.
                @. H_p = F_LP + F_RP
                @. H_m = F_LM + F_RM

        
                #We calculate the temporal derivatives
                #Calculamos la derivada temporal
                @. dfields[:,idx,idy] = -h*(H_p - H_m)
                
                #println("$(dfields[:,idx,idy])")
        end
    end

    h = hh[2]
    N = J[2]
    for idx in 1:J[1]
        for idy in 1:J[2]
            #first we defined shifted indices. The mod function make the indices periodic.
                #Primero definimos los indices desplazados. La función mod hace que los índices sean periódicos.
                idxm3 = mod(((idy-3) - 1),N) + 1
                idxm2 = mod(((idy-2) - 1),N) + 1
                idxm1 = mod(((idy-1) - 1),N) + 1
                idxp1 = mod(((idy+1) - 1),N) + 1
                idxp2 = mod(((idy+2) - 1),N) + 1
                idxp3 = mod(((idy+3) - 1),N) + 1
        
        
                #We give a name to the fields on the stencil points (idx-3, idx-2, idx-1, idx, idx+1, idx+2 and idx+3)
                #Le damos un nombre a los campos en los puntos del stencil (idx-3, idx-2, idx-1, idx, idx+1, idx+2 y idx+3)
                um3 = @view fields[:,idx,idxm3]
                um2 = @view fields[:,idx,idxm2]
                um1 = @view fields[:,idx,idxm1]
                u   = @view fields[:,idx,idy]
                up1 = @view fields[:,idx,idxp1]
                up2 = @view fields[:,idx,idxp2]
                up3 = @view fields[:,idx,idxp3]
        
                #We calculate the maximum propagation speed inside the stencil
                #Calculamos la velocidad de propagación máxima dentro del stencil
                S_MAX = max(Speed_max(up3, par_eq), Speed_max(um3, par_eq), 
                    Speed_max(up2, par_eq), Speed_max(um2, par_eq), Speed_max(up1, par_eq), 
                    Speed_max(um1, par_eq), Speed_max(u, par_eq)) #maximum speed
        
                #We calculate the fluxes on each of the points.
                #Calculamos los flujos en los puntos
                Fy!(F_Pm3, um3, par_eq)
                Fy!(F_Pm2, um2, par_eq)
                Fy!(F_Pm1, um1, par_eq)
                Fy!(F_P, u, par_eq)
                Fy!(F_Pp1, up1, par_eq)
                Fy!(F_Pp2, up2, par_eq)
                Fy!(F_Pp3, up3, par_eq)
        
                #We make the Lax-Friedrichs Flux-Splitting
                #Hacemos la separación de flujos de Lax-Friedrichs
                @. F_Mm3 = 0.5 * (F_Pm3 - S_MAX * um3)
                @. F_Mm2 = 0.5 * (F_Pm2 - S_MAX * um2)
                @. F_Mm1 = 0.5 * (F_Pm1 - S_MAX * um1)
                @. F_M   = 0.5 * (F_P   - S_MAX * u)
                @. F_Mp1 = 0.5 * (F_Pp1 - S_MAX * up1)
                @. F_Mp2 = 0.5 * (F_Pp2 - S_MAX * up2)
                @. F_Mp3 = 0.5 * (F_Pp3 - S_MAX * up3)
                @. F_Pm3 = 0.5 * (F_Pm3 + S_MAX * um3)
                @. F_Pm2 = 0.5 * (F_Pm2 + S_MAX * um2)
                @. F_Pm1 = 0.5 * (F_Pm1 + S_MAX * um1)
                @. F_P   = 0.5 * (F_P   + S_MAX * u)
                @. F_Pp1 = 0.5 * (F_Pp1 + S_MAX * up1)
                @. F_Pp2 = 0.5 * (F_Pp2 + S_MAX * up2)
                @. F_Pp3 = 0.5 * (F_Pp3 + S_MAX * up3)
                #We reconstruct the fluxes in i-1/2 and i+1/2 for the split fluxes
                #Hacemos la reconstrucción de flujos en i-1/2 y i+1/2 para los dos flujos separados
                WENOZreconstruction!(F_RM, F_Mp2, F_Mp1,  F_M,  F_Mm1, F_Mm2, N_Fields)
                WENOZreconstruction!(F_LM, F_Pm3, F_Pm2, F_Pm1, F_P,  F_Pp1, N_Fields)
                WENOZreconstruction!(F_LP, F_Pm2, F_Pm1,  F_P,  F_Pp1, F_Pp2, N_Fields)
                WENOZreconstruction!(F_RP, F_Mp3, F_Mp2, F_Mp1, F_M,  F_Mm1, N_Fields)
        
                #We add the reconstructed split fluxes.
                @. H_p = F_LP + F_RP
                @. H_m = F_LM + F_RM

        
                #We calculate the temporal derivatives
                #Calculamos la derivada temporal
                @. dfields[:,idx,idy] += -h*(H_p - H_m)
                #println("$(dfields[:,idx,idy])")
                #@. dfields[:,idx,idy] += -sourcevec[:,idx,idy]
        end
    end
        
    else 
        error("WENOZ only implemented for 1D and 2D")
    end

end



function wenoz_d_u!(du_fields, u_fields, par, t) # j is the grid position
    #asumimos u unidimensional por ahora
    #en esta version primero están los campos y luego los puntos del espacio.
    par_eq, hh, J, N_Fields, F!, Speed_max, auxvecs = par
    u_Mm3, u_Mm2, u_Mm1, u_M, u_Mp1, u_Mp2, u_Mp3, u_Pm3, u_Pm2, u_Pm1, u_P, u_Pp1, u_Pp2, u_Pp3, F_LP, F_LM, F_RP, F_RM, H_m, H_p, sourcevec = auxvecs
    
    fields = reshape(u_fields, N_Fields, J...)
    dfields = reshape(du_fields, N_Fields, J...)


    #nota: f minuscula o u se usa para hablar de campos, F mayúscula para hablar de Flujos.
    if length(J) == 1
        Fx! = F!
        N = J[1]
        h = hh
        for idx in 1:N
        
            #first we defined shifted indices. The mod function make the indices periodic.
            #Primero definimos los indices desplazados. La función mod hace que los índices sean periódicos.
            idxm3 = mod(((idx-3) - 1),N) + 1
            idxm2 = mod(((idx-2) - 1),N) + 1
            idxm1 = mod(((idx-1) - 1),N) + 1
            idxp1 = mod(((idx+1) - 1),N) + 1
            idxp2 = mod(((idx+2) - 1),N) + 1
            idxp3 = mod(((idx+3) - 1),N) + 1
        
        
            #We give a name to the fields on the stencil points (idx-3, idx-2, idx-1, idx, idx+1, idx+2 and idx+3)
            #Le damos un nombre a los campos en los puntos del stencil (idx-3, idx-2, idx-1, idx, idx+1, idx+2 y idx+3)
            um3 = @view fields[:,idxm3]
            um2 = @view fields[:,idxm2]
            um1 = @view fields[:,idxm1]
            u   = @view fields[:,idx]
            up1 = @view fields[:,idxp1]
            up2 = @view fields[:,idxp2]
            up3 = @view fields[:,idxp3]
        
            #We calculate the maximum propagation speed inside the stencil
            #Calculamos la velocidad de propagación máxima dentro del stencil
            S_MAX = max(Speed_max(up3, par_eq), Speed_max(um3, par_eq), 
                Speed_max(up2, par_eq), Speed_max(um2, par_eq), Speed_max(up1, par_eq), 
                Speed_max(um1, par_eq), Speed_max(u, par_eq)) #maximum speed
        
            #We calculate the fluxes on each of the points.
            #Calculamos los flujos en los puntos
            Fx!(F_Pm3, um3, par_eq)
            Fx!(F_Pm2, um2, par_eq)
            Fx!(F_Pm1, um1, par_eq)
            Fx!(F_P, u, par_eq)
            Fx!(F_Pp1, up1, par_eq)
            Fx!(F_Pp2, up2, par_eq)
            Fx!(F_Pp3, up3, par_eq)
        
            #We make the Lax-Friedrichs Flux-Splitting
            #Hacemos la separación de flujos de Lax-Friedrichs
            @. F_Mm3 = 0.5 * (F_Pm3 - S_MAX * um3)
            @. F_Mm2 = 0.5 * (F_Pm2 - S_MAX * um2)
            @. F_Mm1 = 0.5 * (F_Pm1 - S_MAX * um1)
            @. F_M   = 0.5 * (F_P   - S_MAX * u)
            @. F_Mp1 = 0.5 * (F_Pp1 - S_MAX * up1)
            @. F_Mp2 = 0.5 * (F_Pp2 - S_MAX * up2)
            @. F_Mp3 = 0.5 * (F_Pp3 - S_MAX * up3)
            @. F_Pm3 = 0.5 * (F_Pm3 + S_MAX * um3)
            @. F_Pm2 = 0.5 * (F_Pm2 + S_MAX * um2)
            @. F_Pm1 = 0.5 * (F_Pm1 + S_MAX * um1)
            @. F_P   = 0.5 * (F_P   + S_MAX * u)
            @. F_Pp1 = 0.5 * (F_Pp1 + S_MAX * up1)
            @. F_Pp2 = 0.5 * (F_Pp2 + S_MAX * up2)
            @. F_Pp3 = 0.5 * (F_Pp3 + S_MAX * up3)
            #We reconstruct the fluxes in i-1/2 and i+1/2 for the split fluxes
            #Hacemos la reconstrucción de flujos en i-1/2 y i+1/2 para los dos flujos separados
            WENOZreconstruction!(F_RM, F_Mp2, F_Mp1,  F_M,  F_Mm1, F_Mm2, N_Fields)
            WENOZreconstruction!(F_LM, F_Pm3, F_Pm2, F_Pm1, F_P,  F_Pp1, N_Fields)
            WENOZreconstruction!(F_LP, F_Pm2, F_Pm1,  F_P,  F_Pp1, F_Pp2, N_Fields)
            WENOZreconstruction!(F_RP, F_Mp3, F_Mp2, F_Mp1, F_M,  F_Mm1, N_Fields)
        
            #We add the reconstructed split fluxes.
            @. H_p = F_LP + F_RP
            @. H_m = F_LM + F_RM

        
            #We calculate the temporal derivatives
            #Calculamos la derivada temporal
            @. dfields[:,idx] = -h*(H_p - H_m)
        end
    elseif length(J) == 2
        (Fx!,Fy!) = F!
        h = hh[1]
        N = J[1]
        for idy in 1:J[2]
            for  idx in 1:J[1]
                #first we defined shifted indices. The mod function make the indices periodic.
                #Primero definimos los indices desplazados. La función mod hace que los índices sean periódicos.
                idxm3 = mod(((idx-3) - 1),N) + 1
                idxm2 = mod(((idx-2) - 1),N) + 1
                idxm1 = mod(((idx-1) - 1),N) + 1
                idxp1 = mod(((idx+1) - 1),N) + 1
                idxp2 = mod(((idx+2) - 1),N) + 1
                idxp3 = mod(((idx+3) - 1),N) + 1
        
        
                #We give a name to the fields on the stencil points (idx-3, idx-2, idx-1, idx, idx+1, idx+2 and idx+3)
                #Le damos un nombre a los campos en los puntos del stencil (idx-3, idx-2, idx-1, idx, idx+1, idx+2 y idx+3)
                um3 = @view fields[:,idxm3,idy]
                um2 = @view fields[:,idxm2,idy]
                um1 = @view fields[:,idxm1,idy]
                u   = @view fields[:,idx,idy]
                up1 = @view fields[:,idxp1,idy]
                up2 = @view fields[:,idxp2,idy]
                up3 = @view fields[:,idxp3,idy]
        
                #We calculate the maximum propagation speed inside the stencil
                #Calculamos la velocidad de propagación máxima dentro del stencil
                S_MAX = max(Speed_max(up3, par_eq), Speed_max(um3, par_eq), 
                    Speed_max(up2, par_eq), Speed_max(um2, par_eq), Speed_max(up1, par_eq), 
                    Speed_max(um1, par_eq), Speed_max(u, par_eq)) #maximum speed
        
                #We calculate the fluxes on each of the points.
                #Calculamos los flujos en los puntos
                Fx!(F_Pm3, um3, par_eq)
                Fx!(F_Pm2, um2, par_eq)
                Fx!(F_Pm1, um1, par_eq)
                Fx!(F_P, u, par_eq)
                Fx!(F_Pp1, up1, par_eq)
                Fx!(F_Pp2, up2, par_eq)
                Fx!(F_Pp3, up3, par_eq)
        
                #We make the Lax-Friedrichs Flux-Splitting
                #Hacemos la separación de flujos de Lax-Friedrichs
                @. F_Mm3 = 0.5 * (F_Pm3 - S_MAX * um3)
                @. F_Mm2 = 0.5 * (F_Pm2 - S_MAX * um2)
                @. F_Mm1 = 0.5 * (F_Pm1 - S_MAX * um1)
                @. F_M   = 0.5 * (F_P   - S_MAX * u)
                @. F_Mp1 = 0.5 * (F_Pp1 - S_MAX * up1)
                @. F_Mp2 = 0.5 * (F_Pp2 - S_MAX * up2)
                @. F_Mp3 = 0.5 * (F_Pp3 - S_MAX * up3)
                @. F_Pm3 = 0.5 * (F_Pm3 + S_MAX * um3)
                @. F_Pm2 = 0.5 * (F_Pm2 + S_MAX * um2)
                @. F_Pm1 = 0.5 * (F_Pm1 + S_MAX * um1)
                @. F_P   = 0.5 * (F_P   + S_MAX * u)
                @. F_Pp1 = 0.5 * (F_Pp1 + S_MAX * up1)
                @. F_Pp2 = 0.5 * (F_Pp2 + S_MAX * up2)
                @. F_Pp3 = 0.5 * (F_Pp3 + S_MAX * up3)
                #We reconstruct the fluxes in i-1/2 and i+1/2 for the split fluxes
                #Hacemos la reconstrucción de flujos en i-1/2 y i+1/2 para los dos flujos separados
                WENOZreconstruction!(F_RM, F_Mp2, F_Mp1,  F_M,  F_Mm1, F_Mm2, N_Fields)
                WENOZreconstruction!(F_LM, F_Pm3, F_Pm2, F_Pm1, F_P,  F_Pp1, N_Fields)
                WENOZreconstruction!(F_LP, F_Pm2, F_Pm1,  F_P,  F_Pp1, F_Pp2, N_Fields)
                WENOZreconstruction!(F_RP, F_Mp3, F_Mp2, F_Mp1, F_M,  F_Mm1, N_Fields)
        
                #We add the reconstructed split fluxes.
                @. H_p = F_LP + F_RP
                @. H_m = F_LM + F_RM

        
                #We calculate the temporal derivatives
                #Calculamos la derivada temporal
                @. dfields[:,idx,idy] = -h*(H_p - H_m)
                
                #println("$(dfields[:,idx,idy])")
        end
    end

    h = hh[2]
    N = J[2]
    for idx in 1:J[1]
        for idy in 1:J[2]
            #first we defined shifted indices. The mod function make the indices periodic.
                #Primero definimos los indices desplazados. La función mod hace que los índices sean periódicos.
                idxm3 = mod(((idy-3) - 1),N) + 1
                idxm2 = mod(((idy-2) - 1),N) + 1
                idxm1 = mod(((idy-1) - 1),N) + 1
                idxp1 = mod(((idy+1) - 1),N) + 1
                idxp2 = mod(((idy+2) - 1),N) + 1
                idxp3 = mod(((idy+3) - 1),N) + 1
        
        
                #We give a name to the fields on the stencil points (idx-3, idx-2, idx-1, idx, idx+1, idx+2 and idx+3)
                #Le damos un nombre a los campos en los puntos del stencil (idx-3, idx-2, idx-1, idx, idx+1, idx+2 y idx+3)
                um3 = @view fields[:,idx,idxm3]
                um2 = @view fields[:,idx,idxm2]
                um1 = @view fields[:,idx,idxm1]
                u   = @view fields[:,idx,idy]
                up1 = @view fields[:,idx,idxp1]
                up2 = @view fields[:,idx,idxp2]
                up3 = @view fields[:,idx,idxp3]
        
                #We calculate the maximum propagation speed inside the stencil
                #Calculamos la velocidad de propagación máxima dentro del stencil
                S_MAX = max(Speed_max(up3, par_eq), Speed_max(um3, par_eq), 
                    Speed_max(up2, par_eq), Speed_max(um2, par_eq), Speed_max(up1, par_eq), 
                    Speed_max(um1, par_eq), Speed_max(u, par_eq)) #maximum speed
        
                #We calculate the fluxes on each of the points.
                #Calculamos los flujos en los puntos
                Fy!(F_Pm3, um3, par_eq)
                Fy!(F_Pm2, um2, par_eq)
                Fy!(F_Pm1, um1, par_eq)
                Fy!(F_P, u, par_eq)
                Fy!(F_Pp1, up1, par_eq)
                Fy!(F_Pp2, up2, par_eq)
                Fy!(F_Pp3, up3, par_eq)
        
                #We make the Lax-Friedrichs Flux-Splitting
                #Hacemos la separación de flujos de Lax-Friedrichs
                @. F_Mm3 = 0.5 * (F_Pm3 - S_MAX * um3)
                @. F_Mm2 = 0.5 * (F_Pm2 - S_MAX * um2)
                @. F_Mm1 = 0.5 * (F_Pm1 - S_MAX * um1)
                @. F_M   = 0.5 * (F_P   - S_MAX * u)
                @. F_Mp1 = 0.5 * (F_Pp1 - S_MAX * up1)
                @. F_Mp2 = 0.5 * (F_Pp2 - S_MAX * up2)
                @. F_Mp3 = 0.5 * (F_Pp3 - S_MAX * up3)
                @. F_Pm3 = 0.5 * (F_Pm3 + S_MAX * um3)
                @. F_Pm2 = 0.5 * (F_Pm2 + S_MAX * um2)
                @. F_Pm1 = 0.5 * (F_Pm1 + S_MAX * um1)
                @. F_P   = 0.5 * (F_P   + S_MAX * u)
                @. F_Pp1 = 0.5 * (F_Pp1 + S_MAX * up1)
                @. F_Pp2 = 0.5 * (F_Pp2 + S_MAX * up2)
                @. F_Pp3 = 0.5 * (F_Pp3 + S_MAX * up3)
                #We reconstruct the fluxes in i-1/2 and i+1/2 for the split fluxes
                #Hacemos la reconstrucción de flujos en i-1/2 y i+1/2 para los dos flujos separados
                WENOZreconstruction!(F_RM, F_Mp2, F_Mp1,  F_M,  F_Mm1, F_Mm2, N_Fields)
                WENOZreconstruction!(F_LM, F_Pm3, F_Pm2, F_Pm1, F_P,  F_Pp1, N_Fields)
                WENOZreconstruction!(F_LP, F_Pm2, F_Pm1,  F_P,  F_Pp1, F_Pp2, N_Fields)
                WENOZreconstruction!(F_RP, F_Mp3, F_Mp2, F_Mp1, F_M,  F_Mm1, N_Fields)
        
                #We add the reconstructed split fluxes.
                @. H_p = F_LP + F_RP
                @. H_m = F_LM + F_RM

        
                #We calculate the temporal derivatives
                #Calculamos la derivada temporal
                @. dfields[:,idx,idy] += -h*(H_p - H_m)
                #println("$(dfields[:,idx,idy])")
                #@. dfields[:,idx,idy] += -sourcevec[:,idx,idy]
        end
    end
        
    else 
        error("WENOZ only implemented for 1D and 2D")
    end

end

function wenoz(u,t,par) #acomodada para el RK$_Step o TVD3_Step
    du, parWENOZ = par
    wenoz!(du,u,parWENOZ,t)
    return du
end

#==================== WENO-Z FV (cell averages -> interfaces) ====================#

"""
WENO-Z finite-volume reconstruction from cell averages to interfaces.

Computes values and derivatives at interfaces using the reconstructed quadratic
polynomials directly. Single pass produces all four arrays with periodic BCs.

Returns (uL, uR, duL, duR) with:
- uL[i]: left-biased value at interface i+1/2 (from cell i)
- uR[i]: right-biased value at interface i+1/2 (from cell i+1) via shift
- duL[i], duR[i]: corresponding derivatives at the interface
"""
function WENOZ_FV_reconstruct_from_averages_Per(uavg::AbstractVector{<:Real}, dx::Real)
    n = length(uavg)
    u = collect(Float64, uavg)
    uL = zeros(n)
    uR = zeros(n)
    duL = zeros(n)
    duR = zeros(n)

    periodic_index(i) = (mod(i-1, n) + 1)

    # Precompute inverse matrices for three left-biased stencils (S0,S1,S2

    # Interface at x=0 (i+1/2). Correct cell intervals for left-biased stencils:
    # S0: {i-2,i-1,i}   → [-3dx,-2dx], [-2dx,-dx], [-dx,0]   (a0 = -3dx)
    # S1: {i-1,i,i+1}   → [-2dx,-dx],  [-dx,0],   [0,dx]     (a0 = -2dx)
    # S2: {i,i+1,i+2}   → [-dx,0],     [0,dx],    [dx,2dx]   (a0 = -dx)
    invA0 = invA_for_stencil(-3dx, dx)
    invA1 = invA_for_stencil(-2dx, dx)
    invA2 = invA_for_stencil(-1dx, dx)

    B1 = 13.0/12.0
    eps = 1e-40

    for i in 1:n
        im2 = periodic_index(i-2)
        im1 = periodic_index(i-1)
        i0  = periodic_index(i)
        ip1 = periodic_index(i+1)
        ip2 = periodic_index(i+2)

        # Smoothness indicators for left-biased stencils
        β0 =  B1*(u[i0] - 2*u[ip1] + u[ip2])^2 + 0.25*(3*u[i0] - 4*u[ip1] + u[ip2])^2
        β1 =  B1*(u[im1] - 2*u[i0] + u[ip1])^2 + 0.25*(u[im1] - u[ip1])^2
        β2 =  B1*(u[im2] - 2*u[im1] + u[i0])^2 + 0.25*(u[im2] - 4*u[im1] + 3*u[i0])^2
        τ5 = abs(β2 - β0)
        α0 = 0.1*(1.0 + (τ5/(β0 + eps))^2)
        α1 = 0.6*(1.0 + (τ5/(β1 + eps))^2)
        α2 = 0.3*(1.0 + (τ5/(β2 + eps))^2)
        αs = (α0 + α1 + α2)
        ω0 = α0/αs; ω1 = α1/αs; ω2 = α2/αs

        # Build candidate polynomials p_k(x) = a x^2 + b x + c matching cell averages
        # S0: cells {i-2,i-1,i} over intervals [-3dx,-2dx],[-2dx,-dx],[-dx,0]
        y0 = SVector(u[im2], u[im1], u[i0])
        coeff0 = invA0 * y0
        # S1: cells {i-1,i,i+1} over [-2dx,-dx],[-dx,0],[0,dx]
        y1 = SVector(u[im1], u[i0], u[ip1])
        coeff1 = invA1 * y1
        # S2: cells {i,i+1,i+2} over [-dx,0],[0,dx],[dx,2dx]
        y2 = SVector(u[i0], u[ip1], u[ip2])
        coeff2 = invA2 * y2

        # Interface value p_k(0) = c_k; interface derivative p'_k(0) = b_k
        c0 = coeff0[3]; b0 = coeff0[2]
        c1 = coeff1[3]; b1 = coeff1[2]
        c2 = coeff2[3]; b2 = coeff2[2]

        uL[i]  = ω0*c0 + ω1*c1 + ω2*c2
        duL[i] = ω0*b0 + ω1*b1 + ω2*b2
    end

    # Right states and right derivatives by periodic shift of left ones
    for i in 1:n
        uR[i]  = uL[periodic_index(i+1)]
        duR[i] = duL[periodic_index(i+1)]
    end

    return uL, uR, duL, duR
end


"""
WENO-Z finite-volume reconstruction from cell averages to interfaces.

Computes values and derivatives at interfaces using the reconstructed quadratic
polynomials directly. Single pass produces all four arrays with ghost zones.

Returns (uL, uR, duL, duR) with:
- uL[i]: left-biased value at interface i+1/2 
- uR[i]: right-biased value at interface i+1/2 
- duL[i], duR[i]: corresponding derivatives at the interface
"""
function WENOZ_FV_reconstruct_from_averages(uavg, dx)
    n = length(uavg)-5  # assuming 3 ghost cells on each side. This n is n+1 counting real averages.
    u = collect(Float64, uavg)
    uL = zeros(n)
    uR = zeros(n)
    duL = zeros(n)
    duR = zeros(n)

    pmm = [1/3, -7/6, 11/6]  # coefficients for 3rd-order extrapolation
    p0m = [-1/6, 5/6, 1/3]
    ppm = [1/3, 5/6, -1/6]
    pmp = p0m
    p0p = ppm
    ppp = [11/6, -7/6, 1/3]
    dpmm = [1/2, -2, 3/2] # coefficients for derivative extrapolation
    dp0m = [-1/2, 0, 1/2]
    dppm = [-3/2, 2, -1/2]
    dpmp = dp0m
    dp0p = dppm
    dppp = dp0m
    B1 = 13.0/12.0
    eps = 1e-40

    for i in 3:n+2
        im2 = i-2
        im1 = i-1
        i0  = i
        ip1 = i+1
        ip2 = i+2
        ip3 = i+3

        # Build candidate polynomials p_k(x) = a x^2 + b x + c matching cell averages
        # S0: cells {i-2,i-1,i} over intervals [-3dx,-2dx],[-2dx,-dx],[-dx,0]
        ym = u[i-2:i]
        coeffm = pmm' * ym
        d_coeffm = dpmm' * ym
        # S1: cells {i-1,i,i+1} over [-2dx,-dx],[-dx,0],[0,dx]
        y0 = u[i-1:i+1]
        coeff0 = p0m' * y0
        d_coeff1 = dp0m' * y0
        # S2: cells {i,i+1,i+2} over [-dx,0],[0,dx],[dx,2dx]
        yp = u[i:i+2]
        coeffp = ppm' * yp
        d_coeffp = dppm' * yp

        # Smoothness indicators for left-biased stencils
        βp1 =  B1*(u[i0 ] - 2*u[ip1] + u[ip2])^2 + 0.25*(3*u[i0] - 4*u[ip1] + u[ip2])^2
        β0 =   B1*(u[im1] - 2*u[i0 ] + u[ip1])^2 + 0.25*(u[im1] - u[ip1])^2
        βm1 =  B1*(u[im2] - 2*u[im1] + u[i0 ])^2 + 0.25*(u[im2] - 4*u[im1] + 3*u[i0])^2
        ωm, ω0, ωp = get_omegas(βm1, β0, βp1, eps)

        uL[i-2]  = ωm*coeffm + ω0*coeff0 + ωp*coeffp
        duL[i-2] = (ωm*d_coeffm + ω0*d_coeff1 + ωp*d_coeffp)/dx
        

        # S0: cells {i+1,i+2,i+3} over intervals [0,dx],[dx,2dx],[2dx,3dx]
        ym = y0 
        coeffm = pmp' * ym
        d_coeff0 = dpmp' * ym
        # S1: cells {i,i+1,i+2} over [0,dx],[dx,2dx],[2dx,3dx]
        y0 = yp 
        coeff0 = p0p' * y0
        d_coeff0 = dp0p' * y0
        # S2: cells {i-1,i,i+1} over [-dx,0],[0,dx],[dx,2dx]
        yp = u[i+1:i+3]
        coeffp = ppp' * yp
        d_coeffp = dppp' * yp

    
        #βp1 =  B1*(u[i0] - 2*u[ip1] + u[ip2])^2 + 0.25*(3u[i0] - 4u[ip1] + u[ip2])^2
        βp2 =  B1*(u[ip1] - 2*u[ip2] + u[ip3])^2 + 0.25*(u[ip1] - 4*u[ip2] + 3*u[ip3])^2
        #ω0, ω1, ω2 = get_omegas(β0, βp1, βp2, eps)
        ωp, ω0, ωm = get_omegas(βp2, βp1, β0, eps)
        uR[i-2]  = ωm*coeffm + ω0*coeff0 + ωp*coeffp #ω0*c0 + ω1*c1 + ω2*c2
        duR[i-2] = (ωm*d_coeffm + ω0*d_coeff0 + ωp*d_coeffp)/dx

    end

    return uL, uR, duL, duR
end

function get_omegas(β0, β1, β2,eps=1e-40)
            # #=
            τ5 = 0.0 #abs(β2 - β0)
            α0 = 0.1*(1.0 + (τ5/(β0 + eps))^2)
            α1 = 0.6*(1.0 + (τ5/(β1 + eps))^2)
            α2 = 0.3*(1.0 + (τ5/(β2 + eps))^2)
            # =#
            #=
            α0 = 0.1*(1.0/(β0 + eps)^2)
            α1 = 0.6*(1.0/(β1 + eps)^2)
            α2 = 0.3*(1.0/(β2 + eps)^2)
            =#
            αs = (α0 + α1 + α2)
            ω0 = α0/αs; ω1 = α1/αs; ω2 = α2/αs
            return ω0, ω1, ω2
end



function WENOZ_FV_reconstruct_from_averages!(uL, uR, duL, duR, uavg::AbstractVector{<:Real}, dx::Real)
    ul, ur, dul, dur = WENOZ_FV_reconstruct_from_averages(uavg, dx)
    uL .= ul
    uR .= ur
    duL .= dul
    duR .= dur
    return nothing
end

"""
This is only to compute the derivatives at the interfaces from the left and right states,
when these have been computed with WENOZ_FV_reconstruct_from_averages or similar.
It is only good for boundaries which become constant. For periodic boundaries modify accordingly.
"""
function WENOZ_derivatives_form_uLR(uL, uR, dx)
    duL = similar(uL)
    duR = similar(uR)
    duL_ext = vcat(uL[1], uL[1], uL, uL[end], uL[end])
    duR_ext = vcat(uR[1], uR[1], uR, uR[end], uR[end])
    for i in eachindex(uL)
        duL[i] = -(-duL_ext[i] + 8*duL_ext[i+1] - 8*duL_ext[i+3] + duL_ext[i+4]) / (12*dx)
        duR[i] = -(-duR_ext[i] + 8*duR_ext[i+1] - 8*duR_ext[i+3] + duR_ext[i+4]) / (12*dx)
    end
    return duL, duR
end

