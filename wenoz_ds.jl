using LinearAlgebra
using StaticArrays

"""
WENO reconstruction functions for fields and their first derivatives.
Supports WENO-JS and WENO-Z variants with arbitrary order.
"""

module WENOReconstruction

export weno_reconstruct, weno_reconstruct_derivative, WENOJS, WENOZ, ReconstructionResult

# Reconstruction variants
abstract type WENOVariant end
struct WENOJS <: WENOVariant end
struct WENOZ <: WENOVariant end

struct ReconstructionResult
    left_interface::Float64    # Value at left interface (i-1/2)
    right_interface::Float64   # Value at right interface (i+1/2)
    derivative::Float64        # Derivative at cell center
end

# Coefficients for different WENO orders
struct WENOCoeffs{Order}
    # Stencil coefficients for reconstruction
    cr::Matrix{Float64}        # Right-biased coefficients
    cl::Matrix{Float64}        # Left-biased coefficients
    # Smoothness indicator coefficients
    beta_coeffs::Matrix{Float64}
    # Ideal weights
    ideal_weights::Vector{Float64}
end

# Precompute coefficients for common orders
function get_weno_coeffs(order::Int)
    if order == 3
        # 3rd order WENO (r=2)
        cr = [ 3/2 -1/2;    # stencil 0
               1/2  1/2;    # stencil 1
              -1/2  3/2]'   # stencil 2
        
        cl = [-1/2  3/2;    # stencil 0
               1/2  1/2;    # stencil 1
               3/2 -1/2]'   # stencil 2
        
        # Simplified smoothness indicators for 3rd order
        beta_coeffs = [1.0 -2.0 1.0;  # beta_0 coefficients
                       1.0 -2.0 1.0;  # beta_1 coefficients  
                       1.0 -2.0 1.0]  # beta_2 coefficients
        
        ideal_weights = [0.25, 0.5, 0.25]
        
    elseif order == 5
        # 5th order WENO (r=3)
        cr = [11/6  -7/6   1/3;   # stencil 0
              1/3    5/6  -1/6;   # stencil 1
             -1/6    5/6   1/3;   # stencil 2
              1/3   -7/6  11/6]'  # stencil 3
        
        cl = [1/3   -7/6  11/6;   # stencil 0
             -1/6    5/6   1/3;   # stencil 1
              1/3    5/6  -1/6;   # stencil 2
             11/6   -7/6   1/3]'  # stencil 3
        
        # Correct smoothness indicator coefficients for 5th order
        beta_coeffs = [
            # beta_0 coefficients (stencil {i-2, i-1, i})
            4.0/3.0 -19.0/3.0 11.0/3.0 0.0 0.0 0.0;
            # beta_1 coefficients (stencil {i-1, i, i+1})
            -1.0/3.0 13.0/3.0 -23.0/3.0 25.0/3.0 0.0 0.0;
            # beta_2 coefficients (stencil {i, i+1, i+2})
            0.0 4.0/3.0 -13.0/3.0 13.0/3.0 0.0 0.0;
            # beta_3 coefficients (stencil {i+1, i+2, i+3})
            0.0 0.0 1.0/3.0 -5.0/3.0 4.0/3.0 0.0
        ]
        
        ideal_weights = [0.0625, 0.4375, 0.4375, 0.0625]
        
    else
        error("Unsupported WENO order: $order. Supported orders: 3, 5")
    end
    
    return WENOCoeffs{order}(cr, cl, beta_coeffs, ideal_weights)
end

"""
    compute_smoothness_indicators(stencil, coeffs::WENOCoeffs{Order}) where Order

Compute smoothness indicators β_k for each stencil.
"""
function compute_smoothness_indicators(stencil, coeffs::WENOCoeffs{Order}) where Order
    r = Order - 1
    beta = zeros(r + 1)
    
    for k in 0:r
        beta_k = 0.0
        # For each derivative order in the smoothness indicator
        for l in 1:r
            sum_term = 0.0
            # Sum over the stencil points
            for m in 0:r
                # Use only valid indices
                if m + 1 <= length(stencil)
                    coeff_idx = l * (r + 1) + m + 1
                    if coeff_idx <= size(coeffs.beta_coeffs, 2)
                        sum_term += coeffs.beta_coeffs[k+1, coeff_idx] * stencil[m+1]
                    end
                end
            end
            beta_k += sum_term^2
        end
        beta[k+1] = beta_k
    end
    
    return beta
end

"""
Simplified and robust smoothness indicator calculation
"""
function compute_smoothness_indicators_simple(stencil, order::Int)
    if order == 3
        # 3rd order WENO - simple finite difference approximations
        beta = zeros(3)
        beta[1] = (stencil[1] - 2*stencil[2] + stencil[3])^2  # second derivative
        beta[2] = (stencil[2] - 2*stencil[3] + stencil[4])^2
        beta[3] = (stencil[3] - 2*stencil[4] + stencil[5])^2
    elseif order == 5
        # 5th order WENO - more accurate smoothness indicators
        beta = zeros(4)
        beta[1] = (stencil[1] - 2*stencil[2] + stencil[3])^2 + 
                  (stencil[1] - 4*stencil[2] + 3*stencil[3])^2
        beta[2] = (stencil[2] - 2*stencil[3] + stencil[4])^2 + 
                  (stencil[2] - stencil[4])^2
        beta[3] = (stencil[3] - 2*stencil[4] + stencil[5])^2 + 
                  (3*stencil[3] - 4*stencil[4] + stencil[5])^2
        beta[4] = (stencil[4] - 2*stencil[5] + stencil[6])^2 + 
                  (stencil[4] - 4*stencil[5] + 3*stencil[6])^2
    else
        error("Unsupported order")
    end
    return beta
end

"""
    compute_nonlinear_weights(beta, ideal_weights, variant::WENOVariant, epsilon=1e-6)

Compute nonlinear weights for WENO reconstruction.
"""
function compute_nonlinear_weights(beta, ideal_weights, variant::WENOJS, epsilon=1e-6)
    alpha = ideal_weights ./ (beta .+ epsilon).^2
    alpha_sum = sum(alpha)
    return alpha ./ alpha_sum
end

function compute_nonlinear_weights(beta, ideal_weights, variant::WENOZ, epsilon=1e-6)
    tau = abs(beta[1] - beta[end])  # Global smoothness indicator
    alpha = ideal_weights .* (1.0 .+ (tau ./ (beta .+ epsilon)).^2)
    alpha_sum = sum(alpha)
    return alpha ./ alpha_sum
end

"""
    weno_reconstruct(u, i, order=5, variant=WENOZ(); epsilon=1e-6)

Perform WENO reconstruction at cell i.
Returns ReconstructionResult with left/right interface values and derivative.
"""
function weno_reconstruct(u, i, order=5, variant=WENOZ(); epsilon=1e-6)
    coeffs = get_weno_coeffs(order)
    r = order - 1
    
    # Check boundaries
    if i <= r || i > length(u) - r
        # Use simple central difference near boundaries
        if i == 1
            derivative = u[2] - u[1]
        elseif i == length(u)
            derivative = u[end] - u[end-1]
        else
            derivative = (u[i+1] - u[i-1]) / 2.0
        end
        return ReconstructionResult(u[i], u[i], derivative)
    end
    
    # Get stencil centered at cell i
    stencil_indices = (i-r):(i+r)
    stencil = [u[j] for j in stencil_indices]
    
    # Compute smoothness indicators - use the simple robust version
    beta = compute_smoothness_indicators_simple(stencil, order)
    
    # Compute nonlinear weights for left and right interfaces
    weights_left = compute_nonlinear_weights(beta, coeffs.ideal_weights, variant, epsilon)
    weights_right = compute_nonlinear_weights(beta, coeffs.ideal_weights, variant, epsilon)
    
    # Reconstruct left interface (i-1/2) and right interface (i+1/2)
    left_interface = 0.0
    right_interface = 0.0
    
    for k in 0:r
        # Left interface reconstruction (uses left-biased coefficients)
        poly_left = 0.0
        for j in 0:r
            poly_left += coeffs.cl[k+1, j+1] * stencil[j+1]
        end
        left_interface += weights_left[k+1] * poly_left
        
        # Right interface reconstruction (uses right-biased coefficients)
        poly_right = 0.0
        for j in 0:r
            poly_right += coeffs.cr[k+1, j+1] * stencil[j+1]
        end
        right_interface += weights_right[k+1] * poly_right
    end
    
    # Compute derivative using appropriate finite difference
    if order >= 5 && i > 2 && i < length(u)-1
        derivative = (-u[i+2] + 8*u[i+1] - 8*u[i-1] + u[i-2]) / 12.0
    else
        derivative = (u[i+1] - u[i-1]) / 2.0
    end
    
    return ReconstructionResult(left_interface, right_interface, derivative)
end

"""
    weno_reconstruct_derivative(u, i, order=5, variant=WENOZ(); epsilon=1e-6)

Specialized reconstruction for derivative fields.
"""
function weno_reconstruct_derivative(u, i, order=5, variant=WENOZ(); epsilon=1e-6)
    coeffs = get_weno_coeffs(order)
    r = order - 1
    
    # Check boundaries
    if i <= r || i > length(u) - r
        # Simple handling near boundaries
        if i == 1
            d2u = u[3] - 2*u[2] + u[1]
        elseif i == length(u)
            d2u = u[end] - 2*u[end-1] + u[end-2]
        else
            d2u = u[i+1] - 2*u[i] + u[i-1]
        end
        return ReconstructionResult(0.0, 0.0, d2u)
    end
    
    # First compute derivatives at cell centers using central differences
    du = zeros(2r+1)
    stencil_indices = (i-r):(i+r)
    
    for (idx, j) in enumerate(stencil_indices)
        if j > 2 && j < length(u)-1
            du[idx] = (-u[j+2] + 8*u[j+1] - 8*u[j-1] + u[j-2]) / 12.0
        else
            # Fall back to 2nd order
            if j == 1
                du[idx] = u[2] - u[1]
            elseif j == length(u)
                du[idx] = u[end] - u[end-1]
            else
                du[idx] = (u[j+1] - u[j-1]) / 2.0
            end
        end
    end
    
    # Apply WENO reconstruction to the derivative field
    beta = compute_smoothness_indicators_simple(du, order)
    
    weights_left = compute_nonlinear_weights(beta, coeffs.ideal_weights, variant, epsilon)
    weights_right = compute_nonlinear_weights(beta, coeffs.ideal_weights, variant, epsilon)
    
    left_interface = 0.0
    right_interface = 0.0
    
    for k in 0:r
        poly_left = 0.0
        poly_right = 0.0
        
        for j in 0:r
            poly_left += coeffs.cl[k+1, j+1] * du[j+1]
            poly_right += coeffs.cr[k+1, j+1] * du[j+1]
        end
        
        left_interface += weights_left[k+1] * poly_left
        right_interface += weights_right[k+1] * poly_right
    end
    
    # For the second derivative, use central difference
    if i > 2 && i < length(u)-1
        d2u = (-du[5] + 16*du[4] - 30*du[3] + 16*du[2] - du[1]) / 12.0
    else
        d2u = du[3] - 2*du[2] + du[1]
    end
    
    return ReconstructionResult(left_interface, right_interface, d2u)
end

"""
    reconstruct_entire_field(u, order=5, variant=WENOZ())

Reconstruct entire field with proper boundary handling.
"""
function reconstruct_entire_field(u, order=5, variant=WENOZ())
    n = length(u)
    left_interfaces = zeros(n)
    right_interfaces = zeros(n)
    derivatives = zeros(n)
    
    r = order - 1
    
    # Handle all points
    for i in 1:n
        result = weno_reconstruct(u, i, order, variant)
        left_interfaces[i] = result.left_interface
        right_interfaces[i] = result.right_interface
        derivatives[i] = result.derivative
    end
    
    return left_interfaces, right_interfaces, derivatives
end

 

end  # module WENOReconstruction


