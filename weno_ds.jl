using Printf

struct WENOReconstructor
    nx::Int
    xmin::Float64
    xmax::Float64
    dx::Float64
    x::Vector{Float64}  # Cell centers
    xi::Vector{Float64} # Cell interfaces
    eps::Float64
    gamma::Vector{Float64}
    cr::Matrix{Float64}  # Reconstruction coefficients for right interface
    cl::Matrix{Float64}  # Reconstruction coefficients for left interface
    
    # Gaussian quadrature weights and points
    gauss_weights::Vector{Float64}
    gauss_points::Vector{Float64}
    
    function WENOReconstructor(nx, xmin, xmax, eps=1e-6; gauss_order=8)
        dx = (xmax - xmin) / nx
        x = range(xmin + dx/2, xmax - dx/2, length=nx)  # Cell centers
        xi = range(xmin, xmax, length=nx+1)  # Cell interfaces
        
        # Linear weights for 5th order WENO
        gamma = [0.1, 0.6, 0.3]
        
        # Reconstruction coefficients for u_{i+1/2} from cell averages
        cr = [
            -1/6  5/6  1/3  0    0;    # Stencil 0: i-2, i-1, i
            1/3   5/6 -1/6  0    0;    # Stencil 1: i-1, i, i+1  
            0     1/3  5/6 -1/6  0     # Stencil 2: i, i+1, i+2
        ]
        
        # Reconstruction coefficients for u_{i-1/2}
        cl = [
            0    -1/6  5/6  1/3  0;    # Stencil 0: i-1, i, i+1
            0    1/3   5/6 -1/6  0;    # Stencil 1: i, i+1, i+2
            -1/6 5/6   1/3  0    0     # Stencil 2: i+1, i+2, i+3
        ]
        
        # High-order Gaussian quadrature weights and points
        if gauss_order == 8
            # 8-point Gauss-Legendre quadrature on [-1, 1]
            points = [
                -0.9602898564975362316835609,
                -0.7966664774136267395915539,
                -0.5255324099163289858177390,
                -0.1834346424956498049394761,
                 0.1834346424956498049394761,
                 0.5255324099163289858177390,
                 0.7966664774136267395915539,
                 0.9602898564975362316835609
            ]
            weights = [
                0.1012285362903762591525314,
                0.2223810344533744705443560,
                0.3137066458778872873379622,
                0.3626837833783619829651504,
                0.3626837833783619829651504,
                0.3137066458778872873379622,
                0.2223810344533744705443560,
                0.1012285362903762591525314
            ]
        elseif gauss_order == 16
            # 16-point Gauss-Legendre quadrature for even higher precision
            points = [
                -0.9894009349916499325961542, -0.9445750230732325760779884,
                -0.8656312023878317438804679, -0.7554044083550030338951012,
                -0.6178762444026437484466718, -0.4580167776572273863424194,
                -0.2816035507792589132304605, -0.0950125098376374401853193,
                 0.0950125098376374401853193,  0.2816035507792589132304605,
                 0.4580167776572273863424194,  0.6178762444026437484466718,
                 0.7554044083550030338951012,  0.8656312023878317438804679,
                 0.9445750230732325760779884,  0.9894009349916499325961542
            ]
            weights = [
                0.0271524594117540948517806, 0.0622535239386478928628438,
                0.0951585116824927848099251, 0.1246289712555338720524763,
                0.1495959888165767320815017, 0.1691565193950025381893121,
                0.1826034150449235888667637, 0.1894506104550684962853967,
                0.1894506104550684962853967, 0.1826034150449235888667637,
                0.1691565193950025381893121, 0.1495959888165767320815017,
                0.1246289712555338720524763, 0.0951585116824927848099251,
                0.0622535239386478928628438, 0.0271524594117540948517806
            ]
        else
            # Default to 8-point
            points = [
                -0.9602898564975362316835609, -0.7966664774136267395915539,
                -0.5255324099163289858177390, -0.1834346424956498049394761,
                 0.1834346424956498049394761,  0.5255324099163289858177390,
                 0.7966664774136267395915539,  0.9602898564975362316835609
            ]
            weights = [
                0.1012285362903762591525314, 0.2223810344533744705443560,
                0.3137066458778872873379622, 0.3626837833783619829651504,
                0.3626837833783619829651504, 0.3137066458778872873379622,
                0.2223810344533744705443560, 0.1012285362903762591525314
            ]
        end
        
        new(nx, xmin, xmax, dx, collect(x), collect(xi), eps, gamma, cr, cl, weights, points)
    end
end


function weno5_reconstruct_correct(recon::WENOReconstructor, u_avg::Vector{Float64}, direction="right")
    """Correct WENO5 reconstruction from cell averages to interfaces"""
    n = length(u_avg)
    u_interface = zeros(n)
    
    # Add ghost cells for periodic boundaries
    u_ext = vcat(u_avg[end-2:end], u_avg, u_avg[1:3])
    
    for i in 4:n+3
        if direction == "right"
            # Reconstruction for u_{i+1/2}
            # Stencil 0: cells i-2, i-1, i
            p0 = (1/3)*u_ext[i-3] - (7/6)*u_ext[i-2] + (11/6)*u_ext[i-1]
            beta0 = (13/12)*(u_ext[i-3] - 2*u_ext[i-2] + u_ext[i-1])^2 + 
                   (1/4)*(u_ext[i-3] - 4*u_ext[i-2] + 3*u_ext[i-1])^2
            
            # Stencil 1: cells i-1, i, i+1  
            p1 = (-1/6)*u_ext[i-2] + (5/6)*u_ext[i-1] + (1/3)*u_ext[i]
            beta1 = (13/12)*(u_ext[i-2] - 2*u_ext[i-1] + u_ext[i])^2 + 
                   (1/4)*(u_ext[i-2] - u_ext[i])^2
            
            # Stencil 2: cells i, i+1, i+2
            p2 = (1/3)*u_ext[i-1] + (5/6)*u_ext[i] - (1/6)*u_ext[i+1]
            beta2 = (13/12)*(u_ext[i-1] - 2*u_ext[i] + u_ext[i+1])^2 + 
                   (1/4)*(3*u_ext[i-1] - 4*u_ext[i] + u_ext[i+1])^2
        else
            # Reconstruction for u_{i-1/2}
            # Stencil 0: cells i-1, i, i+1
            p0 = (-1/6)*u_ext[i-2] + (5/6)*u_ext[i-1] + (1/3)*u_ext[i]
            beta0 = (13/12)*(u_ext[i-2] - 2*u_ext[i-1] + u_ext[i])^2 + 
                   (1/4)*(u_ext[i-2] - u_ext[i])^2
            
            # Stencil 1: cells i, i+1, i+2
            p1 = (1/3)*u_ext[i-1] + (5/6)*u_ext[i] - (1/6)*u_ext[i+1]
            beta1 = (13/12)*(u_ext[i-1] - 2*u_ext[i] + u_ext[i+1])^2 + 
                   (1/4)*(3*u_ext[i-1] - 4*u_ext[i] + u_ext[i+1])^2
            
            # Stencil 2: cells i+1, i+2, i+3
            p2 = (11/6)*u_ext[i] - (7/6)*u_ext[i+1] + (1/3)*u_ext[i+2]
            beta2 = (13/12)*(u_ext[i] - 2*u_ext[i+1] + u_ext[i+2])^2 + 
                   (1/4)*(u_ext[i] - 4*u_ext[i+1] + 3*u_ext[i+2])^2
        end
        
        # Nonlinear weights
        alpha0 = 0.1 / (recon.eps + beta0)^2
        alpha1 = 0.6 / (recon.eps + beta1)^2
        alpha2 = 0.3 / (recon.eps + beta2)^2
        
        alpha_sum = alpha0 + alpha1 + alpha2
        omega0 = alpha0 / alpha_sum
        omega1 = alpha1 / alpha_sum  
        omega2 = alpha2 / alpha_sum
        
        # Weighted combination
        u_interface[i-3] = omega0 * p0 + omega1 * p1 + omega2 * p2
    end
    
    return u_interface
end

function test_exact_polynomials()
    """Test that WENO reconstructs polynomials exactly"""
    println("Testing exact polynomial reconstruction")
    println("=" ^ 50)
    
    nx = 20
    xmin, xmax = 0.0, 2.0  # Use smaller domain for better conditioning
    recon = WENOReconstructor(nx, xmin, xmax)
    
    # Test polynomials that should be reconstructed exactly
    test_cases = [
        ("Constant: 2.0", x -> 2.0, x -> 0.0),
        ("Linear: 3x + 1", x -> 3x + 1, x -> 3.0),
        ("Quadratic: x²", x -> x^2, x -> 2x),
    ]
    
    for (name, f, f_prime) in test_cases
        println("\nTesting: ", name)
        
        # Compute cell averages EXACTLY using analytic integration
        u_avg = zeros(nx)
        for i in 1:nx
            x_left = recon.xi[i]
            x_right = recon.xi[i+1]
            
            if name == "Constant: 2.0"
                # ∫ 2.0 dx from x_left to x_right = 2.0 * (x_right - x_left)
                # Average = (1/Δx) * integral = 2.0
                u_avg[i] = 2.0
            elseif name == "Linear: 3x + 1"
                # ∫ (3x + 1) dx = (3/2)x² + x
                integral = (3/2)*x_right^2 + x_right - ((3/2)*x_left^2 + x_left)
                u_avg[i] = integral / recon.dx
            elseif name == "Quadratic: x²"
                # ∫ x² dx = (1/3)x³
                integral = (1/3)*x_right^3 - (1/3)*x_left^3
                u_avg[i] = integral / recon.dx
            end
        end
        
        # Reconstruct
        u_right = weno5_reconstruct_correct(recon, u_avg, "right")
        
        # Check errors
        max_error = 0.0
        for i in 1:nx
            x_interface = recon.xi[i+1]
            exact = f(x_interface)
            error = abs(u_right[i] - exact)
            max_error = max(max_error, error)
            if i <= 5  # Print first few values for debugging
                println("  i=$i, x=$(x_interface), exact=$exact, reconstructed=$(u_right[i]), error=$error")
            end
        end
        
        println("Max reconstruction error: ", max_error)
        println("Should be ~1e-15: ", max_error < 1e-12 ? "✓" : "✗")
        
        # Also test that the cell averages themselves are correct
        println("Cell average verification:")
        for i in 1:min(3, nx)
            x_center = recon.x[i]
            exact_center = f(x_center)
            println("  Cell $i: center_value=$exact_center, cell_avg=$(u_avg[i]), diff=$(abs(u_avg[i] - exact_center))")
        end
    end
end


function weno5_derivative_at_interface(recon::WENOReconstructor, u_avg::Vector{Float64}, direction="right")
    """WENO5 reconstruction of derivative at cell interfaces from cell averages"""
    n = length(u_avg)
    ux_interface = zeros(n)
    
    # Add ghost cells for periodic boundaries
    u_ext = vcat(u_avg[end-2:end], u_avg, u_avg[1:3])
    
    for i in 4:n+3
        if direction == "right"
            # Reconstruction for u_x at i+1/2 interface
            # Stencil 0: cells i-2, i-1, i
            v0 = u_ext[i-3:i+1]  # i-2, i-1, i, i+1
            p0 = (v0[1] - 3*v0[2] + 2*v0[3]) / recon.dx
            beta0 = (13/12)*(v0[1] - 2*v0[2] + v0[3])^2 + 
                   (1/4)*(v0[1] - 4*v0[2] + 3*v0[3])^2
            
            # Stencil 1: cells i-1, i, i+1  
            v1 = u_ext[i-2:i+2]  # i-1, i, i+1, i+2
            p1 = (v1[2] - v1[4]) / (2*recon.dx)  # Central difference
            beta1 = (13/12)*(v1[1] - 2*v1[2] + v1[3])^2 + 
                   (1/4)*(v1[2] - v1[4])^2
            
            # Stencil 2: cells i, i+1, i+2
            v2 = u_ext[i-1:i+3]  # i, i+1, i+2, i+3
            p2 = (-2*v2[1] + 3*v2[2] - v2[3]) / recon.dx
            beta2 = (13/12)*(v2[1] - 2*v2[2] + v2[3])^2 + 
                   (1/4)*(3*v2[1] - 4*v2[2] + v2[3])^2
        else
            # Reconstruction for u_x at i-1/2 interface
            # Stencil 0: cells i-1, i, i+1
            v0 = u_ext[i-3:i+1]  # i-2, i-1, i, i+1  
            p0 = (v0[2] - v0[4]) / (2*recon.dx)  # Central difference
            beta0 = (13/12)*(v0[1] - 2*v0[2] + v0[3])^2 + 
                   (1/4)*(v0[2] - v0[4])^2
            
            # Stencil 1: cells i, i+1, i+2
            v1 = u_ext[i-2:i+2]  # i-1, i, i+1, i+2
            p1 = (-2*v1[1] + 3*v1[2] - v1[3]) / recon.dx
            beta1 = (13/12)*(v1[1] - 2*v1[2] + v1[3])^2 + 
                   (1/4)*(3*v1[1] - 4*v1[2] + v1[3])^2
            
            # Stencil 2: cells i+1, i+2, i+3
            v2 = u_ext[i-1:i+3]  # i, i+1, i+2, i+3
            p2 = (-v2[1] + 3*v2[2] - 2*v2[3]) / recon.dx
            beta2 = (13/12)*(v2[1] - 2*v2[2] + v2[3])^2 + 
                   (1/4)*(v2[1] - 4*v2[2] + 3*v2[3])^2
        end
        
        # Nonlinear weights
        alpha0 = 0.1 / (recon.eps + beta0)^2
        alpha1 = 0.6 / (recon.eps + beta1)^2
        alpha2 = 0.3 / (recon.eps + beta2)^2
        
        alpha_sum = alpha0 + alpha1 + alpha2
        omega0 = alpha0 / alpha_sum
        omega1 = alpha1 / alpha_sum  
        omega2 = alpha2 / alpha_sum
        
        # Weighted combination for derivative
        ux_interface[i-3] = omega0 * p0 + omega1 * p1 + omega2 * p2
    end
    
    return ux_interface
end

# Also let's test with a simple case to debug
function test_simple_linear()
    println("\n" * "=" ^ 50)
    println("Detailed linear function test")
    
    nx = 5
    xmin, xmax = 0.0, 1.0
    recon = WENOReconstructor(nx, xmin, xmax)
    
    f(x) = 2.0 * x + 1.0  # Simple linear function
    
    # Compute exact cell averages
    u_avg = zeros(nx)
    for i in 1:nx
        x_left = recon.xi[i]
        x_right = recon.xi[i+1]
        integral = (x_right^2 + x_right) - (x_left^2 + x_left)  # ∫(2x+1)dx = x² + x
        u_avg[i] = integral / recon.dx
    end
    
    println("Cell averages:")
    for i in 1:nx
        x_center = recon.x[i]
        exact_center = f(x_center)
        println("  Cell $i: center=$(x_center), exact_center=$exact_center, cell_avg=$(u_avg[i])")
    end
    
    println("\nInterfaces:")
    for i in 1:nx+1
        println("  Interface $i: x=$(recon.xi[i]), exact=$(f(recon.xi[i]))")
    end
    
    # Reconstruct
    u_right = weno5_reconstruct_correct(recon, u_avg, "right")
    
    println("\nReconstruction results:")
    for i in 1:nx
        x_interface = recon.xi[i+1]
        exact = f(x_interface)
        error = abs(u_right[i] - exact)
        println("  Interface $i: exact=$exact, reconstructed=$(u_right[i]), error=$error")
    end
end

# Run both tests
#test_exact_polynomials()
#test_simple_linear()


function test_polynomials_up_to_order_4()
    println("Testing WENO reconstruction for polynomials up to order 4")
    println("=" ^ 70)
    
    # Test various polynomials
    test_cases = [
        ("Constant: 5.0", x -> 5.0, x -> 0.0, 0),           # Order 0
        ("Linear: 2x - 3", x -> 2x - 3, x -> 2.0, 1),       # Order 1  
        ("Quadratic: x² + x - 1", x -> x^2 + x - 1, x -> 2x + 1, 2),  # Order 2
        ("Cubic: x³ - 2x", x -> x^3 - 2x, x -> 3x^2 - 2, 3),          # Order 3
        ("Quartic: x⁴ - x² + 1", x -> x^4 - x^2 + 1, x -> 4x^3 - 2x, 4),  # Order 4
    ]
    
    nx = 20
    xmin, xmax = 0.0, 2.0
    recon = WENOReconstructor(nx, xmin, xmax)
    
    for (name, f, f_prime, max_exact_order) in test_cases
        println("\nTesting: $name (should be exact up to order $max_exact_order)")
        
        # Compute exact cell averages using analytic integration
        u_avg = zeros(nx)
        for i in 1:nx
            x_left = recon.xi[i]
            x_right = recon.xi[i+1]
            
            if max_exact_order == 0  # Constant
                integral = f(x_left) * recon.dx
            elseif max_exact_order == 1  # Linear
                # ∫(ax + b)dx = (a/2)x² + bx
                integral = (f(x_right) + f(x_left)) * recon.dx / 2  # Trapezoidal rule is exact
            elseif max_exact_order >= 2  # Higher order - use symbolic integration
                # For polynomial f(x) = a₀ + a₁x + a₂x² + a₃x³ + a₄x⁴
                # ∫f(x)dx = a₀x + (a₁/2)x² + (a₂/3)x³ + (a₃/4)x⁴ + (a₄/5)x⁵
                if name == "Quadratic: x² + x - 1"
                    # f(x) = x² + x - 1, ∫f(x)dx = (1/3)x³ + (1/2)x² - x
                    integral = (1/3)*x_right^3 + (1/2)*x_right^2 - x_right - ((1/3)*x_left^3 + (1/2)*x_left^2 - x_left)
                elseif name == "Cubic: x³ - 2x"  
                    # f(x) = x³ - 2x, ∫f(x)dx = (1/4)x⁴ - x²
                    integral = (1/4)*x_right^4 - x_right^2 - ((1/4)*x_left^4 - x_left^2)
                elseif name == "Quartic: x⁴ - x² + 1"
                    # f(x) = x⁴ - x² + 1, ∫f(x)dx = (1/5)x⁵ - (1/3)x³ + x
                    integral = (1/5)*x_right^5 - (1/3)*x_right^3 + x_right - ((1/5)*x_left^5 - (1/3)*x_left^3 + x_left)
                else
                    # Fallback: use Gaussian quadrature
                    integral = 0.0
                    for (w, p) in zip(recon.gauss_weights, recon.gauss_points)
                        x_mapped = 0.5 * (x_right - x_left) * p + 0.5 * (x_right + x_left)
                        integral += w * f(x_mapped)
                    end
                    integral *= 0.5 * recon.dx  # weights sum to 2
                end
            end
            
            u_avg[i] = integral / recon.dx
        end
        
        # Reconstruct to right interfaces
        u_right_rec = weno5_reconstruct_correct(recon, u_avg, "right")
        ux_right_rec = weno5_derivative_at_interface(recon, u_avg, "right")
        
        # Compute errors
        max_u_error = 0.0
        max_ux_error = 0.0
        rms_u_error = 0.0
        rms_ux_error = 0.0
        
        for i in 1:nx
            x_interface = recon.xi[i+1]
            exact_u = f(x_interface)
            exact_ux = f_prime(x_interface)
            
            u_error = abs(u_right_rec[i] - exact_u)
            ux_error = abs(ux_right_rec[i] - exact_ux)
            
            max_u_error = max(max_u_error, u_error)
            max_ux_error = max(max_ux_error, ux_error)
            rms_u_error += u_error^2
            rms_ux_error += ux_error^2
        end
        
        rms_u_error = sqrt(rms_u_error / nx)
        rms_ux_error = sqrt(rms_ux_error / nx)
        
        println("  Function reconstruction:")
        println("    Max error: $(@sprintf("%.2e", max_u_error))")
        println("    RMS error: $(@sprintf("%.2e", rms_u_error))")
        
        println("  Derivative reconstruction:")  
        println("    Max error: $(@sprintf("%.2e", max_ux_error))")
        println("    RMS error: $(@sprintf("%.2e", rms_ux_error))")
        
        # Check if reconstruction is exact (up to machine precision)
        if max_exact_order >= 2  # WENO5 should be exact for polynomials up to degree 4
            expected_precision = 1e-12
            u_exact = max_u_error < expected_precision
            ux_exact = max_ux_error < expected_precision
            println("  Should be exact: u=$(u_exact ? "✓" : "✗"), u_x=$(ux_exact ? "✓" : "✗")")
        else
            println("  Not expected to be exact (max order: $max_exact_order)")
        end
        
        # Print first few values for inspection
        println("  First few values (i, x_interface, exact_u, reconstructed_u, error):")
        for i in 1:min(3, nx)
            x_interface = recon.xi[i+1]
            exact_u = f(x_interface)
            error_u = abs(u_right_rec[i] - exact_u)
            println("    $i, $(@sprintf("%.3f", x_interface)), $(@sprintf("%.6f", exact_u)), $(@sprintf("%.6f", u_right_rec[i])), $(@sprintf("%.2e", error_u))")
        end
    end
    
    # Additional test: verify reconstruction polynomials directly
    println("\n" * "=" ^ 70)
    println("Direct verification of reconstruction polynomials:")
    test_reconstruction_polynomials_analytically()
end

function test_reconstruction_polynomials_analytically()
    """Test if the reconstruction polynomials are correct by checking exactness"""
    
    # Test with u(x) = x (linear function)
    println("\nTesting reconstruction for u(x) = x:")
    
    # For uniform grid with cell centers at 1,2,3,4,5
    # Cell averages are also 1,2,3,4,5 (since average = center for linear functions)
    u_avg = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # We want to reconstruct at interface between cell 3 and 4, which is at x = 3.5
    # Using stencil centered at i=3: cells i-2=1, i-1=2, i=3, i+1=4, i+2=5
    
    u_im2, u_im1, u_i, u_ip1, u_ip2 = 1.0, 2.0, 3.0, 4.0, 5.0
    
    # Reconstruction polynomials for u_{i+1/2}
    p0 = (1/3)*u_im2 - (7/6)*u_im1 + (11/6)*u_i
    p1 = (-1/6)*u_im1 + (5/6)*u_i + (1/3)*u_ip1  
    p2 = (1/3)*u_i + (5/6)*u_ip1 - (1/6)*u_ip2
    
    exact_interface = 3.5
    
    println("  Stencil 0: p0 = $p0, error = $(p0 - exact_interface)")
    println("  Stencil 1: p1 = $p1, error = $(p1 - exact_interface)")
    println("  Stencil 2: p2 = $p2, error = $(p2 - exact_interface)")
    
    # All should give exactly 3.5 for linear functions
    tolerance = 1e-14
    all_exact = abs(p0 - exact_interface) < tolerance && 
                abs(p1 - exact_interface) < tolerance && 
                abs(p2 - exact_interface) < tolerance
    
    println("  All stencils exact for linear function: $(all_exact ? "✓" : "✗")")
    
    # Test with quadratic function u(x) = x²
    println("\nTesting reconstruction for u(x) = x²:")
    
    # Cell centers at 1,2,3,4,5 → averages = 1,4,9,16,25
    u_avg_quad = [1.0, 4.0, 9.0, 16.0, 25.0]
    u_im2, u_im1, u_i, u_ip1, u_ip2 = 1.0, 4.0, 9.0, 16.0, 25.0
    
    p0_quad = (1/3)*u_im2 - (7/6)*u_im1 + (11/6)*u_i
    p1_quad = (-1/6)*u_im1 + (5/6)*u_i + (1/3)*u_ip1  
    p2_quad = (1/3)*u_i + (5/6)*u_ip1 - (1/6)*u_ip2
    
    exact_interface_quad = 3.5^2  # 12.25
    
    println("  Stencil 0: p0 = $p0_quad, error = $(p0_quad - exact_interface_quad)")
    println("  Stencil 1: p1 = $p1_quad, error = $(p1_quad - exact_interface_quad)")
    println("  Stencil 2: p2 = $p2_quad, error = $(p2_quad - exact_interface_quad)")
    
    # For quadratic functions, the optimal linear combination should be exact
    omega0, omega1, omega2 = 0.1, 0.6, 0.3
    weighted_quad = omega0*p0_quad + omega1*p1_quad + omega2*p2_quad
    println("  Weighted combination: $weighted_quad, error = $(weighted_quad - exact_interface_quad)")
    
    weighted_exact = abs(weighted_quad - exact_interface_quad) < tolerance
    println("  Weighted combination exact for quadratic: $(weighted_exact ? "✓" : "✗")")
end


# Run the comprehensive test
test_polynomials_up_to_order_4()