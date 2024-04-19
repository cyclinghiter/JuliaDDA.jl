function Clausius_Mossoti_Polarizability(ϵ::Number, μ::Number, d::Real)

    # This function calculates the polarizability of a single atom using the Clausius-Mossotti equation.
    # Arguments:
    # - ϵ: permittivity
    # - μ: permeability
    # - d: distance
    # Returns:
    # - α: polarizability

    n = √(Complex(ϵ*μ))
    dV  = 3/(4*π)* d^3
    α = dV * (n^2 - 1) / (n^2 + 2) * (@SArray ones(ComplexF64, 3))
    return α
end


function Lattice_Dispersion_Polarizability(ϵ::Number, μ::Number, d::Real, k::Number)

    # This function calculates the lattice dispersion polarizability of a material,
    # using the Clausius-Mossotti polarizability and a dispersion correction.
    # Arguments:
    # - ϵ: permittivity
    # - μ: permeability
    # - d: distance
    # - k: wavenumber
    # Returns:
    # - α_LDR: lattice dispersion polarizability
    
    # Calculate Clausius-Mossotti polarizability
    α_CM = Clausius_Mossoti_Polarizability(ϵ, μ, d)
    # Define constants and variables
    b1 = -1.8915316
    b2 = 0.1648469
    b3 = -1.7700004
    S = .0
    # Calculate lattice dispersion polarizability using dispersion correction
    α_LDR = @. α_CM / (1 + α_CM / d^3 * ((b1 + (ϵ*μ) * b2 + (ϵ*μ) * b3 * S)*(k * d)^2 - 2/3 * 1im * k^3 * d^3))
    return α_LDR
end
