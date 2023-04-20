# This function calculates the polarizability of a single atom using the Clausius-Mossotti equation.
# Arguments:
# - ϵ: permittivity
# - μ: permeability
# - d: distance
# Returns:
# - α: polarizability
function Clausius_Mossoti_Polarizability(ϵ::Number, μ::Number, d::Real)
    # Calculate refractive index
    n = √(Complex(ϵ*μ))
    # Calculate volume occupied by atom
    dV  = 3/(4*π)* d^3
    # Calculate polarizability using Clausius-Mossotti equation
    α = dV * (n^2 - 1) / (n^2 + 2) * ones(ComplexF64, 3)
    return α
end

# This function calculates the polarizability of a single atom using the Clausius-Mossotti equation,
# taking the values from a Dipole object.
# Arguments:
# - D: Dipole object containing ϵ, μ, and d values
# Returns:
# - α: polarizability
function Clausius_Mossoti_Polarizability(D::Dipole)
    # Check that d value is not missing
    if isnothing(D.d)
        error("d is missing")
    end
    # Calculate polarizability using first function
    α = Clausius_Mossoti_Polarizability(D.ϵ, D.μ, D.d)
    return α
end

# This function calculates the lattice dispersion polarizability of a material,
# using the Clausius-Mossotti polarizability and a dispersion correction.
# Arguments:
# - ϵ: permittivity
# - μ: permeability
# - d: distance
# - k: wavenumber
# Returns:
# - α_LDR: lattice dispersion polarizability
function Lattice_Dispersion_Polarizability(ϵ::Number, μ::Number, d::Real, k::Number)
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

# This function calculates the lattice dispersion polarizability of a material,
# taking the values from a Dipole object and using a dispersion correction.
# Arguments:
# - D: Dipole object containing ϵ, μ, and d values
# - k: wavenumber
# Returns:
# - α: lattice dispersion polarizability
function Lattice_Dispersion_Polarizability(D::Dipole, k)
    # Check that d value is not missing
    if isnothing(D.d)
        error("d is missing")
    end
    # Calculate lattice dispersion polarizability using third function
    α = Lattice_Dispersion_Polarizability(D.ϵ, D.μ, D.d, k)
    return