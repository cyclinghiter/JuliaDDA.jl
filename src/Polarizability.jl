function Clausius_Mossoti_Polarizability(ϵ::Number, μ::Number, d::Real)
    n = √(Complex(ϵ*μ))
    dV  = 3/(4*π)* d^3
    α = dV * (n^2 - 1) / (n^2 + 2) * ones(ComplexF64, 3)
    return α
end

function Clausius_Mossoti_Polarizability(D::Dipole)
    if isnothing(D.d)
        error("d is missing")
    end
    α = Clausius_Mossoti_Polarizability(D.ϵ, D.μ, D.d)
    return α
end

function Lattice_Dispersion_Polarizability(ϵ::Number, μ::Number, d::Real, k::Number)
    α_CM = Clausius_Mossoti_Polarizability(ϵ, μ, d)
    b1 = -1.8915316
    b2 = 0.1648469
    b3 = -1.7700004
    S = .0
    α_LDR = @. α_CM / (1 + α_CM / d^3 * ((b1 + (ϵ*μ) * b2 + (ϵ*μ) * b3 * S)*(k * d)^2 - 2/3 * 1im * k^3 * d^3))
    return α_LDR
end

function Lattice_Dispersion_Polarizability(D::Dipole, k)
    if isnothing(D.d)
        error("d is missing")
    end
    α = Lattice_Dispersion_Polarizability(D.ϵ, D.μ, D.d, k)
    return α
end