abstract type Structure
end

function Base.getproperty(S::T, sym::Symbol) where T <: Structure
    if sym in [:x, :y, :z, 
        :r, :θ, :ϕ, 
        :Px, :Py, :Pz, 
        :Eincx, :Eincy, :Eincz, 
        :αx, :αy, :αz,
        :P, :Einc, :α, :pos]
    return getproperty.(S.Dipoles, sym)
    else
        return getfield(S, sym)
    end
end

function reset_dipoles(S::T) where T <: Structure
    for dip in S.Dipoles
        dip.Einc = SA[0+0im, 0+0im, 0+0im]
        dip.P = nothing
    end
end

mutable struct CustomStructure <: Structure
    Dipoles :: Vector{Dipole}
    name :: String
    function CustomStructure(Dipoles::Vector{Dipole}, name="CustomStructure")
        new(Dipoles, name)
    end
end

mutable struct Sphere <: Structure

    radius :: Real
    n_per_dim :: Int
    ϵ :: Number
    μ :: Number
    Dipoles :: Vector{Dipole}
    p_mode :: Symbol
    name :: String
    
    function Sphere(radius, n_per_dim, ϵ, μ; p_mode = :CM, center = [0, 0, 0], k = nothing, name = "Sphere-$(randstring(12))")
        V = 4/3 * pi * radius^3
        Range = LinRange(-radius, radius, n_per_dim)
        Dipoles = Dipole[]

        for x in Range, y in Range, z in Range
            if x^2 + y^2 + z^2 <= radius^2
                dipole = Dipole(x+center[1], y+center[2], z+center[3])
                push!(Dipoles, dipole)
            end
        end

        N = length(Dipoles)
        d = (V / N)^(1/3)

        if p_mode == :CM
            α = Clausius_Mossoti_Polarizability.(ϵ, μ, d)
        elseif p_mode == :LDR
            if isnothing(k)
                throw(ArgumentError("k is not set properly. In the case of LDR, Set k to a real number to calculate α."))
            else
                α = Lattice_Dispersion_Polarizability.(ϵ, μ, d, k)
            end
        end

        initial_spacing = 2 * radius / (n_per_dim - 1)
        setproperty!.(Dipoles, :pos, getproperty.(Dipoles, :pos) * (d / initial_spacing))
        
        for dipole in Dipoles
            setproperty!(dipole, :α, α)
        end

        new(radius, n_per_dim, ϵ, μ, Dipoles, p_mode, name)
    end
end

mutable struct Box <: Structure

    x :: Number
    y :: Number
    z :: Number
    n_x :: Number
    n_y :: Number
    n_z :: Number
    ϵ :: Number
    μ :: Number
    Dipoles :: Vector{Dipole}
    p_mode :: Symbol
    name :: String

    function Box(x, y, z, n_x, n_y, n_z, ϵ, μ; p_mode=:CM, center=[0, 0, 0], k=nothing, name="Box-$(randstring(12))")
        x_range = LinRange(-x/2, x/2, n_x+1)
        y_range = LinRange(-y/2, y/2, n_y+1)
        z_range = LinRange(-z/2, z/2, n_z+1)
        V = x * y * z

        Dipoles = Dipole[]
        for xi in x_range[1:end-1]
            for yi in y_range[1:end-1]
                for zi in z_range[1:end-1]
                    dipole = Dipole(xi+center[1], yi+center[2], zi+center[3])
                    push!(Dipoles, dipole)
                end
            end
        end

        N = length(Dipoles)
        d = (V / N)^(1/3)

        if p_mode == :CM
            α = Clausius_Mossoti_Polarizability.(ϵ, μ, d)
        elseif p_mode == :LDR
            if isnothing(k)
                throw(ArgumentError("k is not set properly. In the case of LDR, Set k to a real number to calculate α."))
            else
                α = Lattice_Dispersion_Polarizability.(ϵ, μ, d, k)
            end
        end
        
        for dipole in Dipoles
            setproperty!(dipole, :α, α)
        end

        return new(x, y, z, n_x, n_y, n_z, ϵ, μ, Dipoles, p_mode, name)
    end
end

mutable struct Plane <: Structure

    xsize :: Number
    ysize :: Number
    z :: Number
    n_x :: Number
    n_y :: Number
    ϵ :: Number
    μ :: Number
    Dipoles :: Vector{Dipole}
    p_mode :: Symbol
    name :: String

    function Plane(xsize, ysize, z, n_x, n_y, ϵ, μ; p_mode=:CM, center=[0, 0, 0], k=nothing, name="Box-$(randstring(12))")
        x_range = LinRange(-xsize/2, xsize/2, n_x+1)
        y_range = LinRange(-ysize/2, ysize/2, n_y+1)
        d = sqrt((xsize / (n_x+1)) * (ysize / (n_y+1)))

        if p_mode == :CM
            α = Clausius_Mossoti_Polarizability.(ϵ, μ, d)

        elseif p_mode == :LDR
            if isnothing(k)
                throw(ArgumentError("k is not set properly. In the case of LDR, Set k to a real number to calculate α."))
            else
                α = Lattice_Dispersion_Polarizability.(ϵ, μ, d, k)
            end
        end
        
        Dipoles = Dipole[]
        for xi in x_range[1:end-1]
            for yi in y_range[1:end-1]
                dipole = Dipole(xi+center[1], yi+center[2], z+center[3], α=α)
                push!(Dipoles, dipole)
            end
        end

        return new(xsize, ysize, z, n_x, n_y, ϵ, μ, Dipoles, p_mode, name)
    end
end
