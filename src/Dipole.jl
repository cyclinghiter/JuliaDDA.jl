mutable struct Dipole
    """
    A struct representing an electric dipole.

    Fields:
    pos: A 3-element static array of type Real representing the position of the dipole.
    Einc: A 3-element static array of type ComplexF64 representing the incident electric field on the dipole.
    P: A 3-element static array of type ComplexF64 representing the dipole moment of the dipole.
    α: A 3-element static array of type ComplexF64 representing the polarizability tensor of the dipole.
    """

    pos :: SVector{3, Real} 
    Einc :: Union{Nothing, SVector{3, ComplexF64}}  # Use Union{Nothing, ...} to allow for default values.
    P :: Union{Nothing, SVector{3, ComplexF64}}
    α :: Union{Nothing, SVector{3, ComplexF64}}

    # Define a constructor for Dipole objects.
    function Dipole(pos; Einc=nothing, P=nothing, α=nothing)
        # Use isnothing() to check if optional arguments were passed.
        if isnothing(Einc)
            Einc = SA[0+0im, 0+0im, 0+0im]
        end
        # Use new() to create a new object with the specified fields.
        new(pos, Einc, P, α)
    end
end 

# Define another constructor that takes x, y, z coordinates instead of an SVector.
Dipole(x, y, z; Einc=nothing, P=nothing, α=nothing) = Dipole(SA[x,y,z]; Einc=Einc, P=P, α=α)

# Define a custom getproperty() function to allow accessing Dipole fields with dot notation.
function Base.getproperty(Dip::Dipole, sym::Symbol)
    # Use if-elseif statements to return the correct field.
    if sym == :x
        return Dip.pos[1]
    elseif sym == :y
        return Dip.pos[2]
    elseif sym == :z
        return Dip.pos[3]

    elseif sym == :r
        return CarttoSpherical(Dip.pos)[1]
    elseif sym == :θ
        return CarttoSpherical(Dip.pos)[2]
    elseif sym == :ϕ
        return CarttoSpherical(Dip.pos)[3]

    elseif sym == :Eincx
        return Dip.Einc[1]
    elseif sym == :Eincy
        return Dip.Einc[2]
    elseif sym == :Eincz
        return Dip.Einc[3]

    elseif sym == :Px
        return Dip.P[1]
    elseif sym == :Py
        return Dip.P[2]
    elseif sym == :Pz
        return Dip.P[3]

    elseif sym == :αx
        return Dip.α[1]
    elseif sym == :αy
        return Dip.α[2]
    elseif sym == :αz
        return Dip.α[3]

    else
        # If the field isn't one of the above, use getfield() to return it.
        return getfield(Dip, sym)
    end
end

# Define another custom getproperty() function to allow accessing Dipole arrays with dot notation.
function Base.getproperty(Dip::Array{Dipole}, sym::Symbol)
    # Use broadcasting to apply getproperty() to each Dipole in the array.
    getproperty.(Dip, sym)
end
