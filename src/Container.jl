mutable struct Container
    k::Real
    Structures::Vector{Structure}
    Dipoles::Vector{Dipole}

    function Container(k::Real)
        new(k, Structure[], Dipole[])
    end
end

function Base.getproperty(C::Container, sym::Symbol)
    if sym in [:x, :y, :z, 
               :r, :θ, :ϕ, 
               :Px, :Py, :Pz, 
               :Eincx, :Eincy, :Eincz, 
               :αx, :αy, :αz,
               :P, :Einc, :α, :pos]
        return getproperty.(C.Dipoles, sym)
    else
        return getfield(C, sym)
    end
end

function Base.push!(C::Container, S::Structure)
    push!(C.Structures, S)
    for dipole in S.Dipoles
        if ~(dipole in C.Dipoles)
            push!(C.Dipoles, dipole)
        else
            error("Dipole already exists, Can't push it to the container.")
        end
    end
end

get_Einc(C::T) where T <: Union{Container, Structure} = reduce(vcat, C.Einc)
get_P(C::T) where T <: Union{Container, Structure} = reduce(vcat, C.P)
get_α(C::T) where T <: Union{Container, Structure} = reduce(vcat, C.α)
