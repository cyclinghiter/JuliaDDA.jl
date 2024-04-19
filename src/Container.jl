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

function Base.push!(C::Container, S::T) where T <: Structure
    push!(C.Structures, S)
    for dipole in S.Dipoles
        if ~(dipole in C.Dipoles)
            push!(C.Dipoles, dipole)
        else
            error("Same Dipole already exists, Can't push Structure to the container.")
        end
    end
end

function Base.push!(C::Container, D::T) where T <: Dipole
    if ~(D in C.Dipoles)
        push!(C.Dipoles, D)
    else
        error("Same Dipole already exists, Can't push Dipole to the container.")
    end
end

function remove(C::Container, S::T) where T <: Structure
    deleteat!(C.Structures, findall(x->x==S, C.Structures))
    deleteat!(C.Dipoles,findall(x->x in S.Dipoles, C.Dipoles))
end

function reset_dipole(C::Container)
    for dip in C.Dipoles
        dip.Einc = SA[0+0im, 0+0im, 0+0im]
        dip.P = nothing
    end
end

get_Einc(C::T) where T <: Union{Container, Structure} = reduce(vcat, C.Einc)
get_P(C::T) where T <: Union{Container, Structure} = reduce(vcat, C.P)
get_α(C::T) where T <: Union{Container, Structure} = reduce(vcat, C.α)

