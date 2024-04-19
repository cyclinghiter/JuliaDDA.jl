mutable struct Recorder
    pos :: SVector{3}
    E :: SVector{3, ComplexF32}
end

function Recorder(pos::SVector{3})
    return Recorder(pos, SA[0+0im,0+0im,0+0im])
end

function Recorder(x::Real, y::Real, z::Real)
    return Recorder(SA[x, y, z])
end

function Base.getproperty(Rec :: Recorder, sym :: Symbol)
    if sym == :x
        return Rec.pos[1]
    elseif sym == :y
        return Rec.pos[2]
    elseif sym == :z
        return Rec.pos[3]
    elseif sym == :r
        return CarttoSpherical(Rec.pos)[1]
    elseif sym == :θ
        return CarttoSpherical(Rec.pos)[2]
    elseif sym == :ϕ
        return CarttoSpherical(Rec.pos)[3]
    elseif sym == :Ex
        return Rec.E[1]
    elseif sym == :Ey
        return Rec.E[2]
    elseif sym == :Ez
        return Rec.E[3]
    else
        return getfield(Rec, sym)
    end
end

function Base.getproperty(Rec :: Array{Recorder}, sym :: Symbol)
    getproperty.(Rec, sym)
end

function SphericalRecorder(θlist::Vector, ϕlist::Vector; R=1e10)
    recorders = Array{Recorder}(undef, length(θlist), length(ϕlist))
    for (i, θ) in enumerate(θlist)
        for (j, ϕ) in enumerate(ϕlist)
            x = R * sin(θ) * cos(ϕ)
            y = R * sin(θ) * sin(ϕ)
            z = R * cos(θ)
            recorders[i, j] = Recorder(SA[x,y,z], SA[0+0im,0+0im,0+0im])
        end
    end
    return recorders
end

function SphericalRecorder(num_θ::Int, num_ϕ::Int; R=1e10, mode="top_hemisphere")
    if mode == "top_hemisphere"
        θlist = Vector(LinRange(0, π/2, num_θ))
        ϕlist = Vector(LinRange(0, 2π, num_ϕ))
        recorders = SphericalRecorder(θlist, ϕlist, R=R)

    elseif mode == "bottom_hemisphere"
        θlist = Vector(LinRange(π/2, π, num_θ))
        ϕlist = Vector(LinRange(0, 2π, num_ϕ))
        recorders = SphericalRecorder(θlist, ϕlist, R=R)

    elseif mode == "full"
        θlist = Vector(LinRange(0, π, num_θ))
        ϕlist = Vector(LinRange(0, 2π, num_ϕ))
        recorders = SphericalRecorder(θlist, ϕlist, R=R)
    end
    return recorders
end

function UVRecorder(ulist::Vector, vlist::Vector; R=1e10)
    recorders = Array{Recorder}(undef, length(ulist), length(vlist))
    for (i, u) in enumerate(ulist)
        for (j, v) in enumerate(vlist)
            x = R * u
            y = R * v
            z = R * (1 .- (u .^ 2 .+ v .^2))
            recorders[i, j] = Recorder(SA[x,y,z], SA[0+0im,0+0im,0+0im])
        end
    end
    return recorders
end

function UVRecorder(num_u::Int, num_v::Int; R=1e10)
    ulist = Vector(LinRange(-1, 1, num_u))
    vlist = Vector(LinRange(-1, 1, num_v))
    recorders = UVRecorder(ulist, vlist, R=R)
    return recorders
end
