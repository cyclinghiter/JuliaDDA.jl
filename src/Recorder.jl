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

function SphericalRecorder(num_θ::Int, num_ϕ::Int; R=1e10, mode="top_hemi")
    if mode == "top_hemi"
        θlist = Vector(LinRange(0, π/2, num_θ))
        ϕlist = Vector(LinRange(0, 2π, num_ϕ))
        recorders = SphericalRecorder(θlist, ϕlist, R=R)
    elseif mode == "full"
        θlist = Vector(LinRange(0, π, num_θ))
        ϕlist = Vector(LinRange(0, 2π, num_ϕ))
        recorders = SphericalRecorder(θlist, ϕlist, R=R)
    end
    return recorders
end

function PlaneRecorder(xlist::Vector, ylist::Vector, z=0; θ=0, ϕ=0)
    recorders = Array{Recorder}(undef, length(xlist), length(ylist))
    for (i, x) in enumerate(xlist)
        for (j, y) in enumerate(ylist)
            # Oiler rotation
            x = x*cos(θ)*cos(ϕ) - y*cos(θ)*sin(ϕ) + z*sin(θ)
            y = x*sin(ϕ) + y*cos(ϕ)
            z = -x*sin(θ)*cos(ϕ) + y*sin(θ)*sin(ϕ) + z*cos(θ)
            recorders[i, j] = Recorder(SA[x,y,z], SA[0+0im,0+0im,0+0im])
        end
    end
    return recorders
end

function SphericalUVRecorder(ulist::Vector, vlist::Vector; R=1e10)
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

function SphericalUVRecorder(num_u::Int, num_v::Int; R=1e10)
    ulist = Vector(LinRange(-1, 1, num_u))
    vlist = Vector(LinRange(-1, 1, num_v))
    recorders = SphericalUVRecorder(ulist, vlist, R=R)
    return recorders
end

function PoyntingVector(Rec::Recorder; n̂=nothing)
    if isnothing(n̂)
        n̂ = unitvec(Rec.pos)
    end
    H = 1/Z0*cross(n̂, Rec.E)
    return 1/2 * real(cross(Rec.E, conj.(H)))
end

function Power(Rec::Recorder; n̂::T=nothing) where T <: Union{Nothing, SVector{3}}
    if isnothing(n̂)
        n̂ = unitvec(Rec.pos)
    end
    S = PoyntingVector(Rec; n̂=n̂)
    return dot(S, n̂)
end

function PoyntingVector(Recorderlist::Array{Recorder}; n̂=nothing)
    Slist = zeros(SVector{3}, size(Recorderlist))
    for (i, rec) in enumerate(Recorderlist)
        S = PoyntingVector(rec; n̂=n̂)
        Slist[i] = S
    end 
    return Slist
end

function Power(Recorderlist::Array{Recorder}; n̂=nothing)
    Plist = zeros(size(Recorderlist))
    for (i, rec) in enumerate(Recorderlist)
        P = Power(rec; n̂=n̂)
        Plist[i] = P
    end 
    return Plist
end