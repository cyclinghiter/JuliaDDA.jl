function Pan(Obj::T, vec::SVector{3, U}) where {T <: Structure, U <: Real}
    for D in Obj.Dipoles
        D.pos += vec
    end
end

function Rotate(Obj::T, θ; center=SA[0., 0., 0.], n̂=SA[0., 0., 1.]) where {T <: Structure}
    for D in Obj.Dipoles
        D.pos -= center
        D.pos = RotationMatrix(θ; n̂=n̂) * D.pos
        D.pos += center
    end
    return Obj
end
