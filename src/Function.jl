function displacement(v1 :: AbstractVector, v2 :: AbstractVector)
    return v1 .- v2
end

function distance(v1 :: AbstractVector, v2 :: AbstractVector)
    return norm(displacement(v1, v2))
end

function unitvec(v1 :: AbstractVector)
    return v1 / norm(v1)
end

function CarttoSpherical(x, y, z)
    r = sqrt.(x.^2 .+ y .^2 .+ z .^2)
    θ = acos.(z ./ r)
    ϕ = atan.(y, x)
    return r, θ, ϕ
end

function CarttoSpherical(X::AbstractVector)
    x = X[1]
    y = X[2]
    z = X[3]
    r, θ, ϕ = CarttoSpherical(x, y, z)
    return r, θ, ϕ
end

function SphericaltoCart(r, θ, ϕ)
    x = r .* sin.(θ) .* cos.(ϕ)
    y = r .* sin.(θ) .* sin.(ϕ)
    z = r .* cos.(θ)
    return x, y, z
end

function SphericaltoCart(X::AbstractVector)
    r = X[1]
    θ = X[2]
    ϕ = X[3]
    x, y, z = SphericaltoCart(r, θ, ϕ)
    return x, y, z
end

function RotationMatrix(θ; n̂=SA[0., 0., 1.])
    n̂ = n̂ ./ norm(n̂)
    x = n̂[1]
    y = n̂[2]
    z = n̂[3]
    return SA[cos(θ)+x^2*(1-cos(θ)) x*y*(1-cos(θ))-z*sin(θ) x*z*(1-cos(θ))+y*sin(θ);
              y*x*(1-cos(θ))+z*sin(θ) cos(θ)+y^2*(1-cos(θ)) y*z*(1-cos(θ))-x*sin(θ);
              z*x*(1-cos(θ))-y*sin(θ) z*y*(1-cos(θ))+x*sin(θ) cos(θ)+z^2*(1-cos(θ))]
end
