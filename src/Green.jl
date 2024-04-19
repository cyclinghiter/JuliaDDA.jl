function FreeSpaceScalarGreen(k, R)
    return exp(1im * k * R) / (R)
end

function FreeSpaceGreenTensor(k, D1::Dipole, D2::Dipole)
    if D1.pos == D2.pos
        𝔸 = zeros(ComplexF64, (3, 3))
        for i in 1:3
            𝔸[i,i] = D1.α[i] == 0 ? 0 : 1 / D1.α[i]
        end
        return 𝔸
    else
        r = D2.pos - D1.pos
        r_norm = norm(r)
        r_hat = r ./ r_norm
        rrT = r_hat * transpose(r_hat)
        𝔸 = FreeSpaceScalarGreen(k, r_norm) .* (k^2 .* (rrT - I) + (1im * k * r_norm - 1) / r_norm^2 .* (3 .* rrT - I))
        return 𝔸
    end
end

function FreeSpaceGreenTensor(k, D1::Dipole, R::Recorder)
    r = R.pos - D1.pos
    r_norm = norm(r)
    r_hat = r ./ r_norm
    rrT = r_hat * transpose(r_hat)
    𝔸 = FreeSpaceScalarGreen(k, r_norm) .* (k^2 .* (rrT - I) + (1im * k * r_norm - 1) / r_norm^2 .* (3 .* rrT - I))
    return 𝔸
end