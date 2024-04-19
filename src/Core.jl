function CouplingTensorMat(k, Dipolelist :: Array{Dipole}, GreenFunction::Function = FreeSpaceGreenTensor)
    n = length(Dipolelist)
    𝔸 = zeros(ComplexF64, (3*n, 3*n))
    @inbounds Threads.@threads for j in 1:n
        D1 = Dipolelist[j]
        for i in j:n
            D2 = Dipolelist[i]
            CouplingTensorD1D2 = GreenFunction(k, D1, D2)
            𝔸[3*i-2:3*i, 3*j-2:3*j] = CouplingTensorD1D2
            𝔸[3*j-2:3*j, 3*i-2:3*i] = CouplingTensorD1D2
        end
    end
    return 𝔸
end

function CouplingTensorMat(C::Container)
    return CouplingTensorMat(C.k, C.Dipoles)
end

function CouplingTensorMatGPU(k, Dipolelist :: Array{Dipole}, num_threads = 512)
    II = CuArray(Matrix{ComplexF64}(I, 3,3))
    r_array = CuArray(Matrix{ComplexF64}(reduce(hcat, getproperty.(Dipolelist, :pos))))
    A_ = CUDA.zeros(ComplexF64, length(Dipolelist)*3, length(Dipolelist)*3)
    α = CuArray(Matrix{ComplexF64}(reduce(hcat, getproperty.(Dipolelist, :α))))

    function calA(k, r_array, α, A_, n_d)
        i = (threadIdx().x + (blockIdx().x - 1) * blockDim().x)
        if i <= n_d * n_d
            idx1 = (i-1) ÷ n_d + 1
            idx2 = (i-1) % n_d + 1
            if idx1 == idx2
                A_[3*idx1-2, 3*idx2-2] = 1 ./ α[1, idx1]
                if α[1, idx1] == 0
                    A_[3*idx1-2, 3*idx2-2] = 0
                end
                A_[3*idx1-2, 3*idx2-1] = 0
                A_[3*idx1-2, 3*idx2] = 0
                A_[3*idx1-1, 3*idx2-2] = 0
                A_[3*idx1-1, 3*idx2-1] = 1 ./ α[2, idx1]
                if α[2, idx1] == 0
                    A_[3*idx1-1, 3*idx2-1] = 0
                end
                A_[3*idx1-1, 3*idx2] = 0
                A_[3*idx1, 3*idx2-2] = 0
                A_[3*idx1, 3*idx2-1] = 0
                A_[3*idx1, 3*idx2] = 1 ./ α[3, idx1]
                if α[3, idx1] == 0
                    A_[3*idx1, 3*idx2] = 0
                end
                
            else
                r1 = r_array[1, idx1] - r_array[1, idx2]
                r2 = r_array[2, idx1] - r_array[2, idx2]
                r3 = r_array[3, idx1] - r_array[3, idx2]

                rⱼₖ = sqrt(r1^2 + r2^2 + r3^2)

                r̂1 = r1 / rⱼₖ
                r̂2 = r2 / rⱼₖ 
                r̂3 = r3 / rⱼₖ 

                r̂r̂11 = r̂1 * r̂1
                r̂r̂12 = r̂1 * r̂2
                r̂r̂13 = r̂1 * r̂3
                r̂r̂21 = r̂2 * r̂1
                r̂r̂22 = r̂2 * r̂2
                r̂r̂23 = r̂2 * r̂3
                r̂r̂31 = r̂3 * r̂1
                r̂r̂32 = r̂3 * r̂2
                r̂r̂33 = r̂3 * r̂3

                term1 = exp(1im * k * rⱼₖ) / (rⱼₖ) 
                term2 = (1im*k*rⱼₖ .- 1)/ rⱼₖ^2
                k2 = k^2

                A_[3*idx1-2, 3*idx2-2] = term1 * (k2 * (r̂r̂11 - 1) + term2 * (3 * r̂r̂11 - 1))
                A_[3*idx1-2, 3*idx2-1] = term1 * (k2 * r̂r̂12 + term2 * (3 * r̂r̂12))
                A_[3*idx1-2, 3*idx2] = term1 * (k2 * r̂r̂13 + term2 * (3 * r̂r̂13 ))
                A_[3*idx1-1, 3*idx2-2] = term1 * (k2 * r̂r̂21 + term2 * (3 * r̂r̂21 ))
                A_[3*idx1-1, 3*idx2-1] = term1 * (k2 * (r̂r̂22 - 1) + term2 * (3 * r̂r̂22 - 1))
                A_[3*idx1-1, 3*idx2] = term1 * (k2 * r̂r̂23 + term2 * (3 * r̂r̂23 ))
                A_[3*idx1, 3*idx2-2] = term1 * (k2 * r̂r̂31 + term2 * (3 * r̂r̂31))
                A_[3*idx1, 3*idx2-1] = term1 * (k2 * r̂r̂32 + term2 * (3 * r̂r̂32))
                A_[3*idx1, 3*idx2] = term1 * (k2 * (r̂r̂33 - 1) + term2 * (3 * r̂r̂33 - 1))
            end
        end
        return nothing
    end
    @inbounds @cuda threads=num_threads blocks=cld(length(Dipolelist) * length(Dipolelist), num_threads) calA(k, r_array, α, A_, length(Dipolelist))
    return A_
end

function CouplingTensorMatGPU(C::Container)
    return CouplingTensorMatGPU(C.k, C.Dipoles)
end

function GreenTensorMat(k, Dipolelist :: Array{Dipole}, Recorderlist :: Array{Recorder}, GreenFunction::Function = FreeSpaceGreenTensor)
    
    𝔸 = zeros(ComplexF64, (3*length(Recorderlist), 3*length(Dipolelist)))
    @inbounds Threads.@threads for (j,D) in collect(enumerate(Dipolelist))
         for (i,R) in collect(enumerate(Recorderlist))
            GreenDtoR = GreenFunction(k, D, R)
            𝔸[3*i-2:3*i, 3*j-2:3*j] = GreenDtoR
        end
    end       
    return 𝔸
end

function GreenTensorMat(C::Container, Recorderlist :: Array{Recorder}, GreenFunction=FreeSpaceGreenTensor)
    return GreenTensorMat(C.k, C.Dipoles, Recorderlist, GreenFunction)
end

function GreenTensorMatGPU(k, Dipolelist :: Array{Dipole}, Recorderlist :: Array{Recorder}, num_threads=512)
    Recorderlist = vec(Recorderlist)
    II = CuArray(Matrix{ComplexF64}(I, 3,3))
    r1_array = CuArray(Matrix{ComplexF64}(reduce(hcat, getproperty.(Dipolelist, :pos))))
    r2_array = CuArray(Matrix{ComplexF64}(reduce(hcat, getproperty.(Recorderlist, :pos))))
    A_ = CUDA.zeros(ComplexF64, length(Recorderlist)*3, length(Dipolelist)*3)

    function calA(k, r1_array, r2_array, A_, n_d, n_r)
        i = (threadIdx().x + (blockIdx().x - 1) * blockDim().x)
        if i <= n_d * n_r
            idx1 = (i-1) ÷ n_d + 1
            idx2 = (i-1) % n_d + 1
            r1 = r2_array[1, idx1] - r1_array[1, idx2]
            r2 = r2_array[2, idx1] - r1_array[2, idx2]
            r3 = r2_array[3, idx1] - r1_array[3, idx2]

            rⱼₖ = sqrt(r1^2 + r2^2 + r3^2)

            r̂1 = r1 / rⱼₖ
            r̂2 = r2 / rⱼₖ 
            r̂3 = r3 / rⱼₖ 

            r̂r̂11 = r̂1 * r̂1
            r̂r̂12 = r̂1 * r̂2
            r̂r̂13 = r̂1 * r̂3
            r̂r̂21 = r̂2 * r̂1
            r̂r̂22 = r̂2 * r̂2
            r̂r̂23 = r̂2 * r̂3
            r̂r̂31 = r̂3 * r̂1
            r̂r̂32 = r̂3 * r̂2
            r̂r̂33 = r̂3 * r̂3

            term1 = - exp(1im * k * rⱼₖ) / (rⱼₖ)
            term2 = (1im*k*rⱼₖ .- 1)/ rⱼₖ^2
            k2 = k^2

            A_[3*idx1-2, 3*idx2-2] = term1 * (k2 * (r̂r̂11 - 1) + term2 * (3 * r̂r̂11 - 1))
            A_[3*idx1-2, 3*idx2-1] = term1 * (k2 * r̂r̂12 + term2 * (3 * r̂r̂12))
            A_[3*idx1-2, 3*idx2] = term1 * (k2 * r̂r̂13 + term2 * (3 * r̂r̂13 ))
            A_[3*idx1-1, 3*idx2-2] = term1 * (k2 * r̂r̂21 + term2 * (3 * r̂r̂21 ))
            A_[3*idx1-1, 3*idx2-1] = term1 * (k2 * (r̂r̂22 - 1) + term2 * (3 * r̂r̂22 - 1))
            A_[3*idx1-1, 3*idx2] = term1 * (k2 * r̂r̂23 + term2 * (3 * r̂r̂23 ))
            A_[3*idx1, 3*idx2-2] = term1 * (k2 * r̂r̂31 + term2 * (3 * r̂r̂31))
            A_[3*idx1, 3*idx2-1] = term1 * (k2 * r̂r̂32 + term2 * (3 * r̂r̂32))
            A_[3*idx1, 3*idx2] = term1 * (k2 * (r̂r̂33 - 1) + term2 * (3 * r̂r̂33 - 1))
        end
        return nothing
    end
    @inbounds @cuda threads=num_threads blocks=cld(length(Dipolelist) * length(Recorderlist), num_threads) calA(k, r1_array, r2_array, A_, length(Dipolelist), length(Recorderlist))
    return A_
end

function GreenTensorMatGPU(C::Container, Recorderlist :: Array{Recorder})
    return GreenTensorMatGPU(C.k, C.Dipoles, Recorderlist)
end

function CalA(k, Dipolelist :: Array{Dipole}; device="gpu", num_threads=512)
    if device == "gpu"
        return CouplingTensorMatGPU(k, Dipolelist, num_threads)
    else
        return CouplingTensorMat(k, Dipolelist)
    end
end

function CalGreen(k, Dipolelist :: Array{Dipole}, Recorderlist :: Array{Recorder}; device="gpu", num_threads=512, GreenFunction=FreeSpaceGreenTensor)
    if device == "gpu"
        return GreenTensorMatGPU(k, Dipolelist, Recorderlist, num_threads)
    else
        return GreenTensorMat(k, Dipolelist, Recorderlist, GreenFunction)
    end
end

function CalA(C; device="gpu", num_threads=512)
    return CalA(C.k, C.Dipoles, device=device, num_threads = 512)
end

function CalGreen(C::Container, RecArray::Array{Recorder}; device="gpu", GreenFunction=FreeSpaceGreenTensor)
    G = CalGreen(C.k, C.Dipoles, RecArray; device=device, GreenFunction=GreenFunction)
    return G
end

function LinSolveGPU(A::CuArray, b::CuArray)
    AA, b = copy(A), copy(b)
    AA, ipiv = CUDA.CUSOLVER.getrf!(AA)
    x = CUDA.CUSOLVER.getrs!('N', AA, ipiv, b)
    return x
end

function CalPolarization(C::Container; device="gpu", num_threads = 512)

    𝔸 = CalA(C; device=device)
    Einc = get_Einc(C)

    if device == "cpu"
        P = 𝔸 \ Einc

    elseif device == "gpu"
        Einc = CuArray{ComplexF64}(Einc)
        𝔸 = CuArray{ComplexF64}(𝔸)
        P = Vector(LinSolveGPU(𝔸, Einc))
    end

    for (idx, dipole) in enumerate(C.Dipoles)
        dipole.P = P[3*idx-2:3*idx] 
    end

    return P
end 

function _FarFieldWithNoScattering(C::Container, RecArray::Array{Recorder}; device="gpu", GreenFunction = FreeSpaceGreenTensor)
    Recvec = vec(RecArray)
    Einc = get_Einc(C)
    α = get_α(C)
    Green = CalGreen(C, Recvec; device=device, GreenFunction=GreenFunction)

    if device == "gpu"
        Einc = CuArray{ComplexF64}(Einc)
        α = CuArray{ComplexF64}(α)
        Green = CuArray{ComplexF64}(Green)
    end

    Far =  Green * (α .* Einc)
    for (i, Rec) in enumerate(Recvec)
        CUDA.@allowscalar Rec.E += SVector{3, ComplexF64}(Far[3*i-2:3*i])
    end
end

function _FarFieldWithScattering(C::Container, RecArray::Array{Recorder}; device="gpu")
    Recvec = vec(RecArray)
    P = CuArray{ComplexF64}(get_P(C))
    Green = CuArray{ComplexF64}(CalGreen(C, Recvec; device=device))
    Far = Green * P
    for (i, Rec) in enumerate(Recvec)
        CUDA.@allowscalar Rec.E += SVector{3, ComplexF64}(Far[3*i-2:3*i])
    end
end

function CalFarField(C::Container, RecArray::Array{Recorder}, mode::String, device="gpu", GreenFunction = FreeSpaceGreenTensor)
    if mode == "sca"
        _FarFieldWithScattering(C, RecArray; device=device)
    elseif mode == "inc"
        _FarFieldWithNoScattering(C, RecArray; device=device, GreenFunction=GreenFunction)
    else
        error("mode should be 'sca' or 'inc'")
    end
end

function FarFieldPoyntingVector(Rec::Recorder)
    n̂ = unitvec(Rec.pos)
    H = 1/Z0*cross(n̂, Rec.E)
    return 1/2 * real(cross(Rec.E, conj.(H)))
end

function FarFieldPoyntingVector(Recorderlist::Array{Recorder})
    Slist = zeros(size(Recorderlist))
    for (i, rec) in enumerate(Recorderlist)
        S = FarFieldPoyntingVector(rec)
    end 
    return Slist
end

function FarFieldPower(C::Container, θorulist, ϕorvlist; mode="sca", r=1e6, rectype="θϕ")
    if rectype == "θϕ"
        Recorders = SphericalRecorder(θorulist::Vector, ϕorvlist::Vector, R=r)
    elseif rectype == "uv"
        Recorders = SphericalUVRecorder(θorulist::Vector, ϕorvlist::Vector, R=r)
    end
    CalFarField(C, Recorders, mode)
    Slist = FarFieldPoyntingVector(Recorders)
    return Slist
end

function FarFieldPower(C::Container, dθ::Number, dϕ::Number; θrange=(0, π), ϕrange=(0, 2π), mode="sca", r=1e6)
    θlist = list(collect(range(θrange[1], θrange[2], step=dθ)))
    ϕlist = list(collect(range(ϕrange[1], ϕrange[2], step=dϕ)))
    Slit = FarFieldPower(C, θlist, ϕlist; mode=mode, r=r)
    return r^2 * Slist
end
