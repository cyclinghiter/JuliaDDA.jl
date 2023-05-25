using JuliaDDA
using StaticArrays
using LinearAlgebra
using PlotlyJS
using CUDA

freqs = 2.4e9
λ = c / freqs
k = 2π / λ

SMM = Plane(2, 2, 0, 30, 30, 1.01, 1, p_mode=:LDR, k=k)
# Src = DipoleSource(k, SA[0, 0, -0.1], SA[0, 1, 0])
Src = PlaneWave(k .* SA[0, 0, 1], SA[0, 1, 0])

losses = [] 
αhist = []
for i in range(1, 1000)

    reset_dipoles(SMM)
    C = Container(k)
    push!(C, SMM)
    CalEinc(C, Src)
    CalPolarization(C)

    Py = SMM.Py
    αy = SMM.αy

    E_SMM = [II for II in Py ./ αy]
    # E_SMM = SMM.Eincy

    Rec = SphericalUVRecorder(28, 28, R=10)
    urange = LinRange(-1, 1, 28)
    vrange = LinRange(-1, 1, 28)

    CalFarField(C, Rec, "sca")
    for (i,u) in enumerate(urange)
        for (j,v) in enumerate(vrange)
            if u^2 + v^2 >= 1
                Rec[i,j].E = SA[0, 0, 0]
            end
        end
    end

    target = zeros(28, 28)
    # target[8, 8] = 1
    target[9:11, 10:20] .= 1
    # target[19:21, 10:20] .= 1

    Ey = (Rec.Ey) / sum(norm.((Rec.Ey)))

    dLdE = reshape(CuArray(vec(conj.(Ey .- target))), (1,784))
    dEdP = CalGreen(C, vec(Rec), device="gpu")[2:3:end, 2:3:end]
    dPdα = CuArray(reduce(vcat, E_SMM))

    dLdα = (vec(dLdE * dEdP) .* dPdα)
    
    for (i, d) in enumerate(SMM.Dipoles)
        d.α -= 1e-11 .* conj.(SA[dLdα[i], dLdα[i], dLdα[i]])
        d.α -= 0.2 * (d.α)
    end

    if i % 10 == 1
        p1 = plot(heatmap(z=abs.(Ey)))
        p2 = plot(heatmap(z=abs.(Ey .- target)))
        p3 = plot(scatter(y=losses))
        p4 = plot(scatter(y=αhist))
        p5 = plot(heatmap(z=real.(reshape(SMM.αy, (30, 30)))))
        
        p6 = plot(heatmap(z=abs.(Rec.Ey)))
        display([p1 p2 ; p3 p4; p5 p6])
    end

    Loss = sum(norm.(dLdE))
    push!(losses, Loss)
    push!(αhist, (sum(abs.(SMM.αy)) / length(SMM.αy)))

end
