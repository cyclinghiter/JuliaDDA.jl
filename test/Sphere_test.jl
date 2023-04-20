using JuliaDDA
using StaticArrays

k = 1
ka = 2
a = ka / k

C = Container(k)
Object = Sphere(a, 10, 5, 1)
# Object = Box(a, a, a, 10, 10, 10, 2+1.5im, 1)
Ojbect = Rotate(Object, pi/4, n̂=SA[0, 0, 1])

Src = PlaneWave(SA[0, 0, -1], SA[1, 0, 0] ./ sqrt(2))
Rec = SphericalRecorder(50, 50, R=100, mode="full")
push!(C, Object)
CalEinc(C, Src)
CalPolarization(C)
CalFarField(C, Rec, "sca")

Volume = 4/3 * pi * a^3
Po_in = 1 / 2 *ϵ0 * Volume 

PlotScatterers(C, :Px, plot_mode=:norm)
Plot3DPower(Rec, log=true, Po_in = Po_in)
PlotDifferentialCrossSection(C, 50, 50, scale="log", r=100)