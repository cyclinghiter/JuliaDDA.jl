using JuliaDDA
using Test
using StaticArrays

@testset "JuliaDDA.jl" begin
    # Write your tests here.
    @test JuliaDDA.c isa Int64
    @test JuliaDDA.ϵ0 isa Float64
    @test JuliaDDA.μ0 isa Float64
    @test JuliaDDA.Z0 isa Float64
    @test JuliaDDA.Dipole(1,1,1) isa JuliaDDA.Dipole
    @test JuliaDDA.Recorder(SA[1,1,1], SA[1,1,1]) isa JuliaDDA.Recorder
    @test JuliaDDA.Container(1) isa JuliaDDA.Container
    @test JuliaDDA.Sphere(1,10,1,1) isa JuliaDDA.Structure
end
