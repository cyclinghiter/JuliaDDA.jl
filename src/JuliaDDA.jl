module JuliaDDA

using Base
using Base.Threads
using LinearAlgebra
using CUDA
using StaticArrays
using PlotlyJS
using Statistics
using Random

export c, ϵ0, μ0, Z0
export Container, remove, get_Einc, get_P, get_α
export CalA, CalGreen, CalPolarization, CalFarField, FarFieldPoyntingVector, FarFieldPower
export Dipole, reset_dipoles
export displacement, distance, unitvec, CarttoSpherical, SphericaltoCart, RotationMatrix
export FreeSpaceScalarGreen, CouplingTensor, GreenTensor
export Plot3DPower, PlotDirectivity, PlotScatterers, PlotDifferentialCrossSection
export Clausius_Mossoti_Polarizability, Lattice_Dispersion_Polarizability
export Recorder, SphericalRecorder, SphericalUVRecorder, PlaneRecorder, PoyntingVector, Power
export Source, CalEinc, PlaneWave, DipoleSource
export Structure, CustomStructure, Sphere, Box, Plane
export Pan, Rotate

include("Polarizability.jl")
include("Dipole.jl")
include("Structure.jl")
include("Recorder.jl")
include("Container.jl")
include("Function.jl")
include("Core.jl")
include("Constants.jl")
include("Green.jl")
include("Transform.jl")
include("Source.jl")
include("Plot.jl")

end

   