# JuliaDDA

[![Build Status](https://github.com/cyclinghiter/JuliaDDA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cyclinghiter/JuliaDDA.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Usage

This package is based on the reference 

*Draine, B. T., & Flatau, P. J. (1994). Discrete-dipole approximation for scattering calculations. Josa a, 11(4), 1491-1499.*

# Testing

    using JuliaDDA
    using StaticArrays

    # Wave parameters
    k = 1
    ka = 7
    a = ka / k

    # Define Container for load objects 
    C = Container(k)

    # Generate Sphere object 
    Object = Sphere(a, 20, 2+1.5im, 1)
    # You can also Rotate or Pan the Object
    # Ojbect = Rotate(Object, pi/4, n̂=SA[0, 0, 1])

    # plane wave
    # propagation axis :z
    # polarization axis : x
    Src = PlaneWave(SA[0, 0, 1], SA[1, 0, 0]) 

    # Define Recorder to save Electric field at certain position
    Rec = SphericalRecorder(50, 50, R=100, mode="full")

    # push object to the container
    push!(C, Object)

    # Calculate the incident field
    CalEinc(C, Src)

    # Calculate the Polarization of the dipoles
    CalPolarization(C)

    # Calculate the Farfield at recorder positions
    CalFarField(C, Rec, "sca")

    Volume = 4/3 * pi * a^3
    Po_in = ϵ0 * Volume 

    # Plot scatteres in the Container (backend : PlotlyJS)
    PlotScatterers(C, :Px, plot_mode=:real)

    # Plot Far field power
    Plot3DPower(Rec, log=true, Po_in = Po_in)

    # Plot DifferentialCrossSection (not accurate yet)
    PlotDifferentialCrossSection(C, 50, 50, scale="log", r=100)
