# JuliaDDA

[![Build Status](https://github.com/cyclinghiter/JuliaDDA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cyclinghiter/JuliaDDA.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Usage

This package is based on the

*Yurkin, M. A., & Hoekstra, A. G. (2007). The discrete dipole approximation: an overview and recent developments. Journal of Quantitative Spectroscopy and Radiative Transfer, 106(1-3), 558-589.*

# Testing

    using JuliaDDA
    using StaticArrays

    k = 1
    ka = 7
    a = ka / k

    C = Container(k)
    Object = Sphere(a, 20, 5, 1)
    # Object = Box(a, a, a, 10, 10, 10, 2+1.5im, 1)
    Ojbect = Rotate(Object, pi/4, n̂=SA[0, 0, 1])

    Src = PlaneWave(SA[0, 0, -1], SA[1, 1im, 0] ./ sqrt(2))
    Rec = SphericalRecorder(50, 50, R=100, mode="full")
    push!(C, Object)
    CalEinc(C, Src)
    CalPolarization(C)
    CalFarField(C, Rec, "sca")

    Volume = 4/3 * pi * a^3
    Po_in = ϵ0 * Volume 

    PlotScatterers(C, :Py, plot_mode=:real)
    Plot3DPower(Rec, log=true, Po_in = Po_in)
    PlotDifferentialCrossSection(C, 50, 50, scale="log", r=100)
