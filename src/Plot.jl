function Plot3DPower(SphericalRecorder::Array{Recorder};
                     opacity = 1,
                     axis = true,
                     log = false,
                     Po_in = 1)

  Po = Power(SphericalRecorder)

  if log
    Po = @. 10 * clamp(log10(Po / Po_in), 0, Inf64)
  end

  x = Po .* SphericalRecorder.x
  y = Po .* SphericalRecorder.y
  z = Po .* SphericalRecorder.z

  layout = Layout(scene=attr(aspectmode="data"),
  xaxis=attr(visible=axis), 
  yaxis=attr(visible=axis),
  zaxis=attr(visible=axis),)

  p = surface(
  x = x,
  y = y,
  z = z,
  opacity = opacity,
  surfacecolor = Po,
  showaxes = axis,
  colorscale = "Viridis"
  )
  display(plot(p, layout))
end

function PlotDirectivity(SphericalRecorder::Array{Recorder};
                         opacity = 1,
                         axis = true,
                         log = true
                         )

  Po = Power(SphericalRecorder)
  # directivity of antenna
  Po = Po ./ mean(Po)

  if log
    Po = @. 10 * clamp(log10(Po), 0, Inf64)
  end

  x = Po .* SphericalRecorder.x
  y = Po .* SphericalRecorder.y
  z = Po .* SphericalRecorder.z

  layout = Layout(scene=attr(aspectmode="data"),
  xaxis=attr(visible=axis), 
  yaxis=attr(visible=axis),
  zaxis=attr(visible=axis),)

  p = surface(
  x = x,
  y = y,
  z = z,
  opacity = opacity,
  surfacecolor = Po,
  showaxes = axis,
  colorscale = "Viridis"
  )
  display(plot(p, layout))
end

function PlotDifferentialCrossSection(C::Container, nθ, nϕ; scale="log", r=1e10)
  θlist = collect(LinRange(0, π, nθ))
  ϕlist = collect(LinRange(0, 2π, nϕ))
  Rec = SphericalRecorder(θlist, ϕlist, R=r)
  CalFarField(C, Rec, "sca")
  Po = Power(Rec)
  Po = Po ./ mean(Po)
  X = mean(Po, dims=2)
  if scale == "log"
    X = log10.(X)
  end
  display(plot(scatter(x = θlist, y = X[:,1])))
end

function PlotScatterers(C::T, quantity::Symbol; 
                        marker_size = 5,
                        plot_mode = :norm) where T<:Union{Container, Structure}
  Positions = getproperty.(C.Dipoles, :pos)
  Positions = reduce(hcat, Positions)

  color_m = getproperty.(C.Dipoles, quantity)
  color_m = if plot_mode == :norm
             norm.(color_m)
            elseif plot_mode == :real
              real.(color_m)
            elseif plot_mode == :imag
              imag.(color_m)
            elseif plot_mode == :abs
              abs.(color_m)
            end

  p = scatter3d(
        x=Positions[1,:],
        y=Positions[2,:],
        z=Positions[3,:],
        mode="markers",
        type="scatter3d",
        marker = attr(
                  color=color_m,
                  colorscale="Viridis",
                  size=marker_size
                      )
              )
  display(plot(p))
end 
