module DataMethods

using PythonCall
using DataFrames, DataFramesMeta
using CairoMakie, AlgebraOfGraphics
using CSV
using PolygonOps
using StaticArrays


@py begin
    import galstreams
    import astropy.coordinates as coord
    import astropy.units as u
    import ezmist
    import gala.coordinates as galacoord
    # import pyia
end

"""Data curation."""
function curation!(df::DataFrame)::Nothing
    df.color = df.bp - df.rp
    df.pmra_error_rel = df.pmra_error./df.pmra
    @subset!(df, (:pmra_error_rel .< 0.5))
    df.pmdec_error_rel = df.pmdec_error./df.pmdec
    @subset!(df, (:pmdec_error_rel .< 0.5))
    df.parallax_rel_error = df.parallax_error./df.parallax
    @subset!(df, (:parallax_rel_error .< 0.5))
    df.D .= 1.0 ./ df.parallax
    @subset!(df, :D .> 0 )
    return nothing
end

"""CMD filtering."""
function filter_cmd!(df_stream::DataFrame, df_iso::DataFrame)::Nothing
    phase_mask = 0 .<= df_iso.phase .< 3
    df_iso = df_iso[phase_mask,:]
    df_iso.color = df_iso.Gaia_BP_EDR3 - df_iso.Gaia_RP_EDR3
    df_iso.left = df_iso.color .- 0.1
    df_iso.right = df_iso.color .+ 0.1
    pol_x = vcat(df_iso.left, reverse(df_iso.right), df_iso.left[1])
    temp = df_iso.Gaia_G_EDR3
    pol_y = vcat(temp, reverse(temp), temp[1])
    polygon = SVector.(pol_x, pol_y)

    df_stream.distmod = pyconvert(Vector{Float64},coord.Distance(Py(df_stream.D)*u.kpc).distmod.value)
    df_stream.g_abs = df_stream.g - df_stream.distmod
    points = [[df_stream.color[i], df_stream.g_abs[i]] for i in 1:nrow(df_stream) ]
    inside = [inpolygon(p, polygon; in=true, on=false, out=false) for p in points]
    df_stream = df_stream[inside,:]
    return nothing
end

"""Load Galstreams data (one single stream track)."""
function load_stream_track(name_t::String)
    mwsts = galstreams.MWStreams(verbose=false, implement_Off=true)
    resumen = mwsts.summary |> PyPandasDataFrame |> DataFrame
    bool_on = resumen.On .== true
    𝒯 = resumen.TrackName[bool_on]
    track = mwsts[name_t]
    frame = track.stream_frame
    self_coords = track.track.transform_to(frame)
    ϕ₁ = pyconvert(Vector{Float64}, self_coords.phi1.value)
    ϕ₂ = pyconvert(Vector{Float64}, self_coords.phi2.value)
    D = pyconvert(Vector{Float64}, self_coords.distance.value)
    μ₁cosϕ₂ = pyconvert(Vector{Float64}, self_coords.pm_phi1_cosphi2.value)
    μ₂ = pyconvert(Vector{Float64}, self_coords.pm_phi2.value)
    Vᵣ = pyconvert(Vector{Float64}, self_coords.radial_velocity)
    df_track = DataFrame(ra=pyconvert(Vector{Float64},track.track.ra.deg),
                        dec=pyconvert(Vector{Float64},track.track.dec.deg),
                        ϕ₁=ϕ₁, ϕ₂=ϕ₂, μ₁cosϕ₂=μ₁cosϕ₂, μ₂=μ₂, D=D, Vᵣ=Vᵣ)
    return df_track, frame
end

"""Download stellar isochrones"""
function get_isochrone(age::Float64, metal::Float64,
                           phot::String, age_scale::String)::DataFrame
    println("Note that MIST uses [FeH] (not Z abundance).")
    df = ezmist.get_one_isochrone(age=age, FeH=metal, v_div_vcrit=0.0,
                    age_scale=age_scale, output_option="photometry",
                    output=phot, Av_value=0.0).to_pandas()|> PyPandasDataFrame |> DataFrame
    return df
end


"""Compute stream stars' self coordinates and add to dataframe."""
function compute_in_selfCoords!(df::DataFrame, frame::Py)::Nothing
    sky_coords = coord.SkyCoord(ra=Py(df.ra)*u.deg, dec=Py(df.dec)*u.deg, pm_ra_cosdec=Py(df.pmra)*u.mas/u.yr, pm_dec=Py(df.pmdec)*u.mas/u.yr, frame="icrs")
    self_coords = sky_coords.transform_to(frame)
    df.ϕ₁ = pyconvert(Vector{Float64}, self_coords.phi1.deg)
    df.ϕ₂ = pyconvert(Vector{Float64}, self_coords.phi2.deg)
    df.μ₁cosϕ₂ = pyconvert(Vector{Float64}, self_coords.pm_phi1_cosphi2.value)
    df.μ₂ = pyconvert(Vector{Float64}, self_coords.pm_phi2.value)
    return nothing
end


"""Compare two tracks of Fjörm."""
function compare_tracks(stream₁::String, stream₂::String)
    track₁ = mwsts[stream₁]
    track₂ = mwsts[stream₂]
    println(track₁)
    frame = track.stream_frame
    self_coords₁ = track₁.track.transform_to(frame)
    self_coords₂ = track₂.track.transform_to(frame)
    ϕ₁ = pyconvert(Vector{Float64}, self_coords₁.phi1.value)
    D₁ = pyconvert(Vector{Float64}, self_coords₁.distance.value)
    ϕ₂ = pyconvert(Vector{Float64}, self_coords₂.phi1.value)
    D₂ = pyconvert(Vector{Float64}, self_coords₂.distance.value)
    df₁ = DataFrame(x=ϕ₁, y=D₁)
    df₂ = DataFrame(x=ϕ₂, y=D₂)
    size_inches = (6*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = (data(df₂)+data(df₁)*visual(color="red"))*mapping(:x=>L"ϕ₁", :y =>L"D")*visual(Lines)
    draw!(fig, plt, axis=(; limits=((-20,nothing),(0, nothing))))
    electrondisplay(fig)
end

function mask_gc!(df_stream, df_gc)
    for i in 1:nrow(df_gc)
        Δra = df_stream.ra.-df_gc.ra[i]
        Δdec = df_stream.dec.-df_gc.dec[i]
        bool_gc = sqrt.(Δra.^2+Δdec.^2) .> 0.5
        @subset!(df_stream, collect(bool_gc))
    end
end

"""Filter with stream track on the sky."""
function filter_stream_on_sky!(df_stars::DataFrame, df_track::DataFrame, width::Float64)::DataFrame
    up = df_track.ϕ₂.+width
    down =  df_track.ϕ₂.-width
    poly_ϕ₁ = vcat(df_track.ϕ₁, reverse(df_track.ϕ₁), df_track.ϕ₁[1])
    poly_ϕ₂ = vcat(down, reverse(up), down[1])
    polygon = SVector.(poly_ϕ₁, poly_ϕ₂)
    points = [[df_stars.ϕ₁[i], df_stars.ϕ₂[i]] for i in 1:nrow(df_stars) ]
    inside = [inpolygon(p, polygon; in=true, on=false, out=false) for p in points]
    @subset!(df_stars, collect(inside))
end

"""Filter with stream on μ-space."""
function filter_stream_μ_space!(df_stars::DataFrame, df_track::DataFrame, Δμ::Float64)
    left = df_track.μ₁cosϕ₂.-Δμ
    right =  df_track.μ₁cosϕ₂.+Δμ
    poly_y = vcat(df_track.μ₂, reverse(df_track.μ₂), df_track.μ₂[1])
    poly_x = vcat(left, reverse(right), left[1])
    polygon = SVector.(poly_x, poly_y)
    points = [[df_stars.μ₁cosϕ₂[i], df_stars.μ₂[i]] for i in 1:nrow(df_stars) ]
    inside = [inpolygon(p, polygon; in=true, on=false, out=false) for p in points]
    @subset!(df_stars, collect(inside))
end

"""Non-mutating filter with stream track in any of its dimensions."""
function filter_with_track(df_stars::DataFrame, df_track::DataFrame, S::Symbol, σ::Float64)::DataFrame
    if S == :ϕ₂
        q🌠 = df_stars.ϕ₂
        q_track = df_track.ϕ₂
    elseif S == :D
        q🌠 = 1.0 ./ df_stars.parallax
        q_track = df_track.D
    elseif S == :μ₁cosϕ₂
        q🌠 = df_stars.μ₁cosϕ₂
        q_track = df_track.μ₁cosϕ₂
    elseif S == :μ₂
        q🌠 = df_stars.μ₂
        q_track = df_track.μ₂
    elseif S == :Vᵣ
        q🌠 = df_stars.radial_velocity
        q_track = df_track.Vᵣ
    end
    up = q_track .+ σ
    down =  q_track .- σ
    poly_ϕ₁ = vcat(df_track.ϕ₁, reverse(df_track.ϕ₁), df_track.ϕ₁[1])
    poly_q = vcat(down, reverse(up), down[1])
    polygon = SVector.(poly_ϕ₁, poly_q)
    points = [[df_stars.ϕ₁[i], q🌠[i]] for i in 1:nrow(df_stars) ]
    inside = [inpolygon(p, polygon; in=true, on=false, out=false) for p in points]
    return @subset(df_stars, collect(inside))
end

"""Mutating filter with stream track in any of its dimensions."""
function filter_with_track!(df_stars::DataFrame, df_track::DataFrame, S::Symbol, σ::Float64)::Nothing
    if S == :ϕ₂
        q🌠 = df_stars.ϕ₂
        q_track = df_track.ϕ₂
    elseif S == :D
        q🌠 = 1.0 ./ df_stars.parallax
        q_track = df_track.D
    elseif S == :μ₁cosϕ₂
        q🌠 = df_stars.μ₁cosϕ₂
        q_track = df_track.μ₁cosϕ₂
    elseif S == :μ₂
        q🌠 = df_stars.μ₂
        q_track = df_track.μ₂
    elseif S == :Vᵣ
        q🌠 = df_stars.radial_velocity
        q_track = df_track.Vᵣ
    end
    up = q_track .+ σ
    down =  q_track .- σ
    poly_ϕ₁ = vcat(df_track.ϕ₁, reverse(df_track.ϕ₁), df_track.ϕ₁[1])
    poly_q = vcat(down, reverse(up), down[1])
    polygon = SVector.(poly_ϕ₁, poly_q)
    points = [[df_stars.ϕ₁[i], q🌠[i]] for i in 1:nrow(df_stars) ]
    inside = [inpolygon(p, polygon; in=true, on=false, out=false) for p in points]
    @subset!(df_stars, collect(inside))
    return nothing
end


"""Reflex Correction."""
function reflex_correct!(df::DataFrame, frame::Py)
    len = length(df.ϕ₁)
    sky_coords = coord.SkyCoord(phi1=Py(df.ϕ₁)*u.deg, phi2=Py(df.ϕ₂)*u.deg, pm_phi1_cosphi2=Py(df.μ₁cosϕ₂)*u.mas/u.yr, pm_phi2=Py(df.μ₂)*u.mas/u.yr, distance=Py(fill(1.,len))*u.kpc, radial_velocity=Py(fill(0.,len))*u.km/u.s, frame=frame)
    vsun = coord.CartesianDifferential(Py([11., 245., 7.])*u.km/u.s)
    gc_frame = coord.Galactocentric(galcen_v_sun=vsun, z_sun=0*u.pc)
    sky_coords_corr = galacoord.reflex_correct(sky_coords, gc_frame)
    df.μ₁cosϕ₂_corr = pyconvert(Vector{Float64}, sky_coords_corr.pm_phi1_cosphi2.value)
    df.μ₂_corr = pyconvert(Vector{Float64}, sky_coords_corr.pm_phi2.value)
    return nothing
end

end
