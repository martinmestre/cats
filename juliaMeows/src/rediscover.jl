"""Rediscover Fjörm stream from Gaia data."""

using PythonCall
using DataFrames, DataFramesMeta
using CairoMakie, AlgebraOfGraphics
using FITSIO
using ElectronDisplay
using CSV
using PolygonOps
using StaticArrays
using Interpolations
using LinearAlgebra

@py begin
    import galstreams
    import matplotlib.pyplot as plt
    import numpy as np
    import astropy.coordinates as coord
    import astropy.units as u
    import astroquery.xmatch as xmatch
    import astroquery.simbad as simbad
    import astroquery.vizier as vizier
    import ezmist
    import pandas as pd
    import pyia
end
# %%

"Little functions."

"""Plot sky histogram."""
function plot_sky_histo(df::DataFrame, file::String)
    size_inches = (5*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = data(df)*mapping(:ra =>L"RA [$°$]", :dec=>L"Dec [$°$]")*
                            histogram(bins=200)
    ag=draw!(fig, plt, axis=(;limits=((nothing,nothing),(nothing,nothing))))
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
end

function plot_sky_histo_gc(df::DataFrame, file::String, df_gc)
    size_inches = (5*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = (data(df)*histogram(bins=200)+data(df_gc)*visual(color="red"))*mapping(:ra =>L"RA [$°$]", :dec=>L"Dec [$°$]")
    plt_M68 = data(df_gc)*mapping(:ra =>L"RA [$°$]", :dec=>L"Dec [$°$]")*visual(color="black")
    ag = draw!(fig, plt, axis=(;limits=((180,270),(-30,80))))
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
end

"""Plot sky histogram in stream frame."""
function plot_sky_histo_selfFrame(df::DataFrame, file::String, df_track::DataFrame)
    size_inches = (5*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = (data(df)*AlgebraOfGraphics.density()+data(df_track))*mapping(:ϕ₁ =>L"ϕ_1 [°]", :ϕ₂=>L"ϕ_2 [°]")
                            # histogram(bins=100)
    ag = draw!(fig, plt, axis=(;limits=((nothing,nothing),(nothing,nothing))))
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
end

"""Plot sky scatter in stream frame."""
function plot_sky_scatter_selfFrame(df::DataFrame, file::String, df_track::DataFrame)
    size_inches = (5*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = (data(df)*visual(markersize=3)+data(df_track)*visual(markersize=1,color="red"))*mapping(:ϕ₁ =>L"ϕ_1 [°]", :ϕ₂=>L"ϕ_2 [°]")
    ag = draw!(fig, plt, axis=(;limits=((nothing,nothing),(nothing,nothing))))
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
end

"""Plot CMD histogram."""
function plot_cmd_histo(df::DataFrame, file::String)
    size_inches = (5*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = data(df)*mapping(:color=>L"$BP-RP$ [Mag]", :g_abs=>L"$G$ [Mag]")*
                                    histogram(bins=300)
    ag = draw!(fig, plt, axis=(;yreversed=true))
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
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

"""Plot single isochrone."""
function plot_isochrone(df::DataFrame)
    size_inches = (3*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = data(df)*mapping(:color=>L"BP-RP", :Gaia_G_EDR3 =>L"G")*visual(Lines)
    draw!(fig, plt, axis=(;yreversed=true))
    electrondisplay(fig)
#    save(file, fig, pt_per_unit=1)
end

"""Plot isochone plus data."""
function plot_isochrone_data(df_iso::DataFrame, df_s::DataFrame, file::String)
    size_inches = (3*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt_s = data(df_s)*mapping(:color=>L"$BP-RP$ [Mag]", :g_abs=>L"$G$ [Mag]")*histogram(bins=100)
    plt_iso = data(df_iso)*mapping(:color=>L"$BP-RP$ [Mag]", :Gaia_G_EDR3=>L"$G$ [Mag]")*visual(Lines,color="red")
    plt_iso_bord = data(df_iso)*mapping([:left,:right].=>L"$BP-RP$ [Mag]", :Gaia_G_EDR3=>L"$G$ [Mag]")*visual(Lines,color="black")
    ag = draw!(fig, plt_s+plt_iso+plt_iso_bord, axis=(;yreversed=true))#, limits=((0,1.5),(14,22))))
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
end

"""Compute stream stars' self coordinates and add to dataframe."""
function compute_in_selfCoords!(df, frame)
    sky_coords = coord.SkyCoord(ra=Py(df.ra)*u.deg, dec=Py(df.dec)*u.deg, pm_ra_cosdec=Py(df.pmra)*u.mas/u.yr, pm_dec=Py(df.pmdec)*u.mas/u.yr, frame="icrs")
    self_coords = sky_coords.transform_to(frame)
    println(self_coords)
    df.ϕ₁ = pyconvert(Vector{Float64}, self_coords.phi1.deg)
    df.ϕ₂ = pyconvert(Vector{Float64}, self_coords.phi2.deg)
    df.μ₁cosϕ₂ = pyconvert(Vector{Float64}, self_coords.pm_phi1_cosphi2.value)
    df.μ₂ = pyconvert(Vector{Float64}, self_coords.pm_phi2.value)
    return nothing
end

"""Plot μ space."""
function plot_μ_window(df::DataFrame, file::String, window::Vector{Vector{Float64}})
    size_inches = (5*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = data(df)*mapping(:pmra =>L"$μ_{RA}$ [mas/yr]", :pmdec=>L"$μ_{Dec}$ [mas/yr]")*
                            histogram(bins=200)
    ag = draw!(fig, plt, axis=(;limits=((window[1][1], window[1][2]),(window[2][1],window[2][2]))))
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
end

function plot_μ(df::DataFrame, file::String)
    size_inches = (5*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = data(df)*mapping(:pmra =>L"$μ_{RA}$ [mas/yr]", :pmdec=>L"$μ_{Dec}$ [mas/yr]")*
                            histogram(bins=500)
    ag = draw!(fig, plt)
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
end

function plot_μ_selfFrame(df::DataFrame, file::String)
    size_inches = (5*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = data(df)*mapping(:μ₁cosϕ₂ =>L"$μ_1cosϕ_2$ [mas/yr]", :μ₂=>L"$μ_2$ [mas/yr]")*
                            histogram(bins=500)
    ag = draw!(fig, plt)
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
end

function plot_μ_selfFrame_window(df::DataFrame, file::String,  window::Vector{Vector{Float64}})
    size_inches = (6*3, 5*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = (data(df)*AlgebraOfGraphics.density())*mapping(:μ₁cosϕ₂ =>L"$μ_1cosϕ_2$ [mas/yr]", :μ₂=>L"$μ_2$ [mas/yr]")
    ag = draw!(fig, plt, axis=(;limits=((window[1][1], window[1][2]),(window[2][1],window[2][2]))))
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
end

function plot_μ_selfFrame_window(df::DataFrame, df_track, file::String,  window::Vector{Vector{Float64}})
    size_inches = (4*3, 4*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = (data(df)*histogram(bins=700)+data(df_track))*mapping(:μ₁cosϕ₂ =>L"$μ_1cosϕ_2$ [mas/yr]", :μ₂=>L"$μ_2$ [mas/yr]")
    ag = draw!(fig, plt, axis=(;limits=((window[1][1], window[1][2]),(window[2][1],window[2][2]))))
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
end

function plot_μ_scatter_selfFrame_window(df::DataFrame, df_track, file::String,  window::Vector{Vector{Float64}})
    size_inches = (3*3, 3*3)
    size_pt = 72 .* size_inches
    fig = Figure(resolution = size_pt, fontsize = 30)
    plt = (data(df)*visual(markersize=3)+data(df_track)*visual(markersize=2,color="red"))*mapping(:μ₁cosϕ₂ =>L"$μ_1cosϕ_2$ [mas/yr]", :μ₂=>L"$μ_2$ [mas/yr]")
    ag = draw!(fig, plt, axis=(;limits=((window[1][1], window[1][2]),(window[2][1],window[2][2]))))
    colorbar!(fig[1,2], ag)
    electrondisplay(fig)
    save(file, fig, pt_per_unit=1)
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
function filter_stream_on_sky(df_stars::DataFrame, df_track::DataFrame, width::Float64)::DataFrame
    up = df_track.ϕ₂.+width
    down =  df_track.ϕ₂.-width
    poly_ϕ₁ = vcat(df_track.ϕ₁, reverse(df_track.ϕ₁), df_track.ϕ₁[1])
    poly_ϕ₂ = vcat(down, reverse(up), down[1])
    polygon = SVector.(poly_ϕ₁, poly_ϕ₂)
    points = [[df_stars.ϕ₁[i], df_stars.ϕ₂[i]] for i in 1:nrow(df_stars) ]
    inside = [inpolygon(p, polygon; in=true, on=false, out=false) for p in points]
    return df_stars = df_stars[inside,:]
end

"""Filter with stream on μ-space."""
function filter_stream_μ_space(df_stars::DataFrame, df_track::DataFrame, Δμ::Float64)::DataFrame
    left = df_track.μ₁cosϕ₂.-Δμ
    right =  df_track.μ₁cosϕ₂.+Δμ
    poly_y = vcat(df_track.μ₂, reverse(df_track.μ₂), df_track.μ₂[1])
    poly_x = vcat(left, reverse(right), left[1])
    polygon = SVector.(poly_x, poly_y)
    points = [[df_stars.μ₁cosϕ₂[i], df_stars.μ₂[i]] for i in 1:nrow(df_stars) ]
    inside = [inpolygon(p, polygon; in=true, on=false, out=false) for p in points]
    return df_stars = df_stars[inside,:]
end

"""Filter with stream track in any of its dimensions."""
function filter_with_track(df_stars::DataFrame, S::Symbol, σ::Float64)::DataFrame
    if S == :ϕ₂
        tᵥ = df_stars.ϕ₂
    elseif S == :D
        tᵥ .= 1.0 ./ df_stars.parallax
    elseif S == :μ₁cosϕ₂
        tᵥ = df_stars.μ₁cosϕ₂
    elseif S == :μ₂
        tᵥ = df_stars.μ₂
    elseif S == :Vᵣ
        tᵥ = df_stars.radial_velocity
    end
end
# %%

"""Opening the file with extinction corrected magnitudes."""
name_s = "Fjorm"
input_file = "../data/corrected/$(name_s)/corr_GaiaDR3-Fjorm-M68-all.fits"
f = FITS(input_file)
df_stream = DataFrame(f[2])
println("Fields: ", names(df_stream))
# %%

"""Plot histogram of the sky."""

plot_sky_histo(df_stream, "sky_$(name_s).png")
# %%

"""Load galstreams data."""
mwsts = galstreams.MWStreams(verbose=false, implement_Off=true)
resumen = mwsts.summary |> PyPandasDataFrame |> DataFrame
bool_on = resumen.On .== true
𝒯 = resumen.TrackName[bool_on]
name_t = "M68-P19"
# name_t = "Fjorm-I21"
track = mwsts[name_t]
frame = track.stream_frame
self_coords = track.track.transform_to(frame)
ϕ₁ = pyconvert(Vector{Float64}, self_coords.phi1.value)
ϕ₂ = pyconvert(Vector{Float64}, self_coords.phi2.value)
D = pyconvert(Vector{Float64}, self_coords.distance.value)
μ₁cosϕ₂ = pyconvert(Vector{Float64}, self_coords.pm_phi1_cosphi2.value)
μ₂ = pyconvert(Vector{Float64}, self_coords.pm_phi2.value)
Vᵣ = pyconvert(Vector{Float64}, self_coords.radial_velocity)
D_interp = linear_interpolation(ϕ₁, D)
df_track = DataFrame(ra=pyconvert(Vector{Float64},track.track.ra.deg),
                     dec=pyconvert(Vector{Float64},track.track.dec.deg),
                     ϕ₁=ϕ₁, ϕ₂=ϕ₂,
                     μ₁cosϕ₂=μ₁cosϕ₂, μ₂=μ₂)
@subset!(df_track, :ϕ₁ .> -20.)
# %%

"""CMD filtering."""
iso_file = "iso_stream_$(name_s).csv"
mf_file = "mf_stream_$(name_s).csv"
filters = "UBVRIplus"
df_iso = get_isochrone(10.0e9, -1.5, filters, "linear")
CSV.write(iso_file, df_iso)
df_iso = DataFrame(CSV.File(iso_file))
phase_mask = 0 .<= df_iso.phase .< 3
df_mask =df_iso[phase_mask,:]
df_mask.color = df_mask.Gaia_BP_EDR3 - df_mask.Gaia_RP_EDR3
df_mask.left = df_mask.color .- 0.15
df_mask.right = df_mask.color .+ 0.15
new_mask = 4. .< df_mask.Gaia_G_EDR3 .< 8.
df_nmask = df_mask[new_mask,:]
CSV.write(mf_file, df_nmask)

pol_x = vcat(df_nmask.left, reverse(df_nmask.right), df_nmask.left[1])
temp = df_nmask.Gaia_G_EDR3
pol_y = vcat(temp, reverse(temp), temp[1])
polygon = SVector.(pol_x, pol_y)
# %%

df_gc = DataFrame(CSV.File("data/gc_catalog.csv", delim=" ", ignorerepeated=true))
rename!(df_gc,[:RA, :DEC] .=> [:ra, :dec])
mask_gc!(df_stream, df_gc)
df_stream.color = df_stream.bp - df_stream.rp
df_stream.parallax_rel_error = df_stream.parallax_error./df_stream.parallax
@subset!(df_stream, (:parallax_rel_error .< 0.3))
df_stream.pmra_error_rel = df_stream.pmra_error./df_stream.pmra
@subset!(df_stream, (:pmra_error_rel .< 0.3))
df_stream.pmdec_error_rel = df_stream.pmdec_error./df_stream.pmdec
@subset!(df_stream, (:pmdec_error_rel .< 0.3))
compute_in_selfCoords!(df_stream, frame)
@subset!(df_stream, 0 .< :ϕ₁ .< findmax(ϕ₁)[1] )
# @subset!(df_stream, -1 .< :ϕ₂ .< 1 )
df_stream.D = D_interp(df_stream.ϕ₁)
distmod = pyconvert(Vector{Float64},coord.Distance(Py(df_stream.D)*u.kpc).distmod.value)
df_stream.g_abs = df_stream.g - distmod
points = [[df_stream.color[i], df_stream.g_abs[i]] for i in 1:nrow(df_stream) ]
inside = [inpolygon(p, polygon; in=true, on=false, out=false) for p in points]
df_stream = df_stream[inside,:]
GC.gc()
# %%

plot_isochrone_data(df_nmask, df_stream, "CMD_mask.png")
plot_sky_histo(df_stream, "sky_$(name_s)_mf.png")
plot_sky_histo_selfFrame(df_stream, "sky_frame_$(name_s)_mf.png", df_track)
plot_sky_scatter_selfFrame(df_stream, "sky_scatter_frame_$(name_s)_mf.png", df_track)
plot_sky_histo_gc(df_stream, "sky_frame_$(name_s)_mf.png", df_gc)
plot_cmd_histo(df_stream, "CMD.png")
# %%

"""μ Hacks."""
window=[[-5.,5.],[-5.,5.]]
plot_μ_window(df_stream, "μ_$(name_s).png", window)
plot_μ(df_stream,"μ_$(name_s).png")
plot_μ_selfFrame_window(df_stream, df_track, "μ_$(name_s).png", window)
plot_μ_selfFrame_window(df_stream, "μ_$(name_s).png", window)
# @subset!(df_stream, :μ₁cosϕ₂ .> 0.0)
plot_μ_selfFrame_window(df_stream, "μ_$(name_s).png", window)
# %%

compare_tracks("Fjorm-I21", "M68-P19")

# %%

"""Filter with stream track."""
width = 1.
window=[[-7.,7.],[-7.,7.]]
Δμ = 1.
df_filt = filter_stream_on_sky(df_stream, df_track, width)
plot_sky_scatter_selfFrame(df_filt, "sky_scatter_frame_$(name_s)_filt.png", df_track)
plot_μ(df_filt,"μ_$(name_s).png")
plot_μ_scatter_selfFrame_window(df_stream, df_track, "μ_$(name_s).png", window)
df_filt = filter_stream_μ_space(df_stream, df_track, Δμ )
plot_μ(df_filt,"μ_$(name_s).png")
plot_μ_scatter_selfFrame_window(df_filt, df_track, "μ_$(name_s).png", window)
plot_sky_scatter_selfFrame(df_filt, "sky_scatter_frame_$(name_s)_filt.png", df_track)

filter_with_track(df_stream, :μ₁, 1.0)
