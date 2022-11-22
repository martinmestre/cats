"""Rediscover a stellar stream from Gaia data."""

using PythonCall
using DataFrames, DataFramesMeta
using CairoMakie, AlgebraOfGraphics
using FITSIO
using ElectronDisplay
using CSV
using PolygonOps
using StaticArrays
using Interpolations
using Revise

includet("data_methods.jl")
includet("plot_methods.jl")
dm = DataMethods
pm = PlotMethods
# %%

"""Opening the file with extinction corrected magnitudes."""
name_s = "Fjorm"
input_file = "../data/corrected/$(name_s)/corr_GaiaDR3-Fjorm-M68-all.fits"
f = FITS(input_file)
df_stream = DataFrame(f[2])
println("Fields: ", names(df_stream))
# %%

"""Remove known globular clusters."""
df_gc = DataFrame(CSV.File("../data/gc_catalog/Baumgardt/orbits_table.txt", delim=" ", ignorerepeated=true))
dm.rename!(df_gc,[:RA, :DEC] .=> [:ra, :dec])
dm.mask_gc!(df_stream, df_gc)
# %%

"""Curation."""

dm.curation!(df_stream)
# %%

"""CMD filtering."""
iso_file = "data_products/iso_stream_$(name_s).csv"
filters = "UBVRIplus"
df_iso = dm.get_isochrone(10.0e9, -1.5, filters, "linear")
CSV.write(iso_file, df_iso)
df_iso = DataFrame(CSV.File(iso_file))
dm.filter_cmd!(df_stream, df_iso)

# %%

"""Load galstreams data."""
name_t = "M68-P19"
df_track, self_frame = dm.load_stream_track(name_t)
D_interp = linear_interpolation(df_track.ϕ₁, df_track.D)  # only activate if needed.
# %%

"""Compute the stream self-coordinates and reflex correction for both data and track."""

dm.compute_in_selfCoords!(df_stream, self_frame)
dm.reflex_correct!(df_stream, self_frame)
dm.reflex_correct!(df_track, self_frame)
# @subset!(df_stream, -20. .< :ϕ₁ .< findmax(ϕ₁)[1] )
# @subset!(df_stream, -1 .< :ϕ₂ .< 1 )
# %%



"""Apply different filters to the stream."""
σ = 1.0

S = :μ₁cosϕ₂
df_filt = dm.filter_with_track(df_stream, df_track, S, σ)

S₂ = :μ₂
dm.filter_with_track!(df_filt, df_track, S₂, σ)

S₃ = :ϕ₂
df_filt_2 = dm.filter_with_track(df_stream, df_track, S₃, σ)
# %%

"""Do some plots."""

pm.plot_sky_scatter_μ_arrows_selfFrame(df_filt_2[begin:1000:end,:], "sky_scatter_frame_μ_$(name_s)_filt.png", df_track)
pm.plot_sky_scatter_μ_arrows_corr_selfFrame(df_filt_2[begin:100:end,:], "sky_scatter_frame_μ_coor_$(name_s)_filt.png", df_track)
