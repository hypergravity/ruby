#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:35:57 2018

@author: cham
"""

# %%
#%pylab qt5
import numpy as np
from ruby import get_isochrone_grid, IsoGrid
# from ruby import isoc_interp, ezinterp
# from ezpadova.parsec import get_one_isochrone, get_photometry_list
from astropy.table import Table

from ruby.isochrone_interp import ezinterp
from ruby.imf import salpeter

from ezpadova import parsec

# %%
""" 1. define grid """
grid_feh = np.arange(-4., 1.2, 0.1)  # 4
grid_logt = np.arange(6., 10.5, 0.1)  # x2

# ig12 = get_isochrone_grid(
#        grid_feh, grid_logt, model='parsec12s',
#        phot='gaiaDR2',Zsun=0.0152, n_jobs=20, verbose=10)
# ig12.unify(model='parsec12s')

""" 2. download isochrones """
ig_gaiaDR2 = get_isochrone_grid(
    grid_feh, grid_logt, model='parsec12s_r14',
    phot='gaiaDR2', Zsun=0.0152, n_jobs=100, verbose=10)
ig_gaiaDR2.Zsun

# %%
from ruby.isochrone_grid import dump_ig, load_ig

dump_ig(ig_gaiaDR2,
        "/media/cham/Seagate Expansion Drive/parsec/grid/gaiaDR2.dump")
# %%
from ruby.isochrone_grid import dump_ig, load_ig

ig_gaiaDR2 = load_ig(
    "/media/cham/Seagate Expansion Drive/parsec/grid/gaiaDR2.dump")
# %%
import copy

ig = copy.copy(ig_gaiaDR2)

""" modify columns """
from astropy.table import Column

for i in range(ig.niso):
    ig.data[i].add_column(
        Column(np.log10(ig.data[i]["Zini"] / ig.Zsun), "feh_ini"))
    # ig.data[i].remove_column("feh_ini")
    ig.data[i].add_column(Column(np.log10(ig.data[i]["Z"] / ig.Zsun), "feh"))
    # ig.data[i].remove_column("feh")
    ig.data[i].add_column(Column(np.log10(ig.data[i]["Age"]), "logt"))
    # ig.data[i].remove_column("logt")

# %%
# ig.unify(model='parsec12s_r14')

""" 3. select subset of isochrone [columns & rows]"""
# ig.sub_rows(cond=(('label', (0, 8)), ('Mini', (0.0, 8.0)), ('logTe', (3.6, 4.1)), ('logg', (2.0, 5.0))))
ig.sub_rows(cond=(('label', (0, 8)), ('Mini', (0.0, 8.0))))
colnames = ['feh_ini', 'logt', 'Mini', 'Mass', 'logL', 'logTe', 'logg',
            'label', 'Mloss', 'feh', 'mbolmag', 'Gmag', 'G_BPmag', 'G_RPmag']
ig.sub_cols(colnames=colnames)

""" 4. interpolate along Mini, logTe, logg, etc."""
for i in range(ig.niso):
    ig.data[i] = ezinterp(ig[i],
                          restrictions=(
                          ('Mini', 0.02), ('logTe', 0.02), ('logg', 0.05)),
                          mode='linear', Mini='Mini')

""" 5. define volume weight """
ig.set_dt(0.2)
ig.set_dfeh(dfeh=0.5)
ig.set_dm()
ig.set_imf(salpeter)

""" 6. calculate total weight for each point """
from astropy.table import Column

for i in range(ig.niso):
    w = ig.data[i]["dm"] * ig.data[i]["dt"] * ig.data[i]["dfeh"] * ig.data[i][
        "imf"]
    try:
        ig.data[i].add_column(Column(w, "w"))
    except ValueError:
        ig.data[i]["w"] = w

    ig.data[i].add_column(Column(10. ** ig.data[i]["logTe"], "teff"))

""" 7. combine all isochrones """
from astropy import table

combined_iso = table.vstack(list(ig.data))

# %%
# from joblib import dump, load
# dump(ig, "/media/cham/Seagate Expansion Drive/parsec/grid/gaiaDR2.dump")

# %%
""" 1. The prior from model """
rcParams.update({"font.size": 20})
H, xe, ye = np.histogram2d(
    combined_iso["logTe"], combined_iso["logg"],
    bins=(np.arange(3., 5, .005), np.arange(-1., 6., 0.05)), normed=False,
    weights=combined_iso["w"])

fig = figure(figsize=(10, 8))
imshow(np.fliplr(np.log10(H.T)), cmap=cm.jet,
       extent=(5.005, 2.995, 6.025, -1.025), aspect="auto")
colorbar()
xlim(4.5, 3.3)
ylim(5.5, -1)

title("log10(Prior)")
xlabel("$T_{\\rm eff}$ [K]")
ylabel("$\\log{g}$ [dex]")
fig.tight_layout()
fig.savefig("/home/cham/projects/gaia/figs/bayesian/log10_prior.pdf")
fig.savefig("/home/cham/projects/gaia/figs/bayesian/log10_prior.svg")

# %%
from ruby import IsoGrid

x_sun = IsoGrid.predict_from_chi2(
    combined_iso,
    var_colnames=["teff", "logg", "feh_ini"],
    tlf=np.array(np.array([5750, 4.35, 0.0])),
    tlf_err=np.array([100., 0.1, 0.1]),
    return_colnames=("Mini", "logt", "feh_ini"),
    q=(0.16, 0.50, 0.84))

tlf = np.array([5750, 4.35, 0.0])  # solar value
tlf_err = np.array([100., 0.1, 0.1])

var_colnames = ["teff", "logg", "feh_ini"]
# 1. convert isochrone(table) into array
sub_iso = np.array(combined_iso[var_colnames].to_pandas())

# 2. calculate chi2
chi2_values = 0
for i_var in range(len(var_colnames)):
    chi2_values += ((sub_iso[:, i_var] - tlf[i_var]) / tlf_err[
        i_var]) ** 2.
chi2_values *= -0.5

# 3. chi2 --> PDF
p_post = np.exp(chi2_values) * combined_iso["w"]

figure()
plt.hist(10 ** combined_iso["logTe"], weights=p_post,
         bins=np.arange(5000, 6000, 30), normed=True)

figure()
plt.hist(combined_iso["logg"], weights=p_post, bins=np.arange(4, 5, 0.02),
         normed=True)

figure()
plt.hist(combined_iso["feh_ini"], weights=p_post, bins=np.arange(-1, 1, 0.1),
         normed=True)

figure()
plt.hist2d(10 ** combined_iso["logTe"], combined_iso["logg"],
           bins=(np.arange(5200, 6250, 25), np.arange(4, 5, 0.05)),
           weights=p_post, normed=True, cmap=cm.gray_r)
colorbar()

# %%
# H, xe, ye = np.histogram2d(
#        10.**combined_iso["logTe"], combined_iso["logg"],
#        bins=(np.arange(3500., 10000, 50.), np.arange(-1., 6., 0.05)), normed=False,
#        weights=combined_iso["w"])
#
# import seaborn as sns
# Y = np.vstack((10.**combined_iso["logTe"], combined_iso["logg"])).T
# ax = sns.kdeplot(Y, shade = True, cmap = "PuBu")
#
#
# figure()
# imshow(np.fliplr(np.log10(H.T)), cmap=cm.jet, extent=(10000+25, 3500-25, 6.025, -1.025), aspect="auto")
# colorbar()

# %%
try:
    combined_iso.add_column(Column(10. ** combined_iso["logTe"], "teff"))

# %%

test_tlf = np.array([5750, 4.35, 0.0])  # solar value
err_tlf = np.array([100., 0.1, 0.1])

test_tlf = np.array([5500, 2.5, 0.0])
err_tlf = np.array([100., 0.1, 0.1])

test_tlf = np.array([10. ** 3.68, 2.44, 0.0])
err_tlf = np.array([100., 0.1, 0.1])


def chi2(x, x0, err):
    return -0.5 * ((x - x0) / err) ** 2.


# %%timeit

grid_logt = ig.grid_logt
grid_feh = ig.grid_feh
# grid_mini = np.arange(0.3, 5.1, 0.1)

grid_logt = ig.grid_logt
# grid_feh = np.arange(-2.0, 0.8, 0.8)
grid_feh = np.array([-2.0, -1.0, 0., 0.4])
grid_mini = np.logspace(-1, 1, 30)

mesh_logt, mesh_feh, mesh_mini = np.meshgrid(grid_logt, grid_feh, grid_mini)

""" ind: 1:feh, 2:logt, 3:mini """
mesh_label = np.zeros_like(mesh_logt)
# basics, mini, logt, feh
mesh_logt_est = np.zeros_like(mesh_logt)
mesh_logt_err = np.zeros_like(mesh_logt)

mesh_feh_est = np.zeros_like(mesh_logt)
mesh_feh_err = np.zeros_like(mesh_logt)

mesh_mini_est = np.zeros_like(mesh_logt)
mesh_mini_err = np.zeros_like(mesh_logt)

# spectroscopic, teff, logg
mesh_teff = np.zeros_like(mesh_logt)
mesh_teff_est = np.zeros_like(mesh_logt)
mesh_teff_err = np.zeros_like(mesh_logt)

mesh_logg = np.zeros_like(mesh_logt)
mesh_logg_est = np.zeros_like(mesh_logt)
mesh_logg_err = np.zeros_like(mesh_logt)

n_all = np.prod(mesh_logt.shape)
ijk = []
for i in range(mesh_logt.shape[0]):
    for j in range(mesh_logt.shape[1]):
        for k in range(mesh_logt.shape[2]):
            o_interp = ig.interp_mini(
                mesh_logt[i, j, k], mesh_feh[i, j, k], mesh_mini[i, j, k],
                return_colnames=(
                'Mini', 'logt', "feh_ini", 'teff', 'logg', "label"))
            mesh_teff[i, j, k] = o_interp[3]
            mesh_logg[i, j, k] = o_interp[4]
            ijk.append((i, j, k))
            mesh_label[i, j, k] = o_interp[5]
print(len(ijk))

from ipyparallel import Client

rc = Client(profile="default")
dv = rc.direct_view()
dv.push({"combined_iso": combined_iso}).get()
dv.push({"mesh_teff": mesh_teff}).get()
dv.push({"mesh_logg": mesh_logg}).get()
dv.push({"mesh_feh": mesh_feh}).get()
dv.push({"flat_mini": mesh_mini.flatten()}).get()
dv.push({"flat_logt": mesh_logt.flatten()}).get()

dv.execute("import numpy as np").get()
dv.execute("from ruby.isochrone_grid import IsoGrid").get()

cmd = """
x_all = np.zeros((len(ijk), 3, 3))
for i, _ijk in enumerate(ijk):
    _i, _j, _k = _ijk

    if mesh_teff[_i, _j, _k]>0:
        try:
            x_all[i] = IsoGrid.predict_from_chi2(
                combined_iso, 
                var_colnames=["teff", "logg", "feh_ini"],
                tlf=np.array([mesh_teff[_i, _j, _k], mesh_logg[_i, _j, _k], mesh_feh[_i, _j, _k]]), 
                tlf_err=np.array([100., 0.1, 0.1]), 
                return_colnames=("Mini", "logt", "feh_ini"),
                q=(0.16, 0.50, 0.84))
        except ValueError as ae:
            x_all[i] = np.nan
    else:
        x_all[i] = np.nan
"""

# %% test
dv.scatter("ijk", ijk[:100]).get()

dv["ijk"]
dv["len(ijk)"]
dv["x=1"]
% % time
dv.execute(cmd).get()

# %%
dv.scatter("ijk", ijk[:]).get()

dv["ijk"]
dv["len(ijk)"]
dv["x=1"]
dv.execute(cmd).get()
# %%
x_all = dv.gather("x_all").get()
# from joblib import dump
# dump(x_all, "/home/cham/projects/gaia/data/x_all.dump")
for i, (_i, _j, _k) in enumerate(ijk):
    mesh_mini_est[_i, _j, _k] = x_all[i][1, 0]
    mesh_mini_err[_i, _j, _k] = (x_all[i][2, 0] - x_all[i][0, 0]) / 2.
    mesh_logt_est[_i, _j, _k] = x_all[i][1, 1]
    mesh_logt_err[_i, _j, _k] = (x_all[i][2, 1] - x_all[i][0, 1]) / 2.

# %%
rcParams.update({"font.size": 20})
fig = figure(figsize=(10, 8));
ax = fig.add_subplot(111)
im = ax.imshow(mesh_logt_err[2], vmin=0., vmax=0.5, cmap=cm.jet,
               origin="lower",
               extent=(-1 - 1 / 30., 1 + 1 / 30., 6. - 0.05, 10.1 + 0.05),
               aspect="auto")

xticks = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.])
ax.set_xticks(np.log10(xticks))
ax.set_xticklabels(xticks)
colorbar(im)
ax.contour(np.log10(grid_mini), grid_logt, mesh_logt_err[2], [0.1, 0.2, 0.4],
           colors="k", linewidths=[1, 2, 3])
ax.set_xlabel("$M_{ini}$ [$M_\\odot$]")
ax.set_ylabel("$\\log_{10}$ (age / yr)")

ax.plot([0.4, 0.6], [10, 10], lw=1, c="k")
ax.text(.7, 10 - .06, "0.10")

ax.plot([0.4, 0.6], [9.75, 9.75], lw=2, c="k")
ax.text(.7, 9.75 - .06, "0.20")

ax.plot([0.4, 0.6], [9.5, 9.50], lw=3, c="k")
ax.text(.7, 9.5 - .06, "0.40")

ax.set_title("Error of $\\log_{10}$ (age / yr)")
fig.tight_layout()
fig.savefig("/home/cham/projects/gaia/figs/bayesian/logt_error.pdf")
fig.savefig("/home/cham/projects/gaia/figs/bayesian/logt_error.svg")

# %%
rcParams.update({"font.size": 20})
fig = figure(figsize=(10, 8));
ax = fig.add_subplot(111)
im = ax.imshow(np.abs(mesh_logt_est[2] - mesh_logt[2]), vmin=0, vmax=3,
               cmap=cm.jet, origin="lower",
               extent=(-1 - 1 / 30., 1 + 1 / 30., 6. - 0.05, 10.1 + 0.05),
               aspect="auto")

xticks = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.])
ax.set_xticks(np.log10(xticks))
ax.set_xticklabels(xticks)
colorbar(im)
ax.contour(np.log10(grid_mini), grid_logt,
           np.abs(mesh_logt_est[2] - mesh_logt[2]), [0.5, 1.0, 2.0],
           colors="w", linewidths=[1, 2, 3])
ax.set_xlabel("$M_{ini}$ [$M_\\odot$]")
ax.set_ylabel("$\\log_{10}$ (age / yr)")

ax.plot([0.4, 0.6], [10, 10], lw=1, c="k")
ax.text(.7, 10 - .06, "0.5")

ax.plot([0.4, 0.6], [9.75, 9.75], lw=2, c="k")
ax.text(.7, 9.75 - .06, "1.0")

ax.plot([0.4, 0.6], [9.5, 9.50], lw=3, c="k")
ax.text(.7, 9.5 - .06, "2.0")

ax.set_title("Bias of $\\log_{10}$ (age / yr)")
fig.tight_layout()
fig.savefig("/home/cham/projects/gaia/figs/bayesian/logt_bias.pdf")
fig.savefig("/home/cham/projects/gaia/figs/bayesian/logt_bias.svg")

# %%
rcParams.update({"font.size": 20})
fig = figure(figsize=(10, 8));
ax = fig.add_subplot(111)
im = ax.imshow(mesh_mini_err[2] / mesh_mini_est[2], vmin=0., vmax=0.3,
               cmap=cm.jet, origin="lower",
               extent=(-1 - 1 / 30., 1 + 1 / 30., 6. - 0.05, 10.1 + 0.05),
               aspect="auto")

xticks = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.])
ax.set_xticks(np.log10(xticks))
ax.set_xticklabels(xticks)
colorbar(im)
ax.contour(np.log10(grid_mini), grid_logt, mesh_mini_err[2] / mesh_mini_est[2],
           [0.05, 0.1, 0.2], colors="w", linewidths=[1, 2, 3])
ax.set_xlabel("$M_{ini}$ [$M_\\odot$]")
ax.set_ylabel("$\\log_{10}$ (age / yr)")

ax.plot([0.4, 0.6], [10, 10], lw=1, c="k")
ax.text(.7, 10 - .06, "0.05")

ax.plot([0.4, 0.6], [9.75, 9.75], lw=2, c="k")
ax.text(.7, 9.75 - .06, "0.10")

ax.plot([0.4, 0.6], [9.5, 9.50], lw=3, c="k")
ax.text(.7, 9.5 - .06, "0.20")

ax.set_title("Error of $M_{ini}$ [$M_\\odot$]")
fig.tight_layout()
fig.savefig("/home/cham/projects/gaia/figs/bayesian/mass_error.pdf")
fig.savefig("/home/cham/projects/gaia/figs/bayesian/mass_error.svg")

# %%
rcParams.update({"font.size": 20})
fig = figure(figsize=(10, 8));
ax = fig.add_subplot(111)
im = ax.imshow(np.abs(mesh_mini_est[2] - mesh_mini[2]) / mesh_mini[2], vmin=0.,
               vmax=0.1, cmap=cm.jet, origin="lower",
               extent=(-1 - 1 / 30., 1 + 1 / 30., 6. - 0.05, 10.1 + 0.05),
               aspect="auto")

xticks = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.])
ax.set_xticks(np.log10(xticks))
ax.set_xticklabels(xticks)
colorbar(im)
ax.contour(np.log10(grid_mini), grid_logt,
           np.abs(mesh_mini_est[2] - mesh_mini[2]) / mesh_mini[2],
           [0.01, 0.05, 0.08], colors="w", linewidths=[1, 2, 3])
ax.set_xlabel("$M_{ini}$ [$M_\\odot$]")
ax.set_ylabel("$\\log_{10}$ (age / yr)")

ax.plot([0.4, 0.6], [10, 10], lw=1, c="k")
ax.text(.7, 10 - .06, "0.01")

ax.plot([0.4, 0.6], [9.75, 9.75], lw=2, c="k")
ax.text(.7, 9.75 - .06, "0.05")

ax.plot([0.4, 0.6], [9.5, 9.50], lw=3, c="k")
ax.text(.7, 9.5 - .06, "0.08")

ax.set_title("Bias of $M_{ini}$ [$M_\\odot$]")
fig.tight_layout()
fig.savefig("/home/cham/projects/gaia/figs/bayesian/mass_bias.pdf")
fig.savefig("/home/cham/projects/gaia/figs/bayesian/mass_bias.svg")

# %%
rcParams.update({"font.size": 20})
fig = figure(figsize=(10, 8));
ax = fig.add_subplot(111)
im = ax.imshow(mesh_label[2], vmin=0., vmax=7, cmap=cm.jet, origin="lower",
               extent=(-1 - 1 / 30., 1 + 1 / 30., 6. - 0.05, 10.1 + 0.05),
               aspect="auto")

xticks = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.])
ax.set_xticks(np.log10(xticks))
ax.set_xticklabels(xticks)
colorbar(im)
ax.contour(np.log10(grid_mini), grid_logt, mesh_label[2], [0.5, 1.5, 2.5],
           colors="w", linewidths=[1, 2, 3])
ax.set_xlabel("$M_{ini}$ [$M_\\odot$]")
ax.set_ylabel("$\\log_{10}$ (age / yr)")

ax.plot([0.4, 0.6], [10, 10], lw=1, c="k")
ax.text(.7, 10 - .06, "0.5")

ax.plot([0.4, 0.6], [9.75, 9.75], lw=2, c="k")
ax.text(.7, 9.75 - .06, "1.5")

ax.plot([0.4, 0.6], [9.5, 9.50], lw=3, c="k")
ax.text(.7, 9.5 - .06, "2.5")

ax.set_title("Labels")
fig.tight_layout()
fig.savefig("/home/cham/projects/gaia/figs/bayesian/label.pdf")
fig.savefig("/home/cham/projects/gaia/figs/bayesian/label.svg")

# %%
iso = ig.get_iso(9.7, 0)
figure()
plot(iso["teff"], iso["logg"], 'x-')
for i in range(len(iso)):
    #    text(iso["teff"][i], iso["logg"][i], "{:1.0f}".format(iso["label"][i]))
    text(iso["teff"][i], iso["logg"][i], "{:1.3f}".format(iso["Mini"][i]))

# %%

figure()
plot(mesh_teff[2].flatten(), mesh_logg[2].flatten(), '.')

# %%
grid_teff = np.arange(8000., 4000, -100)
grid_logg = np.arange(6, -0.1, -0.2)
# grid_feh  = np.array([0.])

# mesh_teff, mesh_logg, mesh_feh = np.meshgrid(grid_teff, grid_logg, grid_feh)
mesh_teff, mesh_logg = np.meshgrid(grid_teff, grid_logg)

flat_teff = mesh_teff.flatten()
flat_logg = mesh_logg.flatten()

dv.scatter("flat_teff", flat_teff).get()
dv.scatter("flat_logg", flat_logg).get()
dv["flat_teff.shape"]

cmd = """
x_all = np.zeros((len(flat_teff), 3, 3))
for i, (_teff, _logg) in enumerate(zip(flat_teff, flat_logg)):
    try:
        x_all[i] = IsoGrid.predict_from_chi2(
            combined_iso, 
            var_colnames=["teff", "logg", "feh_ini"],
            tlf=np.array([_teff, _logg, 0.]), 
            tlf_err=np.array([100., 0.1, 0.1]), 
            return_colnames=("Mini", "logt", "feh_ini"),
            q=(0.16, 0.50, 0.84))
    except ValueError as ae:
        x_all[i] = np.nan
"""
dv.execute(cmd).get()

# %%
x2 = dv.gather("x_all").get()
mesh_mini_est = x2[:, 1, 0].reshape(*mesh_teff.shape)
mesh_mini_err = ((x2[:, 2, 0] - x2[:, 0, 0]) / 2.).reshape(*mesh_teff.shape)
mesh_logt_est = x2[:, 1, 1].reshape(*mesh_teff.shape)
mesh_logt_err = ((x2[:, 2, 1] - x2[:, 0, 1]) / 2.).reshape(*mesh_teff.shape)
# %%

figure()
imshow(mesh_mini_err)
# %%
if mesh_teff[i, j, k] > 0 and i > 0 and j > 0:
    # do a chi2 matching
    try:
        x = ig.predict_from_chi2(
            combined_iso,
            var_colnames=["teff", "logg", "feh_ini"],
            tlf=np.array(
                [mesh_teff[i, j, k], mesh_logg[i, j, k], mesh_feh[i, j, k]]),
            tlf_err=np.array([100., 0.1, 0.1]),
            return_colnames=("Mini", "logt", "feh_ini"),
            q=(0.16, 0.50, 0.84))
        mesh_mini_est[i, j, k] = x[1, 0]
        mesh_mini_err[i, j, k] = (x[2, 0] - x[0, 0]) / 2.
        mesh_logt_est[i, j, k] = x[1, 1]
        mesh_logt_err[i, j, k] = (x[2, 1] - x[0, 1]) / 2.
        print(i, j, k, n_all, "success")
    except ValueError as ae:
        mesh_mini_est[i, j, k] = np.nan
        mesh_mini_err[i, j, k] = np.nan
        mesh_logt_est[i, j, k] = np.nan
        mesh_logt_err[i, j, k] = np.nan
        print(i, j, k, n_all, "failure")
else:
    mesh_mini_est[i, j, k] = np.nan
    mesh_mini_err[i, j, k] = np.nan
    mesh_logt_est[i, j, k] = np.nan
    mesh_logt_err[i, j, k] = np.nan
    print(i, j, k, n_all, "failure")

x = ig.predict_from_chi2(
    combined_iso,
    var_colnames=["teff", "logg", "feh_ini"],
    tlf=np.array([5500, 2.5, 0.0]),
    tlf_err=np.array([100., 0.1, 0.1]),
    return_colnames=("Mini", "logt", "feh_ini", "G"),
    q=(0.16, 0.50, 0.84))

# %%


var_colnames = ["teff", "logg", "feh_ini"]
sub_iso = np.array(combined_iso[var_colnames].to_pandas())

chi2_values = 0
for ivar in range(len(var_colnames)):
    chi2_values += ((sub_iso[:, ivar] - test_tlf[ivar]) / err_tlf[ivar]) ** 2.
chi2_values *= -0.5

p_post = np.exp(chi2_values) * combined_iso["w"]

u_feh_ini, inv_ind = np.unique(combined_iso["feh_ini"], return_inverse=True)

u_p_post = np.zeros(u_feh_ini.shape)
u_p_post[inv_ind] = u_p_post[inv_ind] + p_post
for i, _ in enumerate(inv_ind):
    if _ < len(u_p_post):
        u_p_post[_] += 0.5 * p_post[i]
        u_p_post[_ + 1] += 0.5 * p_post[i]
    else:
        u_p_post[_] += p_post[i]

from scipy.interpolate import interp1d

interp1d(np.cumsum(u_p_post) / np.sum(u_p_post), u_feh_ini)((0.16, 0.50, 0.84))

figure()
plot(u_feh_ini, u_p_post)
plot(u_feh_ini, np.cumsum(u_p_post))

u_teff, inv_ind = np.unique(combined_iso["teff"], return_inverse=True)
u_p_post = np.zeros_like(u_teff)
u_p_post[inv_ind] += p_post
figure()
plot(u_teff, u_p_post)
plot(u_teff, np.cumsum(u_p_post))
# %% teff
hist, bin_edges = np.histogram(combined_iso["teff"],
                               bins=np.arange(3500., 10000, 50.), normed=True,
                               weights=p_post)

figure()
plt.step(bin_edges[:-1], hist)

# %% logg
hist, bin_edges = np.histogram(combined_iso["logg"],
                               np.arange(-1., 6., 0.1), normed=True,
                               weights=p_post)

figure()
plt.step(bin_edges[:-1], hist)

# %% Mini
hist, bin_edges = np.histogram(combined_iso["Mini"],
                               np.arange(0., 6., 0.2), normed=True,
                               weights=p_post)

figure()
plt.step(bin_edges[:-1], hist)

# %% logt
hist, bin_edges = np.histogram(combined_iso["logt"],
                               np.arange(0., 13., 0.2), normed=True,
                               weights=p_post)

figure()
plt.step(bin_edges[:-1], hist)

# %%

H, xe, ye = np.histogram2d(
    combined_iso["teff"], combined_iso["logg"],
    bins=(np.arange(3500., 10000, 100.), np.arange(-1., 6., 0.1)), normed=True,
    weights=p_post)
H = np.log10(H)
figure()
imshow(np.fliplr(H.T), cmap=cm.jet,
       extent=(10000 + 25, 3500 - 25, 6.025, -1.025),
       aspect="auto", vmin=-10, vmax=np.nanmax(H))

# %%
H, xe, ye = np.histogram2d(
    combined_iso["Mini"], combined_iso["logt"],
    bins=(np.arange(0, 5, 0.1), np.arange(0, 13., 0.2)), normed=True,
    weights=p_post)
H = np.log10(H)
# H *= H>-10
figure()
imshow(H.T, cmap=cm.gray_r,
       extent=(0 - 0.025, 5 + 0.025, 6 - 0.05, 10.0 + 0.05),
       aspect="auto", vmin=-10, vmax=np.nanmax(H))

# %%

figure()
plot(sub_iso[:, 1], chi2_values, 'o', alpha=0.5)

figure()
scatter(sub_iso[:, 0], sub_iso[:, 1], s=10, c=chi2_values, alpha=0.5, vmin=-10,
        vmax=0, cmap=cm.gray_r)
colorbar()
xlim(6500, 5000)
ylim(, 5500)


# %%
figure()
plot(combined_iso["logTe"], combined_iso["logg"], '.')

from joblib import dump

dump(ig, "/media/cham/Seagate Expansion Drive/parsec/grid/gaiaDR2.dump")


# %%
def salpeter(m, ksi0=1.0):
    return ksi0 * (m) ** -2.35


x = np.arange(0.08, 12.0, 0.01)
y = salpeter(x)
figure()
plot(np.log10(x), np.log10(y))
i

d_mini

# %%

figure()
for isoc in ig.data:
    ind = (isoc["label"] >= 1) & (isoc["label"] < 8) & (isoc["Mini"] < 12)
    ind = (isoc["label"] < 9) & (isoc["Mini"] < 12)
    plot(isoc["logTe"][ind], isoc["logg"][ind], 'kx', alpha=0.05)

# %%
figure();
x = ig.get_iso(9.0, 0)
plot(x["logTe"], x["logg"], "r-.")
for i in range(len(x)):
    text(x["logTe"][i], x["logg"][i], "{}".format(x["label"][i]))

x = ig12.get_iso(9.0, 0)
plot(x["logTe"], x["logg"], "b-.")
for i in range(len(x)):
    text(x["logTe"][i], x["logg"][i], "{}".format(x["label"][i]))

# %%
"""
Process:
    0. Download grid d_logt=0.05, d_feh_ini=0.05
    1. unify colnames
    2. subcol & subrow, Mini<12.0, 1<=label<8
    3. interpolate : 
        Mini 0.02
        logTe 0.01
        logg 0.01
    4. calculate weight
    5. add prior/weight

stage. The labels are: 
0=PMS, 
1=MS, 
2=SGB, 
3=RGB, 
(4,5,6)=different stages of CHEB, 
7=EAGB, 
8=TPAGB.
"""
# %%
print(ig)
print(ig12)

# %%
ig12.get_iso(9.0, 0.0)
ig.get_iso(9.0, 0.0)["label"]

unify12(isoc)

# %%
# Mini

# Mass --> Mact
# isoc.rename_column("Mass", "Mact")

# Mass --> Mact


# define (minit, logt, feh) column names


x12 = ig12.get_iso(7.0, 0.0)
x = ig.get_iso(7.0, 0.0)

plot(isoc_interp(x, restrictions=(("Mini", 0.05),),
                 interp_config=(("label", "linear"),), M_ini="Mini"))

figure();
plot(x["Mini"][:-1], np.diff(x["Mini"]))

# %%
from ruby import IsoSet

IsoSet(isoc_list, vgrid_feh)

# %%
# get_one_isochrone(1e9, 0.0152, model="parsec12s",phot="gaia")

grid_feh = np.arange(-4., 1.2, 0.2)
grid_logt = np.arange(6., 10.5, 0.05)

vgrid_feh, vgrid_logt, grid_list, isoc_list = get_isochrone_grid(
    grid_feh, grid_logt, model='parsec12s_r14', phot='gaiaDR2',
    Zsun=0.0152, n_jobs=20, verbose=10)

np.sum([_.data.shape[0] for _ in isoc_list])

from ezpadova import parsec

print(parsec.get_one_isochrone(1e7, 0.02, model='parsec12s',
                               phot='gaia').colnames)
print(parsec.get_one_isochrone(1e7, 0.02, model='parsec12s_r14',
                               phot='gaia').colnames)

# %%

gdr1 = parsec.get_one_isochrone(1e9, 0.0152, model='parsec12s', phot='gaia')
gdr1_r14 = parsec.get_one_isochrone(1e9, 0.0152, model='parsec12s_r14',
                                    phot='gaia')
gdr2 = parsec.get_one_isochrone(1e9, 0.0152, model='parsec12s', phot='gaiaDR2')
# %%
figure()
plot(gdr1["G"] - gdr1["G_RP"], gdr1["G"], '-')
scatter(gdr1["G"] - gdr1["G_RP"], gdr1["G"], s=10, c=gdr1["stage"],
        cmap=cm.jet)

ind = gdr1_r14["label"] < 8
plot(gdr1_r14["Gmag"][ind] - gdr1_r14["G_RPmag"][ind], gdr1_r14["Gmag"][ind],
     '-')
colorbar()
ylim(30, -30)
# %%
figure()
plot(gdr1["Gmag"] - gdr1["G_RPmag"], gdr1["Gmag"], '-')
plot(gdr2["Gmag"] - gdr2["G_RPmag"], gdr2["Gmag"], '-')
