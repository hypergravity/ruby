# -*- coding: utf-8 -*-
"""

Author
------
Bo Zhang

Email
-----
bozhang@nao.cas.cn

Created on
----------
- Fri Mar  25 17:57:00 2016     get a set of PADOVA isochrones grid

Modifications
-------------
- Fri Mar  25 17:57:00 2016     get a set of PADOVA isochrones grid

Aims
----
- get a set of PADOVA isochrones grid
- output the combined isochrone table

"""

import copy
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack, Column
from ezpadova import parsec
from joblib import Parallel, delayed, dump, load
from scipy.interpolate import PchipInterpolator, interp1d

from .imf import salpeter

Zsun = 0.0152  # this value from CMD website
Zmin = 0.0001
Zmax = 0.07
logtmax = 10.13
logtmin = 1.


def _get_one_isochrone(*args, silent=True, **kwargs):
    """ to suppress print when silent=True """
    if not silent:
        return parsec.get_one_isochrone(*args, **kwargs)
    else:
        sys.stdout = open(os.devnull, 'w')
        isoc = parsec.get_one_isochrone(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return isoc


def _find_valid_grid(grid_feh, grid_logt, Zsun=0.0152):
    """ return a valid grid of [feh, logt]

    Parameters
    ----------
    grid_feh: array
        a list [Fe/H] values for isochrone grid
    grid_logt: array
        a list logt values for isochrone grid
    Zsun: float
        the solar metallicity

    """

    grid_feh = np.array(grid_feh).flatten()

    # grid
    grid_logt = np.array(grid_logt).flatten()
    grid_Z = 10. ** grid_feh * Zsun

    ind_valid_Z = np.logical_and(grid_Z >= Zmin, grid_Z <= Zmax)
    ind_valid_logt = np.logical_and(grid_logt >= logtmin, grid_logt <= logtmax)

    # valid grid
    vgrid_feh = grid_feh[ind_valid_Z]
    vgrid_logt = grid_logt[ind_valid_logt]

    # verbose
    print("@Cham: -------------------------------------------------------")
    print("@Cham: the valid range for Z & logt are (%s, %s) & (%s, %s)."
          % (Zmin, Zmax, logtmin, logtmax))
    print("@Cham: -------------------------------------------------------")
    print("@Cham: valid input feh are:  %s" % vgrid_feh)
    print("@Cham: valid input logt are: %s" % vgrid_logt)
    print("@Cham: INvalid input feh are: %s" % grid_feh[~ind_valid_Z])
    print("@Cham: INvalid input logt are: %s" % grid_logt[~ind_valid_logt])
    print("@Cham: -------------------------------------------------------")

    return vgrid_feh, vgrid_logt


def get_isochrone_grid(grid_feh, grid_logt, model="parsec12s", phot="sloan",
                       Zsun=0.0152, n_jobs=8, verbose=10, silent=True,
                       **kwargs):
    """ get a list of isochrones using EZPADOVA

    Parameters
    ----------
    grid_feh: array
        [Fe/H] grid
    grid_logt: array
        logt grid
    model: string
        default is "parsec12s"
    phot: string
        default is "sloan"
    Zsun: float
        default is 0.0152
    n_jobs: int
        if parflat is True, specify number of jobs in JOBLIB
    verbose: int/bool
        verbose level
    **kwargs:
        other keyword arguments for parsec.get_one_isochrone()

    Returns
    -------
    vgrid_feh, vgrid_logt, grid_list, isoc_list

    """
    # validate grid
    vgrid_feh, vgrid_logt = _find_valid_grid(grid_feh, grid_logt, Zsun=Zsun)

    # construct list
    grid_list = []
    feh = []
    logt = []
    for grid_feh_ in vgrid_feh:
        for grid_logt_ in vgrid_logt:
            grid_list.append((10. ** grid_logt_, 10. ** grid_feh_ * Zsun))
            feh.append(grid_feh_)
            logt.append(grid_logt_)

    print("@Cham: you have requested for %s isochrones!" % len(grid_list))
    print("@Cham: -------------------------------------------------------")

    # get isochrones
    if n_jobs > 1:
        # get isochrones in parallel
        isoc_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_get_one_isochrone)(
                grid_list_[0], grid_list_[1], model=model, phot=phot,
                silent=silent, **kwargs)
            for grid_list_ in grid_list)
    else:
        # get isochrones sequentially
        isoc_list = []
        for i, grid_list_ in enumerate(grid_list):
            print(("@Cham: sending request for isochrone "
                   "| (logt=%s, [Fe/H]=%s) | (t=%s, Z=%s) | [%s/%s]...")
                  % (np.log10(grid_list_[0]), np.log10(grid_list_[1] / Zsun),
                     grid_list_[0], grid_list_[1],
                     i + 1, len(grid_list)))
            isoc_list.append(Table(_get_one_isochrone(
                grid_list_[0], grid_list_[1], model=model, phot=phot,
                silent=silent, **kwargs).data))

    # verbose
    print("@Cham: got all requested isochrones!")
    print("@Cham: -------------------------------------------------------")
    print("@Cham: colnames are:")
    print(isoc_list[0].colnames)
    print("@Cham: -------------------------------------------------------")

    # return vgrid_feh, vgrid_logt, grid_list, isoc_list
    return IsoGrid(isoc_list, feh, logt, Zsun)


def dump_ig(ig, fp):
    dump((ig.data, ig.feh, ig.logt, ig.Zsun), fp)
    return


def load_ig(fp):
    return IsoGrid(*load(fp))


class IsoGrid:
    """ a set of isochrones """

    data = np.array([])
    feh = np.array([])
    logt = np.array([])
    Z = np.array([])
    t = np.array([])
    Zsun = 0.0152

    u_feh = np.array([])
    u_logt = np.array([])

    default_coord = ["_lgmass", "_feh", "_lgage", "_eep"]

    @property
    def niso(self):
        return len(self.data)

    @property
    def colnames(self):
        return self.data[0].colnames

    def set_dt(self, dlogt=0.2):
        mapdict = dict()
        hdlogt = dlogt / 2.
        for logt in self.logt:
            mapdict[logt] = 10. ** (logt + hdlogt) - 10. ** (logt - hdlogt)
        for i in range(self.niso):
            dt = np.ones_like(self.data[i]["Mini"]) * mapdict[self.logt[i]]

            if "dt" in self.data[i].colnames:
                self.data[i]["dt"] = dt
            else:
                self.data[i].add_column(Column(dt, "dt"))
        return

    def set_dfeh(self, dfeh=0.2):
        for i in range(self.niso):
            _dfeh = np.ones_like(self.data[i]["Mini"]) * dfeh

            if "dfeh" in self.data[i].colnames:
                self.data[i]["dfeh"] = _dfeh
            else:
                self.data[i].add_column(Column(_dfeh, "dfeh"))
        return

    def set_dm(self):
        for i in range(self.niso):
            _dm = np.ones_like(self.data[i]["Mini"])
            _dm[1:-1] = (self.data[i]["Mini"][2:] - self.data[i]["Mini"][
                                                    :-2]) / 2.
            _dm[0] = 0.5 * (self.data[i]["Mini"][1] - self.data[i]["Mini"][0])
            _dm[-1] = 0.5 * (
                self.data[i]["Mini"][-1] - self.data[i]["Mini"][-2])

            if "dm" in self.data[i].colnames:
                self.data[i]["dm"] = _dm
            else:
                self.data[i].add_column(Column(_dm, "dm"))
        return

    def set_imf(self, func_imf=salpeter):
        for i in range(self.niso):

            imf = func_imf(self.data[i]["Mini"])
            if "imf" in self.data[i].colnames:
                self.data[i]["imf"] = imf
            else:
                self.data[i].add_column(Column(imf, "imf"))
        return

    def sub_cols(self, colnames=["Mini", "logTe"]):
        for i in range(self.niso):
            self.data[i] = self.data[i][colnames]
        return

    def sub_rows(self, cond=(("label", (0, 9)), ("Mini", (0., 12.)))):
        for i in range(self.niso):
            ind = np.ones_like(self.data[i]["Mini"], dtype=bool)
            for colname, (lb, ub) in cond:
                ind &= (self.data[i][colname] >= lb)
                ind &= (self.data[i][colname] <= ub)
            self.data[i] = self.data[i][ind]
        return

    def __getitem__(self, item):
        return self.data[item]

    def __init__(self, isoc_list, isoc_feh, isoc_logt, Zsun=0.0152):
        try:
            self.data = np.array([Table(_.data) for _ in isoc_list])
        except AttributeError:
            self.data = np.array([Table(_.columns) for _ in isoc_list])
        self.feh = np.array(isoc_feh)
        self.logt = np.array(isoc_logt)
        self.Zsun = Zsun
        self.Z = 10. ** self.feh * self.Zsun
        self.t = 10. ** self.logt

        self.grid_feh = np.unique(self.feh)
        self.grid_logt = np.unique(self.logt)

    def __repr__(self):
        s0 = "<IsoSet {} [Fe/H] x {} logt>\n".format(
            len(self.grid_feh), len(self.grid_logt))
        s_feh = "[Fe/H] grid: {}\n".format(self.grid_feh)
        s_logt = "  logt grid: {}\n".format(self.grid_logt)
        s_colnames = "Colnames: {}\n\n".format(self.colnames)
        return s0 + s_feh + s_logt + s_colnames

    def get_iso(self, logt=9.0, feh=0.0):
        d = (self.logt - logt) ** 2. + (self.feh - feh) ** 2.
        ind = np.argmin(d)
        return self.data[ind]

    def unify(self, model="parsec12s_r14"):
        """ make column names consitent for different model """
        if model == "parsec12s_r14":
            for i in range(len(self.data)):
                self.data[i] = self.unify14(self.data[i], Zsun=self.Zsun)
        elif model == "parsec12s":
            for i in range(len(self.data)):
                self.data[i] = self.unify12(self.data[i], Zsun=self.Zsun)
        else:
            raise ValueError("model not valid!")

    @staticmethod
    def unify14(isoc, Zsun=0.0152):
        isoc_ = copy.copy(isoc)

        # index
        isoc_.remove_column("index")

        # Zini & Z
        isoc_.rename_column("Zini", "feh_ini")
        isoc_["feh_ini"] = np.log10(isoc_["feh_ini"] / Zsun)
        isoc_.rename_column("Z", "feh")
        isoc_["feh"] = np.log10(isoc_["feh"] / Zsun)
        # Age --> logt
        isoc_.rename_column("Age", "logt")
        isoc_["logt"] = np.log10(isoc_["logt"])

        # rename magnitudes
        for _colname in isoc_.colnames:
            if _colname[-3:] == "mag":
                isoc_.rename_column(_colname, _colname[:-3])

        return isoc_

    @staticmethod
    def unify12(isoc, Zsun=0.0152):
        isoc_ = copy.copy(isoc)

        # index
        isoc_.remove_column("index")

        # Zini & Z
        isoc_.rename_column("Z", "feh_ini")
        isoc_["feh_ini"] = np.log10(isoc_["feh_ini"] / Zsun)
        # Age --> logt
        isoc_.rename_column("log(age/yr)", "logt")
        # logL/Lo --> logL
        isoc_.rename_column("logL/Lo", "logL")
        # M_ini --> Mini
        isoc_.rename_column("M_ini", "Mini")
        # logG --> logg
        isoc_.rename_column("logG", "logg")
        # M_act --> Mass
        isoc_.rename_column("M_act", "Mass")
        # stage --> label
        isoc_.rename_column("stage", "label")

        return isoc_

    @staticmethod
    def predict_from_chi2(combined_iso,
                          var_colnames=["teff", "logg", "feh_ini"],
                          tlf=np.array([5500, 2.5, 0.0]),
                          tlf_err=np.array([100., 0.1, 0.1]),
                          return_colnames=("Mini", "logt", "feh_ini"),
                          q=(0.16, 0.50, 0.84)):
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

        # 4. PDF --> CDF (for each return colname)
        result = np.zeros((len(q), len(return_colnames)), float)
        for i, colname in enumerate(return_colnames):
            # calculate unique values
            u_y, inv_ind = np.unique(
                combined_iso[colname], return_inverse=True)
            # calclulate CDF
            u_y = np.append(u_y, u_y[-1] * 2 - u_y[-2])
            u_p_post = np.zeros_like(u_y, float)
            # u_p_post[inv_ind] = u_p_post[inv_ind] + p_post
            for j, _ in enumerate(inv_ind):
                u_p_post[_:_ + 2] += 0.5 * p_post[j]

            result[:, i] = interp1d(
                np.cumsum(u_p_post) / np.sum(u_p_post), u_y)(q)

        return result

    def interp_mini(self, logt=9.0, feh=0.0, mini=1.0,
                    return_colnames=("Mini", "logt", "logTe", "logg")):
        """ to return interpolated values for a given isochrone(logt, feh)

        :param logt:
            log age
        :param feh:
            [Fe/H]
        :param mini:
            initial mass
        :param return_colnames:
            colnames of the returned values
        :return:
        """
        iso = self.get_iso(logt=logt, feh=feh)

        result = np.array([])
        for colname in return_colnames:
            I = interp1d(iso["Mini"], iso[colname], bounds_error=False,
                         fill_value=np.nan)
            result = np.append(result, np.float(I(mini)))

        return result

    def absorb(self, ig, colnames=("u", "g")):
        for i in range(self.niso):
            for colname in colnames:
                self.data[i].add_column(ig.data[i][colname])
        return


def chi2(x, x0, err):
    return -0.5 * ((x - x0) / err) ** 2.


# ######################
# DEPRECATED
# ######################

# TODO: implement progress bar
def interpolate_to_cube(grid_feh, grid_logt, grid_mini, isoc_list,
                        cube_quantities=[]):
    """ interpolate a slit of isochrones into data cubes
    grid_feh: array
        [Fe/H] grid
    grid_logt: array
        logt grid
    grid_mini: array
        the M_ini array to which interpolate into
    isoc_list: list
        a list of isochrones (in astropy.table.Table form)
    grid_list: list
        a list of (logt, Z) tuples corresponding to isoc_list
    cube_quantities: list
        a list of names of the quantities to be interpolated

    Returns
    -------
    cube_data_list, cube_name_list

    """
    # flatten grid
    grid_feh = np.array(grid_feh).flatten()
    grid_logt = np.array(grid_logt).flatten()
    grid_mini = np.array(grid_mini).flatten()

    # mesh cube
    cube_logt, cube_feh, cube_mini, = np.meshgrid(grid_logt, grid_feh,
                                                  grid_mini)
    cube_size = cube_feh.shape
    print("@Cham: cube shape: ", cube_size)
    print("@Cham: -------------------------------------------------------")

    # determine cube-quantities
    if len(cube_quantities) == 0:
        # all the quantities besides [feh, logt, Mini]
        colnames = list(isoc_list[0].colnames)
        assert colnames[0] == "Z"
        assert colnames[1] == "logageyr"
        assert colnames[2] == "M_ini"
        cube_quantities = colnames[3:]
    print("@Cham: Interpolating these quantities into cubes ...")
    print("%s" % cube_quantities)
    print("@Cham: -------------------------------------------------------")

    # smoothing along M_ini
    for i in range(len(isoc_list)):
        # Tablize
        if not isinstance(isoc_list[i], Table):
            isoc_list[i] = Table(isoc_list[i].data)
        # smoothing M_ini
        ind_same_mini = np.hstack(
            (False, np.diff(isoc_list[i]["M_ini"].data) == 0))
        sub_same_mini = np.arange(len(isoc_list[i]))[ind_same_mini]
        isoc_list[i].remove_rows(sub_same_mini)

        print("@Cham: smoothing isochrones [%s/%s] | %s rows removed ..."
              % (i + 1, len(isoc_list), len(sub_same_mini)))
    print("@Cham: -------------------------------------------------------")

    # interpolation
    cube_data_list = [cube_feh, cube_logt, cube_mini]
    cube_name_list = ["feh", "logt", "M_ini"]
    for k in range(len(cube_quantities)):
        cube_name = cube_quantities[k]
        c = 0
        cube_data = np.ones(cube_size) * np.nan
        for i in range(len(grid_feh)):
            for j in range(len(grid_logt)):
                this_isoc = isoc_list[c]
                P = PchipInterpolator(this_isoc["M_ini"].data,
                                      this_isoc[cube_name].data,
                                      extrapolate=False)
                # return NaNs when extrapolate
                cube_data[i, j, :] = P(grid_mini)
                print(
                    "@Cham: Interpolating [%s] | {quantity: %s/%s} (%s/%s) ..."
                    % (cube_name, k + 1, len(cube_quantities), c + 1,
                       len(grid_feh) * len(grid_logt)))
                c += 1
        cube_data_list.append(cube_data)
        cube_name_list.append(cube_name)
    print("@Cham: -------------------------------------------------------")

    return cube_data_list, cube_name_list


def cubelist_to_hdulist(cube_data_list, cube_name_list):
    """ transform data cubes into fits HDU list

    Parameters
    ----------
    cube_data_list: list
        a list of cube data
    cube_name_list: list
        a list of quantity names for cube data

    """
    print("@Cham: transforming data cubes into HDU list ...")

    # construct Primary header
    header = fits.Header()
    header["author"] = "Bo Zhang (@NAOC)"
    header["data"] = "isochrone cube"
    header["software"] = "cube constructed using BOPY.HELPER.EZPADOVA"

    # initialize HDU list
    hl = [fits.hdu.PrimaryHDU(header=header)]

    # construct HDU list
    for i in range(len(cube_data_list)):
        hl.append(
            fits.hdu.ImageHDU(data=cube_data_list[i], name=cube_name_list[i]))

    print("@Cham: -------------------------------------------------------")
    return fits.HDUList(hl)


def combine_isochrones(isoc_list):
    """ combine a list of isochrone Tables into 1 Table

    Parameters
    ----------
    isoc_list: list
        a list of isochrones (astropy.table.Table format)

    """

    if isinstance(isoc_list[0], Table):
        # assume that these data are all Table
        comb_isoc = vstack(isoc_list)
    else:
        # else convert to Table
        for i in range(isoc_list):
            isoc_list[i] = Table(isoc_list[i])
        comb_isoc = vstack(isoc_list)

    return comb_isoc


def write_isoc_list(isoc_list, grid_list,
                    dirpath="comb_isoc_parsec12s_sloan", extname=".fits",
                    Zsun=0.0152):
    """ write isochrone list into separate tables

    Parameters
    ----------
    isoc_list: list
        a list of isochrones (in astropy.table.Table format)
    grid_list: list
        (10.**grid_logt_, 10.**grid_feh_*Zsun) pairs
    dirpath: string
        the directory path
    extname: string
        the ext name
    Zsun: float
        the solar metallicity

    """

    assert len(isoc_list) == len(grid_list)
    for i in range(len(isoc_list)):
        fp = dirpath + \
             "_ZSUN" + ("%.5f" % Zsun).zfill(7) + \
             "_LOGT" + ("%.3f" % np.log10(grid_list[i][0])).zfill(6) + \
             "_FEH" + ("%.3f" % np.log10(grid_list[i][1] / Zsun)).zfill(6) + \
             extname
        print("@Cham: writing table [%s] [%s/%s]..." % (
            fp, i + 1, len(isoc_list)))
        isoc_list[i].write(fp, overwrite=True)
    return


def _test():
    """ download a random set of isochrones (sloan)
    Examples
    --------
    >>> from bopy.helpers.ezpadova.isochrone_grid import \
    >>>     (get_isochrone_grid, interpolate_to_cube, cubelist_to_hdulist,
    >>>      combine_isochrones, write_isoc_list)
    """
    # set grid
    grid_logt = [6, 7., 9]
    grid_feh = [-2.2, -1., 0, 1., 10]
    grid_mini = np.arange(0.01, 12, 0.01)

    # get isochrones
    vgrid_feh, vgrid_logt, grid_list, isoc_list = get_isochrone_grid(
        grid_feh, grid_logt, model="parsec12s", phot="sloan", n_jobs=1)

    # transform into cube data
    cube_data_list, cube_name_list = interpolate_to_cube(
        vgrid_feh, vgrid_logt, grid_mini, isoc_list,
        cube_quantities=["M_act", "g", "r"])

    # cube HDUs
    hl = cubelist_to_hdulist(cube_data_list, cube_name_list)
    hl.info()
    # hl.writeto()

    # combine isochrone tables
    comb_isoc = combine_isochrones(isoc_list)
    # comb_isoc.write()

    # write isochrone list into separate files
    # write_isoc_list(isoc_list, grid_list, "/pool/comb_isoc")
    return hl


def _test2():
    """ download full set of isochrones (2MASS)
    Examples
    --------
    >>> from bopy.helpers.ezpadova.isochrone_grid import \
    >>>     (get_isochrone_grid, interpolate_to_cube, cubelist_to_hdulist,
    >>>      combine_isochrones, write_isoc_list)
    """

    # set grid
    grid_logt = np.arange(6.0, 10.5, 0.01)
    grid_feh = np.arange(-4., +1., 0.05)
    grid_mini = np.arange(0.01, 12, 0.01)

    # get isochrones
    vgrid_feh, vgrid_logt, grid_list, isoc_list = get_isochrone_grid(
        grid_feh, grid_logt, model="parsec12s", phot="2mass", n_jobs=100)

    # transform into cube data
    cube_data_list, cube_name_list = interpolate_to_cube(
        vgrid_feh, vgrid_logt, grid_mini, isoc_list,
        cube_quantities=["Z", "logageyr", "M_act", "logLLo", "logTe",
                         "logG", "mbol", "J", "H", "Ks", "int_IMF", "stage"])

    # cube HDUs
    hl = cubelist_to_hdulist(cube_data_list, cube_name_list)
    hl.info()
    hl.writeto("/pool/model/padova/isocgrid/cube_isoc_2mass_full.fits",
               clobber=True)

    # combine isochrone tables
    comb_isoc = combine_isochrones(isoc_list)
    comb_isoc.write("/pool/model/padova/isocgrid/comb_isoc_2mass_full.fits")

    # write isochrone list into separate files
    write_isoc_list(isoc_list, grid_list,
                    "/pool/model/padova/isocgrid/2mass/2mass")
    return hl


if __name__ == "__main__":
    _test()
