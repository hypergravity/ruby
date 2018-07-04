import glob

import numpy as np
from joblib import load


class SpecLib:
    model_spec = np.array([[]])
    model_error = np.array([])

    n_spec = 0
    label = []

    def __init__(self, spec_paths="./*.dump", error_path=""):
        # load file list
        if type(spec_paths) is str:
            # format string
            data_paths = glob.glob(spec_paths)
            if len(data_paths) == 0:
                raise ValueError("No such file! [{}]".format(spec_paths))

        elif type(spec_paths) is list:
            # a list of files
            data_paths = spec_paths

        # load data files
        self.model_spec = np.vstack(
            load(data_path) for data_path in data_paths)
        self.model_error = load(error_path)
        self.n_spec = len(data_paths)

    @staticmethod
    def distribute_ipc(dv, spec_paths, error_path, speclib_name="_speclib_"):
        n_spec = len(spec_paths)
        n_engines = len(dv.get("1"))

        try:
            assert n_spec == n_engines
        except AssertionError as ae:
            raise AssertionError("n_spec_paths[{}] not equal to n_engines[{}]!"
                                 "".format(n_spec, n_engines))

        dv.scatter("_spec_paths_", spec_paths).get()
        dv.push({"_error_path_": error_path}).get()
        dv.execute("from ruby.speclib import SpecLib, chi2").get()
        dv.execute("{}=SpecLib(_spec_paths_, _error_path_)".format(
            speclib_name)).get()
        dv.wait()

        print("SpecLib instance [{}] available now!".format(speclib_name))
        return

    @staticmethod
    def chi2_ipc(dv, spec, spec_error, speclib_name="_speclib_"):
        # push observed data to engines
        dv.push({"_obs_": spec, "_obs_error_": spec_error}).get()
        # calculate chi2 in each engine
        dv.execute(
            "_chi2_ = {}.chi2(_obs_, _obs_error_)".format(speclib_name)).get()
        # gather chi2 values and return
        return np.hstack(dv.get("_chi2_"))

    def chi2(self, spec, spec_error):
        """"""
        return chi2(self.model_spec, self.model_error, spec, spec_error)

    def __repr__(self):
        return "< SpecLib instance {} x {} >\n".format(*self.model_spec.shape)


def chi2(model, model_error, obs, obs_error):
    return np.sum((model - obs) ** 2. / (model_error ** 2. + obs_error ** 2.),
                  axis=1)


def test():
    import glob
    from ipyparallel import Client
    # from ruby.speclib import SpecLib

    # remote cluster
    rc = Client(profile="default")
    print(rc.ids)
    dv = rc.direct_view()

    model_paths_fmt = "/projects/gaia/data/lamost_speclib_dr5_v1_test/spec*.dump"
    model_paths = glob.glob(model_paths_fmt)



    # error = np.random.rand(1900)*0.1
    # from joblib import dump, load
    error_path = "/projects/gaia/data/lamost_speclib_dr5_v1_test/error.dump"
    # dump(error, error_path)

    # TEST: local
    speclib = SpecLib(
        "/projects/gaia/data/lamost_speclib_dr5_v1_test/speclib_000.dump",
        "/projects/gaia/data/lamost_speclib_dr5_v1_test/error.dump")

    # TEST: ipc
    spec_paths = model_paths
    SpecLib.distribute_ipc(dv, model_paths, error_path)
    dv["_speclib_.model_spec[0]"]

    test_spec = dv["_speclib_"][0].model_spec[0]
    test_error = dv["_speclib_"][0].model_error

    chi2_value = SpecLib.chi2_ipc(dv, test_spec, test_error)

    # %%
    from astropy.table import Table
    import matplotlib.pyplot as plt

    comb_iso = Table.read(
        "/projects/gaia/data/parsec/grid/COMBINED_ISOCHRONE_"
        "gaiaDR2_2mass_spitzer_wise_panstarrs1.fits")
    plt.figure()
    plt.scatter(comb_iso["teff"][:10000], comb_iso["logg"][:10000], s=10.,
                c=chi2_value, alpha=.3)
    plt.xlim(9000, 3000)
    plt.ylim(6, 0)
    print("ok")
