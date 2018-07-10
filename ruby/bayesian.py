#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 18:51:20 2018

@author: cham
"""

import emcee
# %pylab qt5
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewtest, skew, anderson

"""
teff, logg, feh, mags(W1)
varpi, sigma_varpi
"""


def parallax_to_distance(varpi):
    """

    :param varpi:
    :return:
    """
    pass


def dmod_to_parallax(m, M=0.):
    """convert distance modulus to parallax

    :param m:
        apparent mag
    :param M:
        absolute mag

    :return varpi:
        parallax [mas]

    """
    return 10. ** (2 - 0.2 * (m - M))


# %%

"""
In this section, I do some tests on normality of varpi(mag).
"""


def lnprob_gauss(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)


def ezchain_gauss(ivar, n_dim=2, n_walkers=100, n_burnin=100, n_chain=1000):
    """ get a mcmc-based gaussian sample
    
    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> c = ezchain_gauss(1, n_dim=1, n_walkers=100, n_chain=1000)
    >>> plt.figure()
    >>> plt.hist(c.flatchain)
    """
    assert n_walkers >= 2 * n_dim

    # run MCMC
    p0 = [np.random.rand(n_dim) for i in range(n_walkers)]
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob_gauss,
                                    args=[ivar])

    pos, prob, state = sampler.run_mcmc(p0, n_burnin)
    sampler.reset()

    sampler.run_mcmc(pos, n_chain)

    return sampler


def test_skewness(dmod=8., test_sigma_mag=np.logspace(-3, 0, 50)):
    # sample a gauss
    # sampler = ezchain_gauss(1, n_dim=1, n_walkers=100, n_chain=1000)
    # flatchain = sampler.flatchain

    flatchain = np.random.randn(100000)

    # test skewness
    skewness_dmod = np.zeros_like(test_sigma_mag)
    skewness_varpi = np.zeros_like(test_sigma_mag)

    # pvalue
    pvalue_dmod = np.zeros_like(test_sigma_mag)
    pvalue_varpi = np.zeros_like(test_sigma_mag)

    # statistic
    statistic_dmod = np.zeros_like(test_sigma_mag)
    statistic_varpi = np.zeros_like(test_sigma_mag)

    for i, sigma_mag in enumerate(test_sigma_mag):
        flatchain_dmod = dmod + flatchain * sigma_mag
        flatchain_varpi = dmod_to_parallax(flatchain_dmod)

        skewness_dmod[i] = skew(flatchain_dmod)
        skewness_varpi[i] = skew(flatchain_varpi)

        skewtest_result_dmod = skewtest(flatchain_dmod)
        pvalue_dmod[i] = skewtest_result_dmod.pvalue
        statistic_dmod[i] = skewtest_result_dmod.statistic

        skewtest_result_varpi = skewtest(flatchain_varpi)
        pvalue_varpi[i] = skewtest_result_varpi.pvalue
        statistic_varpi[i] = skewtest_result_varpi.statistic

    return skewness_dmod, pvalue_dmod, skewness_varpi, pvalue_varpi


def test_ad(dmod=8., test_sigma_mag=np.logspace(-3, 0, 50)):
    flatchain = np.random.randn(100000)

    # test Anderson-Darlin
    ad_dmod = []
    ad_varpi = []

    for i, sigma_mag in enumerate(test_sigma_mag):
        flatchain_dmod = dmod + flatchain * sigma_mag
        flatchain_varpi = dmod_to_parallax(flatchain_dmod)

        ad_dmod.append(anderson(flatchain_dmod.flatten()))
        ad_varpi.append(anderson(flatchain_varpi.flatten()))

    return ad_dmod, ad_varpi


def test1():
    """ test the skewness of dmod and varpi """
    skewness_dmod, pvalue_dmod, skewness_varpi, pvalue_varpi = test_skewness(
        dmod=8., test_sigma_mag=np.logspace(-3, 0, 50))


def test2():
    test_sigma_mag = np.logspace(-3, 0, 50)

    fig, axs = plt.subplots(2, 1, sharex=True)

    for i in range(20):
        print(".", end="")
        skewness_varpi, pvalue_varpi = test_skewness(
            test_sigma_mag=test_sigma_mag)
        axs[0].semilogx(test_sigma_mag, skewness_varpi)
        axs[1].semilogx(test_sigma_mag, pvalue_varpi)

    # plt.loglog(test_sigma_mag, np.abs(skewness_dmod), 'r--', alpha=.4)
    # plt.loglog(test_sigma_mag, pvalue_dmod, 'k--', alpha=.4)
    # plt.loglog(test_sigma_mag, statistic_dmod, 'm--', alpha=.4)

    # plt.loglog(test_sigma_mag, np.abs(skewness_varpi), 'r-', alpha=.4)
    # plt.loglog(test_sigma_mag, pvalue_varpi, 'k-', alpha=.4)
    # plt.loglog(test_sigma_mag, statistic_varpi, 'm-', alpha=.4)


def test_ad_normality():
    test_sigma_mag = test_sigma_mag = np.logspace(-3, 0, 50)
    ad_dmod, ad_varpi = test_ad(dmod=8., test_sigma_mag=np.logspace(-3, 0, 50))

    """ anderson-darlin result """
    plt.rcParams.update({"font.size": 15})
    fig = plt.figure()
    ar_statistic = [_.statistic for _ in ad_varpi]
    ar_cv15 = [_.critical_values[0] for _ in ad_varpi]
    ar_cv10 = [_.critical_values[1] for _ in ad_varpi]
    ar_cv5 = [_.critical_values[2] for _ in ad_varpi]
    ar_cv2p5 = [_.critical_values[3] for _ in ad_varpi]
    ar_cv1 = [_.critical_values[4] for _ in ad_varpi]
    plt.loglog(test_sigma_mag, ar_statistic, 'k-', label="A-D statistic")
    plt.loglog(test_sigma_mag, ar_cv15, label="15% critical value")
    plt.loglog(test_sigma_mag, ar_cv10, label="10% critical value")
    plt.loglog(test_sigma_mag, ar_cv5, label="5% critical value")
    plt.loglog(test_sigma_mag, ar_cv2p5, label="2.5% critical value")
    plt.loglog(test_sigma_mag, ar_cv1, label="1% critical value")
    plt.legend()
    plt.xlabel("magnitude error [mag]")
    plt.ylabel("Anderson-Darlin normality statistic")
    plt.xlim(test_sigma_mag[[0, -1]])
    fig.tight_layout()
    # fig.savefig("/home/cham/projects/gaia/figs/normality/magerr_ad.pdf")
    # fig.savefig("/home/cham/projects/gaia/figs/normality/magerr_ad.svg")
    return fig
