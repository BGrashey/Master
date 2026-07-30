import numpy as np
import emcee
import corner

from scipy.special import gamma, gammaincc


def schechter_log(L, log_phistar, log_Lstar, alpha):
    Lstar = 10**log_Lstar
    phistar = 10**log_phistar
    x = L / Lstar
    return np.log(10) * phistar * x**(alpha + 1) * np.exp(-x)


def fit_schechter(bins, phi, phi_err,
                  p0=(-3.1, 42.8, -1.5),
                  bounds=((-6, 0), (41, 45), (-2, -1)),
                  nwalkers=32,
                  nsteps=5000,
                  nburn=1000,
                  progress=True,
                  seed=42):

    """
    Fittet eine Schechter-Leuchtkraftfunktion via MCMC).

    Parameters
    ----------
    bins, phi, phi_err : array-like
        Bin-Zentren, Phi-Werte und deren Fehler. NaNs werden automatisch
        rausgefiltert.
    p0 : tuple or None
        Startwerte (log_phistar, log_Lstar, alpha). Wenn None, werden
        sinnvolle Defaults geschätzt.
    bounds : tuple of tuples
        Flache Priors (min, max) für (log_phistar, log_Lstar, alpha).
    nwalkers, nsteps, nburn : int
        MCMC-Setup: Anzahl Walker, Gesamtschritte, Burn-in-Schritte.
    progress : bool
        Fortschrittsbalken anzeigen (braucht tqdm).
    seed : int
        Für reproduzierbare Walker-Startpositionen.

    Returns
    -------
    result : dict
        {
          'log_phistar': (median, err_lo, err_hi),
          'log_Lstar':   (median, err_lo, err_hi),
          'alpha':       (median, err_lo, err_hi),
          'phistar':     (median, err_lo, err_hi),   # linear statt log
          'Lstar':       (median, err_lo, err_hi),   # linear statt log
          'sampler':     emcee.EnsembleSampler,       # für Diagnostik/Corner-Plot
          'flat_samples': ndarray (N, 3),              # post-burn-in, geflattet
        }
    """

    bins = np.asarray(bins, dtype=float)
    phi = np.asarray(phi, dtype=float)
    phi_err = np.asarray(phi_err, dtype=float)

    # nur gültige, positive Werte verwenden
    mask = np.isfinite(phi) & np.isfinite(phi_err) & (phi > 0) & (phi_err > 0)
    if mask.sum() < 4:
        raise ValueError(
            f"Zu wenige gültige Bins für einen 3-Parameter-Fit ({mask.sum()} gefunden)."
        )
    L_fit = bins[mask]
    phi_fit = phi[mask]
    err_fit = phi_err[mask]

    (b_phistar, b_Lstar, b_alpha) = bounds

    def log_prior(theta):
        log_phistar, log_Lstar, alpha = theta
        if (b_phistar[0] < log_phistar < b_phistar[1] and
                b_Lstar[0] < log_Lstar < b_Lstar[1] and
                b_alpha[0] < alpha < b_alpha[1]):
            return 0.0
        return -np.inf

    def log_likelihood(theta, L, phi_obs, phi_err_obs):
        model = schechter_log(L, *theta)
        if np.any(~np.isfinite(model)) or np.any(model <= 0):
            return -np.inf
        return -0.5 * np.sum(((phi_obs - model) / phi_err_obs) ** 2)

    def log_probability(theta, L, phi_obs, phi_err_obs):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, L, phi_obs, phi_err_obs)

    # Startwerte schätzen, falls nicht gegeben
    if p0 is None:
        p0 = (
            np.log10(np.median(phi_fit)),
            np.log10(np.median(L_fit)),
            -1.4,
        )

    ndim = 3
    rng = np.random.default_rng(seed)
    # kleine Streuung um p0 für die Walker-Startpositionen
    pos = np.array(p0) + 1e-2 * rng.normal(size=(nwalkers, ndim))

    # sicherstellen, dass Startpositionen innerhalb der Priors liegen
    for i, (lo, hi) in enumerate(bounds):
        pos[:, i] = np.clip(pos[:, i], lo + 1e-3, hi - 1e-3)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(L_fit, phi_fit, err_fit)
    )
    sampler.run_mcmc(pos, nsteps, progress=progress)

    # Burn-in verwerfen, Chains flatten
    flat_samples = sampler.get_chain(discard=nburn, thin=15, flat=True)

    def summarize(samples):
        lo, med, hi = np.percentile(samples, [16, 50, 84])
        return (med, med - lo, hi - med)

    log_phistar_res = summarize(flat_samples[:, 0])
    log_Lstar_res = summarize(flat_samples[:, 1])
    alpha_res = summarize(flat_samples[:, 2])

    # zusätzlich in linearer Form (nützlich zum Reporten)
    phistar_samples = 10**flat_samples[:, 0]
    Lstar_samples = 10**flat_samples[:, 1]
    phistar_res = summarize(phistar_samples)
    Lstar_res = summarize(Lstar_samples)

    return {
        "log_phistar": log_phistar_res,
        "log_Lstar": log_Lstar_res,
        "alpha": alpha_res,
        "phistar": phistar_res,
        "Lstar": Lstar_res,
        "sampler": sampler,
        "flat_samples": flat_samples,
    }

def corner_plot(results):
    fig = corner.corner(
    results["flat_samples"],
    labels=[r"$\log \Phi^*$", r"$\log L^*$", r"$\alpha$"],
    truths=[results["log_phistar"][0], results["log_Lstar"][0], results["alpha"][0]],
    bins=20
    )
    return fig


def cummulative_schechter(L, log_phi, log_L, alpha):
    L_star = 10 ** log_L
    Phi_star = 10**log_phi
    x = L / L_star
    return Phi_star * gamma(alpha + 1) * gammaincc(alpha + 1, x)


import mpmath

def upper_incomplete_gamma(a, x):
    """Robuste unvollständige Gammafunktion, funktioniert auch für a < 0."""
    if np.isscalar(x):
        return float(mpmath.gammainc(a, x))
    return np.array([float(mpmath.gammainc(a, xi)) for xi in np.atleast_1d(x)])


def cum_test(L, log_phistar, log_Lstar, alpha):
    Lstar = 10**log_Lstar
    phistar = 10**log_phistar
    x = np.atleast_1d(L) / Lstar
    Gamma_vals = upper_incomplete_gamma(alpha + 1, x)
    return phistar * Gamma_vals