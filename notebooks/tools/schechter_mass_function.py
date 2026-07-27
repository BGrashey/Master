import numpy as np
import emcee
import corner


def schechter_log(L, log_phistar, log_Lstar, alpha):
    Lstar = 10**log_Lstar
    phistar = 10**log_phistar
    x = L / Lstar
    return np.log(10) * phistar * x**(alpha + 1) * np.exp(-x)


def fit_schechter(bins, phi, phi_err,
                  p0=None,
                  bounds=((-6, 0), (41, 45), (-3, 0)),
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
    truths=[results["log_phistar"][0], results["log_Lstar"][0], results["alpha"][0]]
    )
    return fig





# mock luminosity function

def make_mock_schechter(log_phistar_true, log_Lstar_true, alpha_true,
                         L_min=7e41, L_max=2e43, n_bins=8,
                         rel_noise=0.05, seed=100):
    
    rng = np.random.default_rng(seed)

    bins = np.logspace(np.log10(L_min), np.log10(L_max), n_bins)

    phi_true = schechter_log(bins, log_phistar_true, log_Lstar_true, alpha_true)

    phi_err = rel_noise * phi_true * (1 + 3 * (bins / L_max))  # Fehler wächst zum Bright End

    phi_noisy = phi_true + rng.normal(0, phi_err)

    phi_noisy = np.clip(phi_noisy, 1e-30, None)

    return bins, phi_noisy, phi_err