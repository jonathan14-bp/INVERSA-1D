import numpy as np

DISCREPANCY_FACTOR = 0.0

REG_STRENGTH_BASE = 0.03
REG_FACTOR_FLOOR_EXACT = 0.05
REG_FACTOR_FLOOR_NOISY = 0.05


# ============================================================================
# SECTION 1: FORWARD MODELING & JACOBIAN
# ============================================================================
def VES1DFWD(r, t, s):
    """1D Schlumberger VES forward modeling using the 11-point Ghosh filter."""
    q = 13
    f = 10
    m = 4.438
    x = 0
    e = np.exp((0.5 * np.log(10)) / m)
    h = 2 * q - 2
    u = s * np.exp(-f * np.log(10) / m - x)

    l = len(r)
    n = 1
    li = n + h
    a = np.zeros(li)

    for i in range(li):
        w = l - 1
        T = r[-1]
        while w > 0:
            w = w - 1
            aa = np.tanh(t[w] / u)
            T = (T + r[w] * aa) / (1 + T * aa / r[w])
        a[i] = T
        u = u * e

    i = 0
    rho_a = (
        105 * a[i]
        - 262 * a[i + 2]
        + 416 * a[i + 4]
        - 746 * a[i + 6]
        + 1605 * a[i + 8]
        - 4390 * a[i + 10]
        + 13396 * a[i + 12]
        - 27841 * a[i + 14]
        + 16448 * a[i + 16]
        + 8183 * a[i + 18]
        + 2525 * a[i + 20]
    )
    rho_a = (rho_a + 336 * a[i + 22] + 225 * a[i + 24]) / 10000
    return rho_a


def jacobian(data, r, t, Rho_app1, lr, lt):
    """Finite-difference Jacobian with a 0.1% parameter perturbation."""
    par = 0.001
    # Sensitivity with respect to resistivity parameters
    r2 = r.copy()
    J1 = np.zeros((len(data), lr))

    for i2 in range(lr):
        r2[i2] = (r[i2] * par) + r[i2]
        Rho_app2 = []
        for ii in range(len(data)):
            s = data[ii]
            g = VES1DFWD(r2, t, s)
            Rho_app2.append(g)
        Rho_app2 = np.array(Rho_app2)
        J1[:, i2] = (
            np.log(np.maximum(Rho_app2, 1e-9))
            - np.log(np.maximum(Rho_app1, 1e-9))
        ) / (r[i2] * par)
        r2 = r.copy()

    # Sensitivity with respect to thickness parameters
    t2 = t.copy()
    J2 = np.zeros((len(data), lt))
    for i3 in range(lt):
        t2[i3] = (t[i3] * par) + t[i3]
        Rho_app3 = []
        for ii in range(len(data)):
            s = data[ii]
            g = VES1DFWD(r, t2, s)
            Rho_app3.append(g)
        Rho_app3 = np.array(Rho_app3)
        J2[:, i3] = (
            np.log(np.maximum(Rho_app3, 1e-9))
            - np.log(np.maximum(Rho_app1, 1e-9))
        ) / (t[i3] * par)
        t2 = t.copy()

    J = np.hstack((J1, J2))
    return J


# ============================================================================
# SECTION 2: DATA UTILITIES
# ============================================================================
def add_noise(rho_clean, noise_percent, seed=42):
    """Multiplicative Gaussian noise."""
    if noise_percent <= 0:
        return rho_clean.copy()
    rng = np.random.default_rng(seed)
    sigma = noise_percent / 100.0
    epsilon = rng.normal(loc=0.0, scale=sigma, size=len(rho_clean))
    rho_noisy = rho_clean * (1.0 + epsilon)
    rho_noisy = np.maximum(rho_noisy, 1e-6)
    return rho_noisy

def compute_log_metrics(rho_obs, rho_cal):
    log_rho_obs = np.log(np.maximum(rho_obs, 1e-9))
    log_rho_cal = np.log(np.maximum(rho_cal, 1e-9))
    log_residual = log_rho_obs - log_rho_cal
    misfit = np.dot(log_residual, log_residual)
    # RMS error is based on (ln rho_obs - ln rho_calc), the log-domain
    # counterpart of the relative residual (rho_obs - rho_calc) / rho_obs
    rms_percent = np.sqrt(np.mean(log_residual ** 2)) * 100.0
    return log_residual, misfit, rms_percent

def auto_initial_model(AB2, Rho_app, n_layers, reference_rho=None):
    if reference_rho is None:
        rho_gmean = np.exp(np.mean(np.log(np.maximum(Rho_app, 1e-9))))
        log_rho = np.log10(rho_gmean)
        rho_init = 10 ** np.round(log_rho)
    else:
        rho_init = float(reference_rho)

    r_init = np.full(n_layers, rho_init, dtype=float)

    zmax = np.max(AB2) / 3.0
    n_thick = n_layers - 1
    ratios = 1.5 ** np.arange(n_thick)
    t_init = zmax * ratios / np.sum(ratios)

    return r_init, t_init


# ============================================================================
# SECTION 2b: MIXED-ORDER TIKHONOV REGULARIZATION OPERATOR
# ============================================================================
def build_roughness_operator(lr, lt, exact_roughness):
    Dr = np.eye(lr, k=1) - np.eye(lr, k=0)
    Dr = Dr[:lr - 1, :]

    if exact_roughness:
        if lt >= 3:
            F = np.eye(lt, k=-1) - 2.0 * np.eye(lt, k=0) + np.eye(lt, k=1)
            Lt = F[1:lt - 1, :]
        elif lt == 2:
            Lt = np.array([[-1.0, 1.0]])
        else:
            Lt = np.zeros((0, lt))
    else:
        Lt = np.eye(lt, k=-1) - 2.0 * np.eye(lt, k=0) + np.eye(lt, k=1)

    n_rows_r = Dr.shape[0]
    n_rows_t = Lt.shape[0]

    L = np.zeros((n_rows_r + n_rows_t, lr + lt))
    L[:n_rows_r, :lr] = Dr
    L[n_rows_r:, lr:] = Lt

    return L, n_rows_r, n_rows_t


def roughness_row_weights(wp, lr, lt, n_rows_r, n_rows_t, exact_roughness):
    row_w_r = np.array([max(wp[k], wp[k + 1]) for k in range(n_rows_r)])

    if n_rows_t == 0:
        row_w_t = np.zeros(0)
    elif not exact_roughness:
        row_w_t = np.array([wp[lr + k] for k in range(n_rows_t)])
    elif lt >= 3:
        row_w_t = np.array([wp[lr + k + 1] for k in range(n_rows_t)])
    else:
        row_w_t = np.array([max(wp[lr + k], wp[lr + k + 1])
                            for k in range(n_rows_t)])

    return np.concatenate([row_w_r, row_w_t])


# =============================================================================
# REFERENCE MODELS - 3 HOMOGENEOUS MODELS
# =============================================================================
MODEL_CONFIG = {
    "Homogeneous 10 Ohm.m": {
        "reference_rho": 10.0,
        "true_model_r": None,
        "true_model_t": None
    },
    "Homogeneous 100 Ohm.m": {
        "reference_rho": 100.0,
        "true_model_r": None,
        "true_model_t": None
    },
    "Homogeneous 1000 Ohm.m": {
        "reference_rho": 1000.0,
        "true_model_r": None,
        "true_model_t": None
    }
}


# ============================================================================
# SECTION 3: OBJECTIVE FUNCTION & GOLDEN SECTION SEARCH (LM CORE)
# ============================================================================
def funcGSS(A_full, dd_full, m, lam, current_iteration,
            AB2, Rho_app, lr, lt):
    """Objective function for the Golden Section Search (GSS)."""
    JTJ = A_full.T @ A_full
    g = A_full.T @ dd_full
    I = np.eye(JTJ.shape[0])

    try:
        dmg = np.linalg.solve(JTJ + lam * I, g)
    except np.linalg.LinAlgError:
        return np.inf

    dmg = np.clip(dmg, -2, 2)

    alpha = min(0.2, 0.1 + 0.02 * current_iteration)

    m2 = np.exp(np.log(np.maximum(m, 1e-9)) + alpha * dmg)
    m2 = np.maximum(m2, 1e-6)

    r2 = m2[:lr]
    t2 = m2[lr:lr + lt]

    Rho_app2 = np.array([
        VES1DFWD(r2, t2, AB2[i])
        for i in range(len(AB2))
    ])

    _, misfit2, _ = compute_log_metrics(Rho_app, Rho_app2)
    return misfit2


def gss_lm(A_full, dd_full, m, a_log, b_log, current_iteration,
           AB2, Rho_app, lr, lt):
    """Golden Section Search (GSS) for the optimal lambda."""
    tau = (np.sqrt(5) - 1) / 2
    eps = 1e-6

    x1 = a_log + (1 - tau) * (b_log - a_log)
    x2 = a_log + tau * (b_log - a_log)

    f1 = funcGSS(A_full, dd_full, m, 10**x1, current_iteration,
                 AB2, Rho_app, lr, lt)
    f2 = funcGSS(A_full, dd_full, m, 10**x2, current_iteration,
                 AB2, Rho_app, lr, lt)

    for _ in range(50):
        if abs(b_log - a_log) < eps:
            break

        if f1 < f2:
            b_log = x2
            x2 = x1
            f2 = f1
            x1 = a_log + (1 - tau) * (b_log - a_log)
            f1 = funcGSS(A_full, dd_full, m, 10**x1, current_iteration,
                         AB2, Rho_app, lr, lt)
        else:
            a_log = x1
            x1 = x2
            f1 = f2
            x2 = a_log + tau * (b_log - a_log)
            f2 = funcGSS(A_full, dd_full, m, 10**x2, current_iteration,
                         AB2, Rho_app, lr, lt)

    lam_opt_log = x1 if f1 < f2 else x2
    return 10 ** lam_opt_log


# ============================================================================
# SECTION 4: CORE LM INVERSION
# ============================================================================
def run_lm_inversion(
    AB2,
    Rho_app,
    n_layers,
    max_iter,
    reference_rho=None,
    progress_callback=None,
    target_rms=None, 
):
    # Initial model
    r, t = auto_initial_model(
        AB2,
        Rho_app,
        n_layers,
        reference_rho=reference_rho
    )
    m = np.concatenate([r, t])
    lr = len(r)
    lt = len(t)

    target_rms_value = float(target_rms) if target_rms is not None else 0.0
    if not np.isfinite(target_rms_value) or target_rms_value < 0.0:
        target_rms_value = 0.0
    noise_free_mode = target_rms_value <= 0.0

    # Inversion loop parameters
    iteration = 0
    maxiteration = max_iter
    b = []

    # Track the best model across iterations
    best_overall_m = None
    best_overall_rho = None
    best_overall_rms = np.inf
    best_overall_iter = 0

    worsen_count = 0
    max_worsen_iter = 5

    armijo_c = 1e-4
    armijo_rho = 0.5
    max_ls = 10

    # Iteration stagnation detection
    stagnant_count = 0
    max_stagnant_iter = 5

    # Convergence criteria tolerances
    misfit_reduction_tol = 1e-4
    model_update_tol = 1e-3

    convergence_rms_guard = max(2.0 * target_rms_value, 5.0)
    false_convergence_count = 0
    max_false_convergence = 5

    # Multi-lambda search parameters
    max_lambda_trials = 10
    lambda_growth_factor = 1.5

    lambda_floor_relaxation = lambda_growth_factor ** (-8)

    log_messages = []

    # Initial forward response
    Rho_app_init = np.array([VES1DFWD(r, t, AB2[i]) for i in range(len(AB2))])

    # ========================================================================
    # LEVENBERG-MARQUARDT INVERSION ITERATION LOOP
    # ========================================================================
    while iteration < maxiteration:
        r = m[:lr]
        t = m[lr:lr + lt]

        # Compute the current response and error metrics
        Rho_app1 = np.array([VES1DFWD(r, t, AB2[i]) for i in range(len(AB2))])
        er1, misfit1, rms_percent1 = compute_log_metrics(Rho_app, Rho_app1)

        # Update the overall best model if a lower RMS is found
        if rms_percent1 < best_overall_rms:
            best_overall_rms = rms_percent1
            best_overall_m = m.copy()
            best_overall_rho = Rho_app1.copy()
            best_overall_iter = iteration

        dd = er1.copy()

        # Jacobian and multiplicative scaling
        A = jacobian(AB2, r, t, Rho_app1, lr, lt)
        scale = np.concatenate([r, t])
        A_scaled = A * scale

        # =====================================================================
        # TIKHONOV REGULARIZATION
        # =====================================================================
        L, n_rows_r, n_rows_t = build_roughness_operator(
            lr, lt, exact_roughness=noise_free_mode
        )

        # Resolution-based weighting
        sens = np.linalg.norm(A_scaled, axis=0)
        sens = sens / np.maximum(np.max(sens), 1e-12)
        wp = 1.0 / (sens + 0.1)
        wp = wp / np.max(wp)
        row_w = roughness_row_weights(
            wp, lr, lt, n_rows_r, n_rows_t, exact_roughness=noise_free_mode
        )
        if L.shape[0] > 0:
            L = row_w[:, None] * L

        # Adaptive relaxation toward the target fit
        tgt = target_rms_value
        relax_scale = 2.0
        excess = max(rms_percent1 - tgt, 0.0)
        reg_floor = (REG_FACTOR_FLOOR_EXACT if noise_free_mode
                     else REG_FACTOR_FLOOR_NOISY)
        reg_factor = max(excess / (excess + relax_scale), reg_floor)

        reg_strength = REG_STRENGTH_BASE * reg_factor
        reg_scale = reg_strength * np.linalg.norm(A_scaled, ord='fro') / np.maximum(
            np.linalg.norm(L, ord='fro'), 1e-12)
        reg_scale = np.clip(reg_scale, 1e-6, 1e2)

        log_m = np.log(np.maximum(m, 1e-9))
        A_full = np.vstack([A_scaled, reg_scale * L])
        dd_full = np.concatenate([dd, -reg_scale * (L @ log_m)])

        # DAMPING FLOOR BASED ON THE CONDITION NUMBER
        # The spectrum is derived from the eigenvalues of J^T J ONLY to set
        # the damping floor; the LM solver itself still solves
        # (J^T J + lambda I) dm = g.
        JTJ_floor = A_full.T @ A_full
        eig_full = np.linalg.eigvalsh(JTJ_floor)
        s_max = np.sqrt(np.maximum(eig_full[-1], 0.0))
        s_min = np.sqrt(np.maximum(eig_full[0], 0.0))
        cond_ratio = s_max / np.maximum(s_min, 1e-12)
        lambda_scale = np.clip(cond_ratio / 1000.0, 0.001, 10.0)
        beta_floor = lambda_scale * s_max         
        lam_floor = beta_floor ** 2             

        if noise_free_mode:
            lam_floor = lam_floor * lambda_floor_relaxation

        a_log = max(-5.0, float(np.log10(max(lam_floor, 1e-12))))
        b_log = max(5.0, a_log + 1.0)

        # [LM CORE] Find the optimal lambda via Golden Section Search
        # (lower bound constrained)
        lam_init = gss_lm(A_full, dd_full, m, a_log, b_log, iteration,
                          AB2, Rho_app, lr, lt)
        lam_init = max(lam_init, lam_floor)
        
        m_old = m.copy()    

        # Initialize the default best candidate
        best_m = m.copy()
        best_rho = Rho_app1.copy()
        best_misfit = misfit1
        best_rms_percent = rms_percent1

        say = 0
        accepted = False

        # [LM CORE] MULTI-LAMBDA PROGRESSIVE SEARCH
        while say < max_lambda_trials:
            lam = lam_init * (lambda_growth_factor ** say)
            lam = max(lam, lam_floor)

            # [LM CORE] LM SOLVER: (J^T J + lambda I) dm = J^T d
            JTJ = A_full.T @ A_full
            g = A_full.T @ dd_full
            I = np.eye(JTJ.shape[0])

            try:
                dmg = np.linalg.solve(JTJ + lam * I, g)
            except np.linalg.LinAlgError:
                say += 1
                continue

            dmg = np.clip(dmg, -2, 2)

            # [LM CORE] ARMIJO BACKTRACKING LINE SEARCH
            alpha_max = min(0.2, 0.1 + 0.02 * iteration)
            alpha_current = alpha_max
            candidate_found = False

            for ls_step in range(max_ls):
                mg_try = np.exp(np.log(np.maximum(m, 1e-9)) + alpha_current * dmg)
                mg_try = np.maximum(mg_try, 1e-6)

                r_try = mg_try[:lr]
                t_try = mg_try[lr:lr + lt]

                Rho_try = np.array([
                    VES1DFWD(r_try, t_try, AB2[i_ls])
                    for i_ls in range(len(AB2))
                ])

                er_try, misfit_try, rms_percent_try = compute_log_metrics(Rho_app, Rho_try)

                if misfit_try < misfit1 * (1.0 - armijo_c * alpha_current):
                    best_m = mg_try
                    best_rho = Rho_try
                    best_misfit = misfit_try
                    best_rms_percent = rms_percent_try
                    candidate_found = True
                    break

                alpha_current *= armijo_rho

            if candidate_found:
                accepted = True
                break
            else:
                say += 1

        # ====================================================================
        # RESULT PROCESSING
        # ====================================================================
        if not accepted:
            # Stagnation: no update satisfied the acceptance criterion
            stagnant_count += 1
            log_messages.append(
                f"Iteration {iteration}: no update accepted "
                f"(stagnation {stagnant_count}/{max_stagnant_iter})"
            )
            b.append([rms_percent1, iteration])

            if stagnant_count >= max_stagnant_iter:
                log_messages.append("Process stopped due to repeated stagnation.")
                break

            iteration += 1
            if progress_callback:
                progress_callback(iteration, maxiteration, rms_percent1)
            continue

        # Model update accepted
        m = best_m
        iteration += 1
        stagnant_count = 0

        if best_rms_percent < best_overall_rms:
            best_overall_rms = best_rms_percent
            best_overall_m = m.copy()
            best_overall_rho = best_rho.copy()
            best_overall_iter = iteration
            worsen_count = 0
        else:
            worsen_count += 1

        # Convergence evaluation metrics
        relative_misfit_reduction = (misfit1 - best_misfit) / max(misfit1, 1e-12)
        relative_model_update = np.linalg.norm(
            np.log(np.maximum(m, 1e-9)) - np.log(np.maximum(m_old, 1e-9))
        ) / np.sqrt(len(m))

        b.append([best_rms_percent, iteration])

        if progress_callback:
            progress_callback(iteration, maxiteration, best_rms_percent)

        # =====================================================================
        # STOPPING CRITERIA
        # =====================================================================
        if (not noise_free_mode) and best_rms_percent <= DISCREPANCY_FACTOR * target_rms_value:
            log_messages.append(
                f"Discrepancy criterion reached at iteration {iteration}: "
                f"RMS={best_rms_percent:.3f}% <= assumed noise level "
                f"{target_rms_value:.3f}%."
            )
            break

        if (relative_misfit_reduction < misfit_reduction_tol and
            relative_model_update < model_update_tol):
            if best_rms_percent > convergence_rms_guard:
                false_convergence_count += 1
                log_messages.append(
                    f"Iteration {iteration}: model change and misfit are "
                    f"already very small, but RMS is still {best_rms_percent:.3f}%. "
                    f"Convergence not yet declared "
                    f"({false_convergence_count}/{max_false_convergence})."
                )
                if false_convergence_count >= max_false_convergence:
                    log_messages.append(
                        "Process stopped: the model stopped improving even "
                        "though the data is not yet well fitted. Check the "
                        "number of layers or the initial reference model."
                    )
                    break
            else:
                log_messages.append(
                    f"Converged at iteration {iteration}: "
                    f"misfit_reduction={relative_misfit_reduction:.2e}, "
                    f"model_update={relative_model_update:.2e}"
                )
                break

        if worsen_count >= max_worsen_iter:
            log_messages.append(
                f"Stopped: RMS did not improve for {max_worsen_iter} consecutive "
                f"iterations. Model reverted to iteration {best_overall_iter}."
            )
            break

    # =========================================================================
    # RESULT FINALIZATION
    # =========================================================================
    if best_overall_m is not None:
        m_final = best_overall_m.copy()
        rho_final = best_overall_rho.copy()
    else:
        m_final = m.copy()
        rho_final = Rho_app_init.copy()

    b_array = np.array(b) if len(b) > 0 else np.array([[0, 0]])

    result = {
        'm_final': m_final,
        'rho_final': rho_final,
        'r_final': m_final[:lr],
        't_final': m_final[lr:lr + lt],
        'lr': lr,
        'lt': lt,
        'n_layers': n_layers,
        'best_rms': best_overall_rms,
        'best_iter': best_overall_iter,
        'iteration_done': iteration,
        'max_iter': maxiteration,
        'rms_history': b_array,
        'rho_initial': Rho_app_init,
        'target_rms': target_rms_value,
        'noise_free_mode': noise_free_mode,
        'log_messages': log_messages
    }

    return result