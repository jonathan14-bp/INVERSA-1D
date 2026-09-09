import numpy as np


# =============================================================================
# BAGIAN 1 : FORWARD MODELING DAN JACOBIAN
# =============================================================================

def VES1DFWD(r, t, s):
    """Forward modeling 1D VES Schlumberger - Filter Ghosh 11-point"""
    # Konstanta filter Ghosh
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

    # Rekursi transformasi resistivitas dari bedrock ke permukaan
    for i in range(li):
        w = l - 1
        T = r[-1]
        while w > 0:
            w = w - 1
            aa = np.tanh(t[w] / u)
            T = (T + r[w] * aa) / (1 + T * aa / r[w])
        a[i] = T
        u = u * e

    # Konvolusi filter digital 11-point
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
    """Jacobian finite-difference perturbasi 0,1%"""
    par = 0.001
    # Sensitivitas terhadap parameter resistivitas
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

    # Sensitivitas terhadap parameter ketebalan
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


# =============================================================================
# BAGIAN 2 : UTILITAS DATA DAN METRIK ERROR
# =============================================================================

def add_noise(rho_clean, noise_percent, seed=42):
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
    # RMS Error : (ln ρ_obs − ln ρ_calc) = log dari (ρ_obs − ρ_calc)/ρ_obs
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


# =============================================================================
# MODEL REFERENSI - 3 MODEL HOMOGEN
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


# =============================================================================
# BAGIAN 3 : INTI INVERSI SVD
# =============================================================================

def run_svd_inversion(
    AB2,
    Rho_app,
    n_layers,
    max_iter,
    reference_rho=None,
    progress_callback=None,
    target_rms=None, 
):
    # Initial Model Awal
    r, t = auto_initial_model(
        AB2,
        Rho_app,
        n_layers,
        reference_rho=reference_rho
    )
    m = np.concatenate([r, t])
    lr = len(r)
    lt = len(t)

    # Parameter loop inversi
    iteration = 0
    maxiteration = max_iter
    b = []

    # Pelacakan model terbaik sepanjang iterasi
    best_overall_m = None
    best_overall_rho = None
    best_overall_rms = np.inf
    best_overall_iter = 0

    worsen_count = 0
    max_worsen_iter = 5

    # Parameter backtracking line search Armijo
    armijo_c = 1e-4
    armijo_rho = 0.5
    max_ls = 10

    # Deteksi stagnasi iterasi
    stagnant_count = 0
    max_stagnant_iter = 5

    # Toleransi kriteria konvergensi
    misfit_reduction_tol = 1e-4
    model_update_tol = 1e-3

    log_messages = []

    # Respons forward awal
    Rho_app_init = np.array([VES1DFWD(r, t, AB2[i]) for i in range(len(AB2))])

    # =========================================================================
    # LOOP ITERASI INVERSI SVD
    # =========================================================================
    while iteration < maxiteration:
        r = m[:lr]
        t = m[lr:lr+lt]

        # Hitung respons dan metrik error saat ini
        Rho_app1 = np.array([VES1DFWD(r, t, AB2[i]) for i in range(len(AB2))])
        er1, misfit1, rms_percent1 = compute_log_metrics(Rho_app, Rho_app1)

        # Update best overall jika ditemukan RMS lebih kecil
        if rms_percent1 < best_overall_rms:
            best_overall_rms = rms_percent1
            best_overall_m = m.copy()
            best_overall_rho = Rho_app1.copy()
            best_overall_iter = iteration

        dd = er1.copy()

        # Jacobian dan scaling multiplicative
        A = jacobian(AB2, r, t, Rho_app1, lr, lt)
        scale = np.concatenate([r, t])
        A_scaled = A * scale

        # =====================================================================
        # REGULARISASI Tikhonov
        # =====================================================================
        Dr = np.eye(lr, k=1) - np.eye(lr, k=0)
        Dr = Dr[:lr - 1, :]
        Lt = (np.eye(lt, k=-1) - 2.0 * np.eye(lt, k=0) + np.eye(lt, k=1))

        L = np.zeros((Dr.shape[0] + lt, lr + lt))
        L[:Dr.shape[0], :lr] = Dr
        L[Dr.shape[0]:, lr:] = Lt

        # Pembobotan berbasis resolusi
        sens = np.linalg.norm(A_scaled, axis=0)
        sens = sens / np.maximum(np.max(sens), 1e-12)
        wp = 1.0 / (sens + 0.1)
        wp = wp / np.max(wp)
        row_w_r = np.array([max(wp[k], wp[k + 1]) for k in range(lr - 1)])
        row_w_t = np.array([wp[lr + k] for k in range(lt)])
        row_w = np.concatenate([row_w_r, row_w_t])
        L = row_w[:, None] * L

        # RELAKSASI ADAPTIF terhadap target fit
        tgt = float(target_rms) if (target_rms is not None and target_rms > 0) else 0.0
        relax_scale = 2.0
        excess = max(rms_percent1 - tgt, 0.0)
        reg_factor = max(excess / (excess + relax_scale), 0.05)   # lantai kecil utk kestabilan

        reg_strength = 0.03 * reg_factor
        reg_scale = reg_strength * np.linalg.norm(A_scaled, ord='fro') / np.maximum(
            np.linalg.norm(L, ord='fro'), 1e-12)
        reg_scale = np.clip(reg_scale, 1e-6, 1e2)

        log_m = np.log(np.maximum(m, 1e-9))
        A_full = np.vstack([A_scaled, reg_scale * L])
        dd_full = np.concatenate([dd, -reg_scale * (L @ log_m)])

        # Dekomposisi SVD pada sistem augmented
        U, S, Vt = np.linalg.svd(A_full, full_matrices=False)
        V = Vt.T
        ss = len(S)

        # Inisialisasi damping berbasis condition number
        cond_ratio = S[0] / np.maximum(S[-1], 1e-12)
        lambda_scale = np.clip(cond_ratio / 1000.0, 0.001, 10.0)
        beta_init = lambda_scale * S[0]

        say = 1
        accepted = False

        # Inisialisasi best candidate default
        best_m = m.copy()
        best_rho = Rho_app1.copy()
        best_misfit = misfit1
        best_rms_percent = rms_percent1
        best_dmg = np.zeros_like(m)

        # =====================================================================
        # MULTI-BETA PROGRESSIVE SEARCH
        # =====================================================================
        while say < ss:
            beta = beta_init * (1.5 ** (say - 1))
            beta = max(beta, 1e-5)

            # Damped pseudo-inverse di ruang singular
            SS = np.zeros((ss, ss))
            for i4 in range(ss):
                SS[i4, i4] = S[i4] / (S[i4] ** 2 + beta ** 2)

            dmg = V @ SS @ U.T @ dd_full
            dmg = np.clip(dmg, -2, 2)

            # Backtracking line search Armijo
            alpha_max = min(0.3, 0.1 + 0.02 * iteration)
            alpha = alpha_max
            candidate_found = False

            for ls_step in range(max_ls):
                mg_try = np.exp(np.log(np.maximum(m, 1e-9)) + alpha * dmg)
                mg_try = np.maximum(mg_try, 1e-6)

                r_try = mg_try[:lr]
                t_try = mg_try[lr:lr+lt]

                Rho_try = np.array([
                    VES1DFWD(r_try, t_try, AB2[i5])
                    for i5 in range(len(AB2))
                ])

                er_try, misfit_try, rms_percent_try = compute_log_metrics(Rho_app, Rho_try)

                # Kriteria sufficient decrease
                if misfit_try < misfit1 * (1.0 - armijo_c * alpha):
                    best_m = mg_try
                    best_rho = Rho_try
                    best_misfit = misfit_try
                    best_rms_percent = rms_percent_try
                    best_dmg = alpha * dmg
                    candidate_found = True
                    break

                alpha *= armijo_rho

            if candidate_found:
                accepted = True
                break
            else:
                say += 1

        # =====================================================================
        # PEMROSESAN HASIL ITERASI
        # =====================================================================
        if not accepted:
            # Stagnasi: tidak ada update yang memenuhi kriteria
            stagnant_count += 1
            log_messages.append(
                f"Iterasi {iteration}: tidak ada update diterima "
                f"(stagnasi {stagnant_count}/{max_stagnant_iter})"
            )
            b.append([rms_percent1, iteration])

            if stagnant_count >= max_stagnant_iter:
                log_messages.append("Proses dihentikan karena stagnasi berulang.")
                break

            iteration += 1
            if progress_callback:
                progress_callback(iteration, maxiteration, rms_percent1)
            continue

        # Update model diterima
        m_old = m.copy()
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

        # Metrik evaluasi konvergensi
        relative_misfit_reduction = (misfit1 - best_misfit) / max(misfit1, 1e-12)
        relative_model_update = np.linalg.norm(
            np.log(np.maximum(m, 1e-9)) - np.log(np.maximum(m_old, 1e-9))
        ) / np.sqrt(len(m))

        b.append([best_rms_percent, iteration])

        if progress_callback:
            progress_callback(iteration, maxiteration, best_rms_percent)

        # =====================================================================
        # KRITERIA PENGHENTIAN
        # =====================================================================
        if (relative_misfit_reduction < misfit_reduction_tol and
            relative_model_update < model_update_tol):
            log_messages.append(
                f"Konvergen pada iterasi {iteration}: "
                f"misfit_red={relative_misfit_reduction:.2e}, "
                f"model_upd={relative_model_update:.2e}"
            )
            break

        if worsen_count >= max_worsen_iter:
            log_messages.append(
                f"Dihentikan: RMS tidak membaik {max_worsen_iter} iterasi berturut-turut. "
                f"Model dikembalikan ke iterasi {best_overall_iter}."
            )
            break

    # =========================================================================
    # FINALISASI HASIL
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
        't_final': m_final[lr:lr+lt],
        'lr': lr,
        'lt': lt,
        'n_layers': n_layers,
        'best_rms': best_overall_rms,
        'best_iter': best_overall_iter,
        'iteration_done': iteration,
        'max_iter': maxiteration,
        'rms_history': b_array,
        'rho_initial': Rho_app_init,
        'log_messages': log_messages
    }

    return result    