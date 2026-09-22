"""
===============================================================================
ABOUT PAGE - INVERSA-1D
Overview of the application followed by the methodological formulation of the
1D resistivity inversion framework (damped SVD and Levenberg-Marquardt),
reproduced from the Methods chapter of the accompanying research article.
===============================================================================
"""

from html import escape
from textwrap import dedent

import streamlit as st


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="About | INVERSA-1D",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# HTML RENDER HELPER
# =============================================================================

def render_html(markup):
    """Render HTML safely in Streamlit, stripping triple-quote indentation."""
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


# =============================================================================
# APPLICATION / RESEARCH INFORMATION
# =============================================================================

APP_NAME = "INVERSA-1D"
PROGRAM = "Geophysical Engineering"
INSTITUTION = "Sumatra Institute of Technology (ITERA)"
LOCATION = "South Lampung, Indonesia"


# =============================================================================
# CUSTOM CSS  (layout only; colours follow the active Streamlit theme)
# =============================================================================

st.markdown("""
<style>
    /* ---- Layout and accents; text colours follow the active Streamlit theme ---- */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1120px;
    }

    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}
    header {visibility: visible;}

    /* ---- Page heading ---- */
    .about-title {
        color: #e74c3c;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.03em;
    }

    .about-subtitle {
        color: #64748b;
        font-size: 0.98rem;
        line-height: 1.75;
        text-align: justify;
        max-width: 900px;
        margin-bottom: 1.4rem;
    }

    .about-rule {
        width: 100%;
        height: 1px;
        background: linear-gradient(
            90deg, transparent 0%, #e2e8f0 12%, #e2e8f0 88%, transparent 100%
        );
        margin: 0.4rem 0 1.6rem 0;
    }

    /* ---- Section / subsection titles ---- */
    .methods-overview-title {
        font-size: 1.62rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.6rem 0 0.6rem 0;
        padding-left: 0.75rem;
        border-left: 5px solid #e74c3c;
    }

    .methods-section-title {
        font-size: 1.28rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 2.0rem 0 0.55rem 0;
        padding-bottom: 0.45rem;
        border-bottom: 2px solid #e74c3c;
    }

    .methods-subsection-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin: 1.5rem 0 0.4rem 0;
        padding-left: 0.6rem;
        border-left: 3px solid #94a3b8;
    }

    /* ---- Body prose ---- */
    .methods-body {
        font-size: 0.95rem;
        line-height: 1.85;
        text-align: justify;
        margin-bottom: 0.35rem;
    }

    /* ---- Equation styling (st.latex / KaTeX) ---- */
    .katex-display {
        margin: 0.9rem 0 !important;
        padding: 0.15rem 0;
        overflow-x: auto;
        overflow-y: hidden;
    }

    .katex {
        font-size: 1.06em;
    }

    code {
        background: #f1f5f9 !important;
        color: #e74c3c !important;
        border-radius: 4px;
        padding: 0.1rem 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE HEADING
# =============================================================================

render_html(f"""
<div class="about-title">About</div>
<div class="about-subtitle">
    <b>{escape(APP_NAME)}</b> is a web-based application for the one-dimensional
    inversion of Schlumberger Vertical Electrical Sounding (VES) resistivity data
    using the damped Singular Value Decomposition (SVD) and Levenberg&ndash;Marquardt
    (LM) methods. It was developed for academic purposes as part of an undergraduate
    final project (tugas akhir) in the {escape(PROGRAM)} program at
    {escape(INSTITUTION)}, {escape(LOCATION)}, and accompanies the authors' research
    on adaptive 1D resistivity inversion.
</div>
<div class="about-rule"></div>
""")


# =============================================================================
# METHODOLOGY - OVERVIEW
# =============================================================================

render_html('<div class="methods-overview-title">Methodology</div>')

render_html(f"""
<div class="methods-body">
    This section summarizes the methodological formulation implemented in
    {escape(APP_NAME)}, reproduced from the Methods section of the accompanying
    research article. Both inversion schemes share the same forward operator,
    Jacobian, and evaluation metrics, and differ only in how the model update is
    solved and how the damping factor is determined, with all computations carried
    out in the natural-logarithm domain. The formulation is presented in three
    parts, namely forward modeling, inversion, and model evaluation, and in its
    present form has been evaluated on synthetic Schlumberger sounding data.
</div>
""")


# =============================================================================
# 1. FORWARD MODELING
# =============================================================================

render_html('<div class="methods-section-title">1. Forward Modeling</div>')

render_html("""
<div class="methods-body">
    The apparent resistivity response for a horizontally layered earth model is
    calculated through one-dimensional forward modeling. The relationship between
    apparent resistivity and layer parameters is expressed by the first-order
    Hankel integral:
</div>
""")

st.latex(r"\rho_a = (S)^2 \int_0^{\infty} T_1(\lambda)\, J_1(S\lambda)\,\lambda\, d\lambda")

render_html("""
<div class="methods-body">
    where <b>S = AB/2</b> is half of the current electrode spacing, &lambda; is the
    integration variable, <b>J<sub>1</sub></b> is the first-order Bessel function,
    and <b>T<sub>1</sub>(&lambda;)</b> is the resistivity transform function. For an
    <b>n</b>-layer model, T<sub>1</sub>(&lambda;) is computed recursively from the
    bottom layer toward the surface as follows:
</div>
""")

st.latex(r"T_i(\lambda) = \frac{T_{i+1}(\lambda) + \rho_i \tanh(\lambda h_i)}{1 + T_{i+1}(\lambda)\tanh(\lambda h_i)/\rho_i}\,; \quad i = n,\, n-1,\, \ldots,\, 1 ")

render_html("""
<div class="methods-body">
    where &rho;<sub>i</sub> and h<sub>i</sub> are the resistivity and thickness of
    the <b>i</b>-th layer, respectively, <b>N</b> is the total number of layers.
    The recursion starts from the bottom layer, which is treated as a half-space,
    such that <b>T<sub>n</sub> = &rho;<sub>n</sub></b>. The integral in Equation (1)
    can be simplified using a digital linear filter as follows:
</div>
""")

st.latex(r"\rho_a(s) = \sum_{k=1}^{K} T_1(\lambda_k)\, f_k")

render_html("""
<div class="methods-body">
    where <b>f<sub>k</sub></b> is the digital filter coefficient and <b>K</b> is the
    number of filter coefficients. Based on the above equations, the relationship
    between apparent resistivity and model parameters is a nonlinear function.
</div>
""")


# =============================================================================
# 2. INVERSION
# =============================================================================

render_html('<div class="methods-section-title">2. Inversion</div>')

render_html("""
<div class="methods-body">
    In the inversion process, the objective is to determine model parameters whose
    response matches the observed data. In this study, all model parameters are
    represented in the natural-logarithm domain to ensure positivity, reduce
    parameter-scaling effects, and improve numerical stability throughout the
    iterative inversion process. This representation is particularly important
    because resistivity and thickness values commonly span several orders of
    magnitude, so working in the logarithmic domain keeps the estimated parameters
    physically meaningful while preventing parameters with larger magnitudes from
    dominating the update. On this basis, the general relationship between the model
    parameters and the data can be expressed as follows:
</div>
""")

st.latex(r"\mathbf{G}(\mathbf{m}) = \mathbf{d}")

render_html("""
<div class="methods-body">
    where <b>d</b> is the observed data vector whose <b>N</b> elements are the
    apparent resistivity values measured at the corresponding half-current electrode
    spacings (<b>AB/2</b>), <b>G</b> is the nonlinear forward operator that maps a
    given model to its predicted apparent resistivity response at those same
    spacings, and and <b>m</b> is the model parameter vector describing an <b>n</b>-layer
    earth. Each of the <b>n</b> layers has its own resistivity, whereas only the upper
    <b>n &minus; 1</b> layers have a definable thickness because the lowermost layer is
    treated as a half-space extending infinitely downward. The model vector therefore
    comprises <b>n</b> resistivities and <b>n &minus; 1</b> thicknesses, giving
    <b>2n &minus; 1</b> unknowns in total. If Equation (4) is nonlinear, a first-order
    Taylor series approximation can be applied as follows:
</div>
""")

st.latex(r"\mathbf{d} = \mathbf{G}(\mathbf{m}_0 + \Delta\mathbf{m}) \approx \mathbf{G}(\mathbf{m}_0) + \mathbf{J}\,\Delta\mathbf{m}")

render_html("""
<div class="methods-body">
    where <b>&Delta;m</b> denotes the model update vector that is added at each
    iteration, which can be expressed mathematically as follows:
</div>
""")

st.latex(r"\Delta\mathbf{m} = \mathbf{m}^{k+1} - \mathbf{m}^{k}")

render_html("""
<div class="methods-body">
    where <b>k</b> is the iteration index. Furthermore, the parameter <b>J</b> is the
    Jacobian matrix whose elements are the partial derivatives of the operator
    <b>G</b> with respect to the model parameters, and can be expressed mathematically
    as follows:
</div>
""")

st.latex(r"J_{ij} = \frac{\partial G_i(\mathbf{m})}{\partial m_j}")

render_html("""
<div class="methods-body">
    where, in this study, the Jacobian matrix <b>J</b> is calculated using the
    finite-difference method with a parameter perturbation of 0.1%.
</div>
""")


# =============================================================================
# 2.1 TIKHONOV REGULARIZATION
# =============================================================================

render_html('<div class="methods-subsection-title">2.1 Tikhonov Regularization</div>')

render_html("""
<div class="methods-body">
    In this study, mixed-order Tikhonov regularization is employed, in which the
    order of the derivative operator is chosen to match the physical behaviour
    expected of each parameter type. First-order regularization is applied to the
    resistivity parameters to preserve possible sharp contrasts between layers,
    whereas second-order regularization is applied to the thickness parameters to
    suppress oscillatory solutions and promote smooth structural variations. This
    approach is consistent with the principle of smoothness-constrained inversion,
    which states that the desired model should not only fit the observed data but
    should also be as simple or as smooth as possible to avoid over-interpretation.
    Accordingly, the regularization serves to stabilize the inversion system of
    equations, reduce sensitivity to noise, and produce solutions that are more
    robust and physically consistent with subsurface characteristics. In general,
    the regularized objective function is expressed as the sum of the data misfit
    term and the regularization term as follows:
</div>
""")

st.latex(r"\Phi(\mathbf{m}) = \left\| \mathbf{d} - \mathbf{G}(\mathbf{m}) \right\|^{2} + \gamma^{2} \left\| \mathbf{L}\mathbf{m} \right\|^{2}")

render_html("""
<div class="methods-body">
    where <b>L</b> represents the roughness operator and &gamma; represents the
    regularization weight. In this study, resistivity roughness is calculated using
    the first-order derivative:
</div>
""")

st.latex(r"\left(\mathbf{L}_{\rho}\mathbf{m}\right)_{k} = \rho_{k+1} - \rho_{k}")

render_html("""
<div class="methods-body">
    whereas thickness roughness is calculated using the second-order derivative:
</div>
""")

st.latex(r"\left(\mathbf{L}_{h}\mathbf{m}\right)_{k} = h_{k-1} - 2h_{k} + h_{k+1}")

render_html("""
<div class="methods-body">
    The regularization weight (&gamma;) is not held constant but is updated adaptively
    at each iteration according to the current level of agreement between the observed
    and modelled data. Because the non-linear problem is solved through successive
    linearization, the regularization parameter can be re-estimated at every
    linearized step instead of being fixed beforehand, following a cooling-type
    schedule in which strong regularization early in the process is progressively
    relaxed as the solution improves. Accordingly, a comparatively large weight is
    maintained while the misfit remains high, so that the added constraint stabilizes
    the early iterations and suppresses noise-driven structure; the weight is then
    gradually reduced as the misfit decreases, preventing the inversion from enforcing
    excessive data fitting beyond the information genuinely contained in the data. If
    Equation (8) is minimized, the following regularized Gauss&ndash;Newton normal
    equation is obtained:
</div>
""")

st.latex(r"\left(\mathbf{J}^{T}\mathbf{J} + \gamma^{2}\mathbf{L}^{T}\mathbf{L}\right)\Delta\mathbf{m} = \mathbf{J}^{T}\Delta\mathbf{d} - \gamma^{2}\mathbf{L}^{T}\mathbf{L}\mathbf{m}")

render_html("""
<div class="methods-body">
    where <b>&Delta;d</b> denotes the data residual vector, defined as the difference
    between the observed data and the calculated data based on the current model,
    which can be expressed mathematically as follows:
</div>
""")

st.latex(r"\Delta\mathbf{d} = \mathbf{d} - \mathbf{G}(\mathbf{m})")

render_html("""
<div class="methods-body">
    Equation (11) is formulated as an augmented least-square problem through the
    construction of an augmented Jacobian matrix, which combines the Jacobian matrix
    (<b>J</b>) and the regularization term (&gamma;<b>L</b>), together with the
    augmented residual vector (<b>d</b>). This formulation allows the influence of
    both the data and the regularization to be considered simultaneously during the
    inversion process. This augmented form subsequently serves as the basis for the
    solution procedures employed in both inversion schemes applied in this study.
</div>
""")


# =============================================================================
# 2.2 LEVENBERG-MARQUARDT
# =============================================================================

render_html('<div class="methods-subsection-title">2.2 Levenberg-Marquardt</div>')

render_html("""
<div class="methods-body">
    The Levenberg&ndash;Marquardt scheme introduces a damping factor (&lambda;) into
    the computation, such that the model update becomes:
</div>
""")

st.latex(r"\left(\mathbf{J}^{T}\mathbf{J} + \lambda\mathbf{I}\right)\Delta\mathbf{m} = \mathbf{J}^{T}\mathbf{d}")

render_html("""
<div class="methods-body">
    The damping factor (&lambda;) serves as a control parameter. When &lambda; is
    small, the method approaches the Gauss&ndash;Newton scheme with faster convergence,
    whereas a large &lambda; causes the method to approach gradient descent, which is
    more stable. The model is then updated using the following Equation:
</div>
""")

st.latex(r"\mathbf{m}^{(k+1)} = \mathbf{m}^{(k)} + \alpha\, \Delta\mathbf{m}")

render_html("""
<div class="methods-body">
    where &alpha; is the step length and <b>k</b> denotes the iteration number. This
    update is performed in the logarithmic domain and subsequently transformed back to
    the original domain using the exponential function, thereby ensuring that the
    resulting resistivity and thickness parameters remain positive.
    <br><br>
    The damping value (&lambda;) is not fixed but is instead determined optimally using
    the Golden Section Search (GSS) method so that it can adapt to the conditions of
    each iteration. Golden Section Search (GSS) minimizes the misfit as a function of
    &lambda; over the range log<sub>10</sub>(&lambda;) thereby efficiently covering
    several orders of magnitude. GSS progressively narrows the search interval using
    the golden ratio. If the search interval is defined as <b>[a, b]</b>, then the two
    interval reduction points are calculated using the golden ratio as follows:
</div>
""")

st.latex(r"x_1 = a + (1-\tau)(b-a) \quad \text{and} \quad x_2 = a + \tau(b-a)")

render_html("""
<div class="methods-body">
    where &tau; is the golden ratio with a value of <b>&tau; = 0.618</b>. During each
    search iteration, if &Phi;(x<sub>1</sub>) &lt; &Phi;(x<sub>2</sub>) the upper bound
    of the interval is updated to x<sub>2</sub>, and vice versa. This process is
    repeated continuously while reducing the search interval until its width becomes
    smaller than the specified tolerance value. The resulting optimum damping factor is
    then constrained so that it does not fall below a predefined lower threshold in
    order to maintain numerical stability throughout the inversion process. Once
    convergence is achieved, the damping factor considered optimal is assigned as the
    final result of the Golden Section Search procedure and is used in the model update
    for that inversion iteration.
</div>
""")

st.latex(r"\lambda_{optimum} = 10^{\,x_{optimum}}")


# =============================================================================
# 2.3 SINGULAR VALUE DECOMPOSITION
# =============================================================================

render_html('<div class="methods-subsection-title">2.3 Singular Value Decomposition</div>')

render_html("""
<div class="methods-body">
    The Singular Value Decomposition scheme decomposes the matrix (<b>J</b>) into
    several matrices as follows:
</div>
""")

st.latex(r"\mathbf{J} = \mathbf{U}\mathbf{S}\mathbf{V}^{T}")

render_html("""
<div class="methods-body">
    where <b>U</b> is an <b>M &times; N</b> matrix whose elements consist of left
    singular vectors, <b>V</b> is an <b>N &times; N</b> matrix whose elements consist
    of right singular vectors, and <b>S</b> is a diagonal matrix whose main diagonal
    contains the singular values. Substituting Equation (17) into the
    Gauss&ndash;Newton normal equation yields the following expression:
</div>
""")

st.latex(r"\left(\mathbf{U}\mathbf{S}^{2}\mathbf{V}^{T} + \lambda\mathbf{I}\right)\Delta\mathbf{m} = \mathbf{V}\mathbf{S}\mathbf{U}^{T}\left[\mathbf{d} - \mathbf{G}(\mathbf{m}^{k})\right]")

render_html("""
<div class="methods-body">
    where <b>S</b> is a diagonal matrix, whose main diagonal contains the singular
    values of matrix <b>J</b> (&sigma;<sub>1</sub>, &sigma;<sub>2</sub>,
    &sigma;<sub>3</sub>, ..., &sigma;<sub>n</sub>).
</div>
""")

st.latex(r"\left(\mathbf{U}\mathbf{S}^{2}\mathbf{V}^{T} + \lambda\mathbf{I}\right) = \left(\mathbf{V}\,\operatorname{diag}\!\left(\sigma_j^{2}\right)\mathbf{V}^{T} + \lambda\mathbf{I}\right); \quad j = 1, 2, \ldots, N")

render_html("""
<div class="methods-body">
    Equation (19) is solved to obtain the model update used in the inversion, which
    can be expressed as follows:
</div>
""")

st.latex(r"\mathbf{m}^{k+1} = \mathbf{m}^{k} + \mathbf{V}\,\operatorname{diag}\!\left(\frac{\sigma_i}{\sigma_i^{2} + \lambda}\right)\mathbf{U}^{T}\left[\mathbf{d} - \mathbf{G}(\mathbf{m}^{k})\right]")

render_html("""
<div class="methods-body">
    In Equation (20), the operator applied to the data residual is the damped
    (regularized) pseudo-inverse of the Jacobian rather than the Jacobian itself. It is
    obtained by replacing the na&iuml;ve inverse of each singular value
    (1/&sigma;<sub>i</sub>), with the damped factor
    (&sigma;<sub>i</sub>/(&sigma;<sub>i</sub><sup>2</sup>+&lambda;)), so that the
    contribution of small singular value is suppressed rather than amplified. This
    damped pseudo-inverse, denoted <b>J<sup>#</sup></b>, can be written explicitly as:
</div>
""")

st.latex(r"\mathbf{J}^{\#} = \mathbf{V}\,\operatorname{diag}\!\left(\frac{\sigma_i}{\sigma_i^{2} + \beta^{2}}\right)\mathbf{U}^{T}, \quad \beta^{2} = \lambda,")

render_html("""
<div class="methods-body">
    where <b>J<sup>#</sup></b> is the damped pseudo-inverse of <b>J</b> and
    &beta;<sup>2</sup> is the damping parameter. Because the factor
    (&sigma;<sub>i</sub>/(&sigma;<sub>i</sub><sup>2</sup>+&beta;<sup>2</sup>)) remains
    bounded even when &sigma;<sub>i</sub> is very small, the damping prevents poorly
    resolved model components from being governed by near-zero singular values, thereby
    stabilizing the inversion.
</div>
""")

st.latex(r"\kappa = \frac{\sigma_{\max}}{\sigma_{\min}}, \quad \beta_0^{2} = \eta\,\sigma_{\max}^{2}, \quad \eta_{\min} \le \eta \le \eta_{\max}")

render_html("""
<div class="methods-body">
    where &kappa; is the condition number and &eta; is a bounded scaling factor. Rather
    than being fixed in advance, &eta; is selected adaptively from the condition number
    itself. The ratio &beta;<sub>0</sub>/&sigma;<sub>max</sub> is set proportional to
    &kappa; relative to a fixed reference level, so that &eta; grows as the system
    becomes more ill-conditioned and the initial damping &beta;<sub>0</sub><sup>2</sup>
    is automatically strengthened to improve stability. To keep this adaptation within a
    reasonable range &eta; is constrained to a bounded interval
    (&eta;<sub>min</sub>, &eta;<sub>min</sub>), which prevents the damping from becoming
    either excessively strong or excessively weak: excessive damping slows convergence
    toward the Gauss&ndash;Newton solution, whereas insufficient damping permits unstable
    correction steps. Finally, if the resulting update step fails to reduce the misfit,
    the damping is increased progressively until a step is obtained that improves the
    model and decreases the misfit.
</div>
""")


# =============================================================================
# 3. MODEL EVALUATION
# =============================================================================

render_html('<div class="methods-section-title">3. Model Evaluation</div>')

render_html("""
<div class="methods-body">
    The quality of the data fit is evaluated using the Root Mean Square Error (RMSE),
    which is calculated from the difference between the observed data
    (<b>d<sub>obs</sub></b>) and the calculated data (<b>d<sub>cal</sub></b>) as
    expressed by the following Equation:
</div>
""")

st.latex(r"\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\frac{\mathbf{d}_{obs} - \mathbf{d}_{cal}}{\mathbf{d}_{obs}}\right)^{2}} \times 100\%")

render_html("""
<div class="methods-body">
    where <b>N</b> is the number of data points, and a smaller RMSE indicates a closer
    fit. Consistent with the inversion, the RMSE is evaluated in the natural-logarithm
    domain. Because the RMSE reflects the fit at a single iteration rather than the rate
    of progress, convergence is assessed from two further quantities, the relative
    misfit reduction and the relative model update, the latter defined as the norm of
    the change in the log-domain model vector normalized by &radic;(2n &minus; 1). The
    inversion is deemed converged once both fall below their tolerances,
    <b>1 &times; 10<sup>&minus;4</sup></b> and <b>1 &times; 10<sup>&minus;3</sup></b>,
    respectively, which are stated explicitly to ensure reproducibility.
    <br><br>
    Since convergence represents only the ideal case, the process is also safeguarded
    against non-convergence and stalling. A maximum iteration limit bounds the total
    computational effort. Each candidate update, formed from a trial damping value and a
    backtracked step length, is accepted only if it satisfies a sufficient-decrease
    (Armijo-type) condition, requiring the trial misfit to fall below the current misfit
    by at least a fraction <b>c &middot; &alpha;</b> of it, with
    <b>c = 10<sup>&minus;4</sup></b> and &alpha; the step length. An iteration is
    regarded as stagnant when no candidate meets this condition, leaving the model
    unchanged. As isolated stagnant iterations are common in nonlinear inversion,
    termination is triggered only after five consecutive occurrences a patience window
    that tolerates transient failures without prolonging a stalled run. Likewise, if the
    RMS error does not improve over five consecutive iterations, the process stops and
    the model reverts to the best solution obtained previously. These safeguards,
    together with the convergence test above, keep the inversion controlled, stable, and
    computationally efficient.
</div>
""")