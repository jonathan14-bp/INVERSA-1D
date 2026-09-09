"""
===============================================================================
INVERSA-1D
1D Geoelectric Resistivity Inversion

Interactive Streamlit application for the 1D inversion of Vertical Electrical
Sounding (VES) resistivity data. The application is built around an extensible
inversion framework; the inversion engines currently available are exposed
through the user interface.
===============================================================================
"""

from html import escape

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

from svd_core import MODEL_CONFIG, VES1DFWD, run_svd_inversion
from svd_core import add_noise as add_noise_svd
from lm_core import run_lm_inversion


# =============================================================================
# APPLICATION CONSTANTS
# =============================================================================

MODE_SYNTHETIC = "new_synthetic_data"
MODE_MANUAL = "manual_ves_point"
MODE_UPLOAD = "upload_ves_point"

INFINITY_SYMBOL = "∞"

MODE_LABELS = {
    MODE_SYNTHETIC: "New Synthetic Data",
    MODE_MANUAL: "Manual VES Point",
    MODE_UPLOAD: "Upload VES Point",
}


# =============================================================================
# REFERENCE MODEL CONSTANTS
# =============================================================================

REF_MODEL_HOM_10 = "Homogeneous 10 Ohm.m"
REF_MODEL_HOM_100 = "Homogeneous 100 Ohm.m"
REF_MODEL_HOM_1000 = "Homogeneous 1000 Ohm.m"

REFERENCE_MODEL_OPTIONS = [
    REF_MODEL_HOM_10,
    REF_MODEL_HOM_100,
    REF_MODEL_HOM_1000,
]

REFERENCE_MODEL_OPTIONS_BY_MODE = {
    MODE_SYNTHETIC: REFERENCE_MODEL_OPTIONS,
    MODE_MANUAL: REFERENCE_MODEL_OPTIONS,
    MODE_UPLOAD: REFERENCE_MODEL_OPTIONS,
}

DEFAULT_REFERENCE_MODEL_BY_MODE = {
    MODE_SYNTHETIC: REF_MODEL_HOM_100,
    MODE_MANUAL: REF_MODEL_HOM_100,
    MODE_UPLOAD: REF_MODEL_HOM_100,
}


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Home | INVERSA-1D",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# MATPLOTLIB GLOBAL STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.edgecolor": "#475569",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 150,
    "figure.dpi": 110,
})


# =============================================================================
# CUSTOM CSS 
# =============================================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}
    header {visibility: visible;}
    
    .app-title {
        color: #e74c3c;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
    }

    .app-subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }

    .section-header {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #e2e8f0;
    }

    .sublabel {
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
        margin-top: 0.9rem;
        margin-bottom: 0.3rem;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0rem 0rem;
        margin-bottom: 0.45rem;
    }

    .section-separator {
        width: 100%;
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent 0%,
            #e2e8f0 15%,
            #e2e8f0 85%,
            transparent 100%
        );
        margin: 0.6rem 0;
    }

    .custom-info-box {
        background: #eff6ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #2563eb;
        color: #1e3a8a;
        font-size: 0.88rem;
    }

    .custom-warning-box {
        background: #fffbeb;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #f59e0b;
        color: #92400e;
        font-size: 0.88rem;
    }

    .custom-success-box {
        background: #ecfdf5;
        color: #166534;
        padding: 0.55rem 0.9rem;
        border-radius: 6px;
        font-size: 0.84rem;
        font-weight: 600;
        border-left: 3px solid #16a34a;
    }

    .load-status-success {
        background: #ecfdf5;
        border: 1px solid rgba(22, 163, 74, 0.35);
        border-left: 3px solid #16a34a;
        color: #166534 !important;
        padding: 0.65rem 0.9rem;
        border-radius: 6px;
        font-size: 0.84rem;
        font-weight: 600;
        margin: 0.55rem 0;
    }

    .load-status-error {
        background: #fef2f2;
        border: 1px solid rgba(220, 38, 38, 0.35);
        border-left: 3px solid #dc2626;
        color: #991b1b !important;
        padding: 0.65rem 0.9rem;
        border-radius: 6px;
        font-size: 0.84rem;
        font-weight: 600;
        margin: 0.55rem 0;
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
# SESSION STATE INITIALIZATION
# =============================================================================
default_session_state = {
    "welcome_complete": False,
    "data_mode": None,
    "data_loaded": False,
    "inversion_done": False,
    "result": None,
    "method_used": None,
    "load_status": None,
    "load_message": None,

    # Reference model state
    "reference_model_choice": None,
    "reference_rho": None,

    # Synthetic data state
    "synthetic_n_layers_locked": None,
    "synthetic_true_model_r": None,
    "synthetic_true_model_t": None,
    
    # Synthetic sampling helper
    "synthetic_auto_sampling": True,
    "synthetic_curve_type": None,
    "synthetic_ab2_min_used": None,
    "synthetic_ab2_max_used": None,
}

for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =============================================================================
# UI STATE HELPERS
# =============================================================================
def reset_loaded_data_state():
    """Clear all loaded data, previews, and load-status flags from the session."""
    keys_to_remove = [
        "data_loaded",
        "inversion_done",
        "result",
        "method_used",
        "AB2",
        "Rho_app_clean",
        "Rho_app_noisy",
        "df_display",
        "noise_level",
        "model_name",
        "true_model_r",
        "true_model_t",
        "load_status",
        "load_message",
        "reference_model_choice",
        "reference_rho",
        "reference_model_selectbox",
        "manual_table",
        "manual_editor",
        "data_uploader",

        # Synthetic data keys
        "synthetic_model_table",
        "synthetic_model_snapshot",        
        "synthetic_model_editor",
        "synthetic_n_layers_input",
        "synthetic_n_layers_current",
        "synthetic_n_layers_locked",
        "synthetic_true_model_r",
        "synthetic_true_model_t",
        "synthetic_ab2_min",
        "synthetic_ab2_max",
        "synthetic_n_points",
        
        # Synthetic sampling helper
        "synthetic_auto_sampling",
        "synthetic_curve_type",
        "synthetic_ab2_min_used",
        "synthetic_ab2_max_used",
    ]

    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.data_loaded = False
    st.session_state.inversion_done = False
    st.session_state.result = None
    st.session_state.method_used = None
    st.session_state.load_status = None
    st.session_state.load_message = None
    st.session_state.reference_model_choice = None
    st.session_state.reference_rho = None
    st.session_state.synthetic_n_layers_locked = None
    st.session_state.synthetic_true_model_r = None
    st.session_state.synthetic_true_model_t = None
    st.session_state.synthetic_auto_sampling = True
    st.session_state.synthetic_curve_type = None
    st.session_state.synthetic_ab2_min_used = None
    st.session_state.synthetic_ab2_max_used = None


def clear_inversion_result():
    """Discard any previous inversion result when a new dataset is loaded or a load fails."""
    st.session_state.inversion_done = False
    st.session_state.result = None
    st.session_state.method_used = None


def go_to_welcome_screen():
    """Return to the welcome screen and reset all loaded data and results."""
    reset_loaded_data_state()
    st.session_state.welcome_complete = False
    st.session_state.data_mode = None
    st.rerun()

def select_data_mode(mode):
    """
    Activate the selected data input mode from the welcome screen and
    initialize the default reference model associated with that mode.
    """
    reset_loaded_data_state()
    st.session_state.welcome_complete = True
    st.session_state.data_mode = mode

    st.session_state.reference_model_choice = DEFAULT_REFERENCE_MODEL_BY_MODE.get(
        mode,
        REF_MODEL_HOM_100
    )

    st.rerun()

def set_load_status(is_success, message):
    st.session_state.load_status = "success" if is_success else "error"
    st.session_state.load_message = message

def render_load_status():
    status = st.session_state.get("load_status")
    message = st.session_state.get("load_message")

    if status is None or message is None:
        return

    css_class = (
        "load-status-success"
        if status == "success"
        else "load-status-error"
    )

    st.markdown(
        f"<div class='{css_class}'>{escape(str(message))}</div>",
        unsafe_allow_html=True
    )

def parse_float_value(value):
    """Parse a table cell into a float, tolerating common decimal and thousands separators."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "":
        return np.nan

    text = text.replace(" ", "")

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "")
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return np.nan

def resize_manual_table(df, target_rows):
    df_resized = df.copy()
    df_resized = df_resized.reindex(range(target_rows))
    df_resized = df_resized.fillna("")
    return df_resized

def create_default_synthetic_table(n_layers):
    rows = []

    for i in range(n_layers):
        rows.append({
            "Layer": str(i + 1),
            "Resistivity (Ohm.m)": "100",
            "Thickness (m)": INFINITY_SYMBOL if i == n_layers - 1 else "10"
        })

    return pd.DataFrame(rows)

def resize_synthetic_table(df, target_layers):
    """
    Resize synthetic model table while preserving existing resistivity
    and thickness values as much as possible.
    """
    default_df = create_default_synthetic_table(target_layers)

    if df is None or len(df) == 0:
        return default_df

    df_old = df.copy()
    n_copy = min(len(df_old), target_layers)

    for i in range(n_copy):
        if "Resistivity (Ohm.m)" in df_old.columns:
            default_df.loc[i, "Resistivity (Ohm.m)"] = df_old.loc[i, "Resistivity (Ohm.m)"]

        if i < target_layers - 1 and "Thickness (m)" in df_old.columns:
            old_thickness = str(df_old.loc[i, "Thickness (m)"]).strip()

            if old_thickness in [INFINITY_SYMBOL, "infinite", "Infinity", "inf", "Inf", ""]:
                default_df.loc[i, "Thickness (m)"] = "10"
            else:
                default_df.loc[i, "Thickness (m)"] = old_thickness

    default_df["Layer"] = [str(i + 1) for i in range(target_layers)]
    default_df.loc[target_layers - 1, "Thickness (m)"] = INFINITY_SYMBOL

    return default_df

def classify_basic_ves_curve_type(r):
    """
    Classify basic 3-layer VES curve type based on resistivity pattern.

    H: rho1 > rho2 < rho3
    K: rho1 < rho2 > rho3
    A: rho1 < rho2 < rho3
    Q: rho1 > rho2 > rho3

    This function only classifies the basic 3-layer case.
    """
    if r is None or len(r) != 3:
        return None

    rho1, rho2, rho3 = r

    if rho1 > rho2 and rho2 < rho3:
        return "H"
    elif rho1 < rho2 and rho2 > rho3:
        return "K"
    elif rho1 < rho2 and rho2 < rho3:
        return "A"
    elif rho1 > rho2 and rho2 > rho3:
        return "Q"
    else:
        return "Undetermined"

def estimate_synthetic_ab2_limits(t):
    if t is None or len(t) == 0:
        return 1.0, 100.0

    t = np.asarray(t, dtype=float)

    positive_t = t[t > 0]

    if len(positive_t) == 0:
        return 1.0, 100.0

    min_thickness = np.min(positive_t)
    total_depth_to_halfspace = np.sum(positive_t)

    # Minimum AB/2 should be small enough to sample the first layer response.
    recommended_min = max(0.001, min_thickness / 20.0)

    # Maximum AB/2 should be large enough to see the basement/half-space effect.
    recommended_max = max(100.0, total_depth_to_halfspace * 25.0)

    recommended_max = min(recommended_max, 1_000_000.0)

    if recommended_max <= recommended_min:
        recommended_max = recommended_min * 100.0

    return recommended_min, recommended_max

def render_section_separator():
    st.markdown(
        '<div class="section-separator"></div>',
        unsafe_allow_html=True
    )

def render_data_preview(df):
    df_preview = pd.DataFrame({
        "": np.arange(len(df)),
        "AB/2": df["AB_2"].values,
        "Rho Apparent": df["rho_app_obs"].values
    })
    styled_preview = (
        df_preview.style
        .format({
            "AB/2": "{:g}",
            "Rho Apparent": "{:.4f}"
        })
        .set_properties(
            subset=[""],
            **{
                "color": "#64748b",
                "font-weight": "600"
            }
        )
    )
    st.dataframe(
        styled_preview,
        use_container_width=True,
        height=280,
        hide_index=True
    )

# =============================================================================
# PLOT STYLE HELPERS
# =============================================================================
def collect_positive_finite_values(*arrays):
    values = []

    for arr in arrays:
        if arr is None:
            continue

        arr_np = np.asarray(arr, dtype=float).ravel()
        arr_np = arr_np[np.isfinite(arr_np)]
        arr_np = arr_np[arr_np > 0]

        if len(arr_np) > 0:
            values.append(arr_np)

    if len(values) == 0:
        return np.array([], dtype=float)

    return np.concatenate(values)


def adaptive_log_limits(*arrays, pad_fraction=0.08, min_decades=0.35):
    values = collect_positive_finite_values(*arrays)

    if len(values) == 0:
        return 1e-3, 1.0

    log_values = np.log10(values)

    log_min = np.min(log_values)
    log_max = np.max(log_values)

    if np.isclose(log_min, log_max):
        center = log_min
        half_span = min_decades / 2.0
        log_min = center - half_span
        log_max = center + half_span
    else:
        current_span = log_max - log_min

        if current_span < min_decades:
            center = 0.5 * (log_min + log_max)
            half_span = min_decades / 2.0
            log_min = center - half_span
            log_max = center + half_span
            current_span = min_decades

        pad = current_span * pad_fraction
        log_min -= pad
        log_max += pad

    lower = 10 ** log_min
    upper = 10 ** log_max

    if lower <= 0 or not np.isfinite(lower):
        lower = 1e-6

    if upper <= lower or not np.isfinite(upper):
        upper = lower * 10.0

    return lower, upper

def expanded_decade_log_limits(
    *arrays,
    extra_decades=1,
    lower_bound=1e-6,
    upper_bound=None
):

    values = collect_positive_finite_values(*arrays)

    if len(values) == 0:
        return 1.0, 1000.0

    log_values = np.log10(values)

    log_min_data = np.min(log_values)
    log_max_data = np.max(log_values)

    # Full decade limits from the data
    log_min_decade = np.floor(log_min_data)
    log_max_decade = np.ceil(log_max_data)

    # Expand by selected number of decades
    log_min = log_min_decade - extra_decades
    log_max = log_max_decade + extra_decades

    lower = 10 ** log_min
    upper = 10 ** log_max

    if lower_bound is not None:
        lower = max(lower, lower_bound)

    if upper_bound is not None:
        upper = min(upper, upper_bound)

    if upper <= lower:
        upper = lower * 100.0

    return lower, upper

def adaptive_linear_limits(values, pad_fraction=0.08, min_span=1.0, lower_bound=None):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0, 1.0

    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmin, vmax):
        span = max(abs(vmin) * 0.1, min_span)
        lower = vmin - span / 2.0
        upper = vmax + span / 2.0
    else:
        span = max(vmax - vmin, min_span)
        pad = span * pad_fraction
        lower = vmin - pad
        upper = vmax + pad

    if lower_bound is not None:
        lower = max(lower, lower_bound)

    if upper <= lower:
        upper = lower + min_span

    return lower, upper

def collect_depth_axis_data(*thickness_arrays):
    interface_depths = [0.0]
    all_thickness_values = []

    deepest_interface = 0.0
    controlling_last_thickness = None

    for thick in thickness_arrays:
        if thick is None:
            continue

        thick_np = np.asarray(thick, dtype=float).ravel()
        thick_np = thick_np[np.isfinite(thick_np)]
        thick_np = thick_np[thick_np > 0]

        if len(thick_np) == 0:
            continue

        cumulative_depth = np.cumsum(thick_np)

        interface_depths.extend(cumulative_depth.tolist())
        all_thickness_values.extend(thick_np.tolist())

        if cumulative_depth[-1] > deepest_interface:
            deepest_interface = float(cumulative_depth[-1])
            controlling_last_thickness = float(thick_np[-1])

    interface_depths = np.array(interface_depths, dtype=float)
    interface_depths = interface_depths[np.isfinite(interface_depths)]
    interface_depths = interface_depths[interface_depths >= 0]

    if len(interface_depths) == 0:
        interface_depths = np.array([0.0], dtype=float)

    interface_depths = np.unique(np.round(interface_depths, decimals=10))
    all_thickness_values = np.array(all_thickness_values, dtype=float)

    return (
        interface_depths,
        all_thickness_values,
        deepest_interface,
        controlling_last_thickness
    )

def adaptive_depth_max_for_earth_model(
    *thickness_arrays,
    pad_fraction=0.20,
    halfspace_extension_fraction=0.60,
    fallback_depth=1.0
):

    (
        interface_depths,
        all_thickness_values,
        deepest_interface,
        controlling_last_thickness
    ) = collect_depth_axis_data(*thickness_arrays)

    if deepest_interface <= 0 or len(all_thickness_values) == 0:
        return fallback_depth, interface_depths

    characteristic_thickness = float(np.median(all_thickness_values))

    if controlling_last_thickness is None:
        controlling_last_thickness = characteristic_thickness

    # Adaptive extension below deepest interface.
    bottom_padding = max(
        deepest_interface * pad_fraction,
        controlling_last_thickness * halfspace_extension_fraction,
        characteristic_thickness * 0.50
    )

    # Prevent too much empty space below the model.
    bottom_padding = min(
        bottom_padding,
        max(deepest_interface, characteristic_thickness * 2.0)
    )

    zmax = deepest_interface + bottom_padding

    if zmax <= deepest_interface:
        zmax = deepest_interface * 1.10

    if not np.isfinite(zmax) or zmax <= 0:
        zmax = fallback_depth

    return zmax, interface_depths

def adaptive_depth_ticks(interface_depths, zmax, max_ticks=12):
    if interface_depths is None:
        interface_depths = np.array([0.0], dtype=float)

    interface_depths = np.asarray(interface_depths, dtype=float)
    interface_depths = interface_depths[np.isfinite(interface_depths)]
    interface_depths = interface_depths[
        (interface_depths >= 0) & (interface_depths <= zmax)
    ]

    if len(interface_depths) == 0:
        interface_depths = np.array([0.0], dtype=float)

    interface_depths = np.unique(np.round(interface_depths, decimals=10))

    candidate_ticks = np.unique(
        np.concatenate([
            interface_depths,
            np.array([float(zmax)])
        ])
    )

    if len(candidate_ticks) <= max_ticks:
        return candidate_ticks

    internal_depths = candidate_ticks[
        (candidate_ticks > 0) & (candidate_ticks < zmax)
    ]

    n_internal_allowed = max_ticks - 2

    if n_internal_allowed <= 0 or len(internal_depths) == 0:
        return np.array([0.0, zmax], dtype=float)

    selected_indices = np.linspace(
        0,
        len(internal_depths) - 1,
        n_internal_allowed
    ).astype(int)

    selected_internal = internal_depths[selected_indices]

    ticks = np.unique(
        np.concatenate([
            np.array([0.0]),
            selected_internal,
            np.array([float(zmax)])
        ])
    )

    return ticks

def format_depth_tick(value):
    value = float(value)

    if abs(value) >= 1000:
        return f"{value:.0f}"
    elif abs(value) >= 100:
        return f"{value:.1f}"
    elif abs(value) >= 10:
        return f"{value:.2f}"
    elif abs(value) >= 1:
        return f"{value:.3f}"
    else:
        return f"{value:.4g}"

def style_loglog_plot(ax):
    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=1.0,
        alpha=0.75,
        color="#cbd5e1"
    )
    ax.grid(
        True,
        which="minor",
        linestyle="--",
        linewidth=0.7,
        alpha=0.55,
        color="#e2e8f0"
    )
    ax.tick_params(which="major", length=5, width=1.0)
    ax.tick_params(which="minor", length=3, width=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#475569")

def style_linear_plot(ax):
    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=1.0,
        alpha=0.75,
        color="#cbd5e1"
    )
    ax.grid(
        True,
        which="minor",
        linestyle="--",
        linewidth=0.6,
        alpha=0.55,
        color="#e2e8f0"
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#475569")

def get_next_nice_depth_step(current_step):
    current_step = float(current_step)

    if current_step <= 0 or not np.isfinite(current_step):
        return 10.0

    exponent = np.floor(np.log10(current_step))
    base = 10 ** exponent

    nice_steps = np.array([1.0, 2.0, 5.0, 10.0]) * base

    larger_steps = nice_steps[nice_steps > current_step * (1.0 + 1e-12)]

    if len(larger_steps) > 0:
        return float(larger_steps[0])

    return float(10.0 * base)

def regular_depth_axis(
    zmax,
    preferred_step=10.0,
    max_ticks=11
):

    if zmax is None or not np.isfinite(zmax) or zmax <= 0:
        step = float(preferred_step)
        axis_depth_max = step
        ticks = np.arange(0.0, axis_depth_max + step * 0.5, step)
        return axis_depth_max, ticks, step

    step = float(preferred_step)

    if step <= 0:
        step = 10.0

    axis_depth_max = np.ceil(float(zmax) / step) * step

    if axis_depth_max <= 0:
        axis_depth_max = step

    ticks = np.arange(0.0, axis_depth_max + step * 0.5, step)

    # If too many labels, increase interval automatically.
    while len(ticks) > max_ticks:
        step = get_next_nice_depth_step(step)
        axis_depth_max = np.ceil(float(zmax) / step) * step

        if axis_depth_max <= 0:
            axis_depth_max = step

        ticks = np.arange(0.0, axis_depth_max + step * 0.5, step)

    return axis_depth_max, ticks, step

def format_regular_depth_tick(value):
    value = float(value)

    if np.isclose(value, round(value)):
        return f"{int(round(value))}"

    return f"{value:g}"

def regular_log_decade_y_axis(
    *arrays,
    start_power=0,
    max_ticks=12
):

    values = collect_positive_finite_values(*arrays)

    ymin = 10.0 ** start_power

    if len(values) == 0:
        ymax = 10.0 ** (start_power + 1)
        ticks = 10.0 ** np.arange(start_power, start_power + 2)
        return ymin, ymax, ticks

    data_max = np.max(values)

    if not np.isfinite(data_max) or data_max <= 0:
        ymax = 10.0 ** (start_power + 1)
        ticks = 10.0 ** np.arange(start_power, start_power + 2)
        return ymin, ymax, ticks

    max_power = int(np.ceil(np.log10(data_max)))

    # Ensure upper limit is always above the lower limit.
    if max_power <= start_power:
        max_power = start_power + 1

    # If too many labels, keep decade style but increase the interval.
    tick_powers = np.arange(start_power, max_power + 1)

    if len(tick_powers) > max_ticks:
        step = int(np.ceil(len(tick_powers) / max_ticks))
        tick_powers = tick_powers[::step]

        if tick_powers[-1] != max_power:
            tick_powers = np.append(tick_powers, max_power)

    ymax = 10.0 ** max_power
    ticks = 10.0 ** tick_powers

    return ymin, ymax, ticks

def adaptive_log_decade_axis(
    *arrays,
    extra_decades=0,
    edge_margin_decades=0.05,
    max_ticks=12,
    lower_bound=1e-9,
    upper_bound=None
):

    values = collect_positive_finite_values(*arrays)

    if len(values) == 0:
        return 1.0, 10.0, np.array([1.0, 10.0])

    log_values = np.log10(values)
    log_min_data = float(np.min(log_values))
    log_max_data = float(np.max(log_values))

    # Decades that bracket the data.
    log_low = np.floor(log_min_data)
    log_high = np.ceil(log_max_data)

    # Push a side out by one decade when the extreme value lies on / very close
    # to a decade line, so the point is not drawn exactly on the border.
    if (log_min_data - log_low) < edge_margin_decades:
        log_low -= 1.0

    if (log_high - log_max_data) < edge_margin_decades:
        log_high += 1.0

    # Optional symmetric widening (Earth Model style).
    log_low -= extra_decades
    log_high += extra_decades

    lower = 10.0 ** log_low
    upper = 10.0 ** log_high

    if lower_bound is not None:
        lower = max(lower, lower_bound)

    if upper_bound is not None:
        upper = min(upper, upper_bound)

    if upper <= lower:
        upper = lower * 10.0

    # Integer-decade tick locations spanning the framed range.
    tick_low = int(np.floor(np.log10(lower) + 1e-9))
    tick_high = int(np.ceil(np.log10(upper) - 1e-9))

    if tick_high <= tick_low:
        tick_high = tick_low + 1

    tick_powers = np.arange(tick_low, tick_high + 1)

    # Keep the decade style but thin the labels if the span is very wide.
    if len(tick_powers) > max_ticks:
        step = int(np.ceil(len(tick_powers) / max_ticks))
        thinned = tick_powers[::step]
        if thinned[-1] != tick_powers[-1]:
            thinned = np.append(thinned, tick_powers[-1])
        tick_powers = thinned

    decade_ticks = 10.0 ** tick_powers

    return lower, upper, decade_ticks

# =============================================================================
# WELCOME SCREEN
# =============================================================================
if not st.session_state.get("welcome_complete", False):
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem 2rem 2rem;">
        <h1 style="color:#e74c3c; font-size:3.2rem; font-weight:800;
                   margin-bottom:1.5rem; letter-spacing:-0.03em;">
            INVERSA-1D
        </h1>
        <h3 style="color:#334155; font-size:1.4rem;
                   margin-bottom:3.2rem; font-weight:400;">
            1D Geoelectric Resistivity Inversion for Vertical Electrical Sounding (VES) Data
        </h3>
        <p style="color:#64748b; font-size:1.1rem;
                  max-width:760px; margin:0 auto 3rem;">
            Choose a VES data input mode below to begin the 1D resistivity inversion workflow.
        </p>
    </div>
    """, unsafe_allow_html=True)

    welcome_left, welcome_center, welcome_right = st.columns(
        [1, 1, 1],
        gap="large"
    )

    with welcome_left:
        if st.button(
            "NEW SYNTHETIC DATA",
            use_container_width=True,
            help="Generate a synthetic sounding curve from a user-defined layered earth model.",
            key="mode_new_synthetic_data"
        ):
            select_data_mode(MODE_SYNTHETIC)

    with welcome_center:
        if st.button(
            "MANUAL VES POINT",
            use_container_width=True,
            help="Enter or paste field VES data directly into the dataset table.",
            key="mode_manual_ves_point"
        ):
            select_data_mode(MODE_MANUAL)

    with welcome_right:
        if st.button(
            "UPLOAD VES POINT",
            use_container_width=True,
            help="Import VES data from an Excel file containing AB/2 and apparent resistivity columns.",
            key="mode_upload_ves_point"
        ):
            select_data_mode(MODE_UPLOAD)

    st.markdown("""
    <div style="text-align:center; padding-top:2rem;
                color:#64748b; font-size:0.9rem; line-height:1.8;">
        <b>New Synthetic Data</b>: generate a synthetic sounding curve from a layered earth model
        &nbsp;|&nbsp;
        <b>Manual VES Point</b>: enter or paste field data directly into the table
        &nbsp;|&nbsp;
        <b>Upload VES Point</b>: import an Excel file with <b>AB/2</b> and <b>Apparent Resistivity</b> columns
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# =============================================================================
# TOP BAR
# =============================================================================
top_col_title, top_col_reset = st.columns([8, 1.2], gap="large")

with top_col_title:
    st.markdown("""
    <div style="padding: 0.2rem 0 0.6rem 0;">
        <h1 class="app-title">INVERSA-1D</h1>
        <div class="app-subtitle">
            1D Geoelectric Resistivity Inversion
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col_reset:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Reset", use_container_width=True, key="reset_to_welcome"):
        go_to_welcome_screen()

st.divider()

# =============================================================================
# MAIN LAYOUT
# =============================================================================
col_dataset, col_method, col_result = st.columns(
    [1.05, 1.0, 2.4],
    gap="large"
)

# =============================================================================
# COLUMN 1: DATASET
# =============================================================================
with col_dataset:
    st.markdown(
        '<p class="section-header">Dataset</p>',
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # MODE: NEW SYNTHETIC DATA
    # -------------------------------------------------------------------------
    if st.session_state.data_mode == MODE_SYNTHETIC:
        st.success(
            "NEW SYNTHETIC DATA — Generate a synthetic sounding curve from a "
            "layered earth model defined by resistivity and thickness per layer."
        )

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            '<p class="sublabel">Number of Synthetic Layers</p>',
            unsafe_allow_html=True
        )

        n_layers_synthetic = st.number_input(
            "Number of synthetic layers",
            min_value=2,
            max_value=10,
            value=int(st.session_state.get("synthetic_n_layers_input", 3)),
            step=1,
            label_visibility="collapsed",
            key="synthetic_n_layers_input",
            help="Number of layers used to generate the synthetic VES model."
        )

        auto_sampling = st.checkbox(
            "Auto-adjust AB/2 range for clearer synthetic curve shape",
            value=True,
            key="synthetic_auto_sampling",
            help=(
                "Automatically extends the AB/2 range based on layer thickness. "
                "This helps H, K, A, and Q synthetic curves show their full trend, "
                "especially the deeper half-space response."
            )
        )

        n_layers_synthetic = int(n_layers_synthetic)

        if "synthetic_model_table" not in st.session_state:
            st.session_state.synthetic_model_table = create_default_synthetic_table(
                n_layers_synthetic
            )
            st.session_state.synthetic_n_layers_current = n_layers_synthetic

        if st.session_state.get("synthetic_n_layers_current") != n_layers_synthetic:
            source_table = st.session_state.get(
                "synthetic_model_snapshot",
                st.session_state.get("synthetic_model_table")
            )
            st.session_state.synthetic_model_table = resize_synthetic_table(
                source_table,
                n_layers_synthetic
            )
            st.session_state.synthetic_n_layers_current = n_layers_synthetic

            if "synthetic_model_editor" in st.session_state:
                del st.session_state["synthetic_model_editor"]

            st.rerun()

        st.markdown(
            '<p class="sublabel">Synthetic Model Parameters</p>',
            unsafe_allow_html=True
        )

        last_layer_idx = n_layers_synthetic - 1

        edited_synthetic_table = st.data_editor(
            st.session_state.synthetic_model_table,
            num_rows="fixed",
            use_container_width=True,
            height=260,
            hide_index=True,
            disabled=["Layer"],
            column_config={
                "Layer": st.column_config.TextColumn(
                    "Layer",
                    disabled=True,
                    help="Layer number. This column is locked."
                ),
                "Resistivity (Ohm.m)": st.column_config.TextColumn(
                    "Resistivity (Ohm.m)",
                    help="Layer resistivity in Ohm.m. Must be greater than zero."
                ),
                "Thickness (m)": st.column_config.TextColumn(
                    "Thickness (m)",
                    help="Layer thickness in meter. The last layer is treated as half-space."
                )
            },
            key="synthetic_model_editor"
        )

        st.session_state.synthetic_model_snapshot = edited_synthetic_table.copy()

        synthetic_model_for_generation = edited_synthetic_table.copy()
        for i in range(n_layers_synthetic):
            thickness_value = str(
                synthetic_model_for_generation.loc[i, "Thickness (m)"]
            ).strip()
            if i < last_layer_idx:
                if thickness_value in [INFINITY_SYMBOL, "infinite", "Infinity", "inf", "Inf", ""]:
                    synthetic_model_for_generation.loc[i, "Thickness (m)"] = "10"
            else:
                synthetic_model_for_generation.loc[i, "Thickness (m)"] = INFINITY_SYMBOL

        render_section_separator()

        st.markdown(
            '<p class="sublabel">Synthetic VES Sampling</p>',
            unsafe_allow_html=True
        )

        sampling_col1, sampling_col2 = st.columns(2)

        with sampling_col1:
            ab2_min = st.number_input(
                "AB/2 min",
                min_value=0.001,
                max_value=1_000_000.0,
                value=float(st.session_state.get("synthetic_ab2_min", 1.0)),
                step=1.0,
                key="synthetic_ab2_min",
                help="Minimum AB/2 spacing in meter."
            )

        with sampling_col2:
            ab2_max = st.number_input(
                "AB/2 max",
                min_value=0.001,
                max_value=1_000_000.0,
                value=float(st.session_state.get("synthetic_ab2_max", 100.0)),
                step=1.0,
                key="synthetic_ab2_max",
                help="Maximum AB/2 spacing in meter."
            )

        n_points = st.number_input(
            "Number of synthetic data points",
            min_value=5,
            max_value=300,
            value=int(st.session_state.get("synthetic_n_points", 30)),
            step=1,
            key="synthetic_n_points",
            help="Number of AB/2 samples generated in logarithmic spacing."
        )

        if st.button(
            "GENERATE SYNTHETIC DATA",
            type="primary",
            use_container_width=True,
            key="generate_synthetic_data"
        ):
            try:
                df_model = synthetic_model_for_generation.copy()

                resistivity_values = []
                thickness_values = []

                for i in range(n_layers_synthetic):
                    rho_value = parse_float_value(
                        df_model.loc[i, "Resistivity (Ohm.m)"]
                    )

                    if np.isnan(rho_value):
                        raise ValueError(
                            f"Invalid resistivity value at layer {i + 1}."
                        )

                    resistivity_values.append(float(rho_value))

                    if i < n_layers_synthetic - 1:
                        thickness_value = parse_float_value(
                            df_model.loc[i, "Thickness (m)"]
                        )

                        if np.isnan(thickness_value):
                            raise ValueError(
                                f"Invalid thickness value at layer {i + 1}."
                            )

                        thickness_values.append(float(thickness_value))

                r_synthetic = np.array(resistivity_values, dtype=float)
                t_synthetic = np.array(thickness_values, dtype=float)

                if np.any(r_synthetic <= 0):
                    st.session_state.data_loaded = False
                    clear_inversion_result()
                    set_load_status(
                        False,
                        "Failed to generate synthetic data. All resistivity values must be greater than zero."
                    )

                elif np.any(t_synthetic <= 0):
                    st.session_state.data_loaded = False
                    clear_inversion_result()
                    set_load_status(
                        False,
                        "Failed to generate synthetic data. All thickness values must be greater than zero."
                    )

                elif ab2_min <= 0 or ab2_max <= 0:
                    st.session_state.data_loaded = False
                    clear_inversion_result()
                    set_load_status(
                        False,
                        "Failed to generate synthetic data. AB/2 min and AB/2 max must be greater than zero."
                    )

                elif ab2_min >= ab2_max:
                    st.session_state.data_loaded = False
                    clear_inversion_result()
                    set_load_status(
                        False,
                        "Failed to generate synthetic data. AB/2 max must be greater than AB/2 min."
                    )

                else:
                    # Adaptive synthetic AB/2 sampling
                    # H, K, A, and Q curve shapes can be displayed more fully.
                    curve_type = classify_basic_ves_curve_type(r_synthetic)

                    ab2_min_used = float(ab2_min)
                    ab2_max_used = float(ab2_max)

                    if st.session_state.get("synthetic_auto_sampling", True):
                        recommended_min, recommended_max = estimate_synthetic_ab2_limits(
                            t_synthetic
                        )

                        ab2_min_used = min(ab2_min_used, recommended_min)
                        ab2_max_used = max(ab2_max_used, recommended_max)

                        ab2_min_used = max(ab2_min_used, 0.001)
                        ab2_max_used = min(ab2_max_used, 1_000_000.0)

                        if ab2_max_used <= ab2_min_used:
                            ab2_max_used = ab2_min_used * 100.0
                            ab2_max_used = min(ab2_max_used, 1_000_000.0)

                    AB2_synthetic = np.logspace(
                        np.log10(ab2_min_used),
                        np.log10(ab2_max_used),
                        int(n_points)
                    )

                    rho_app_synthetic = np.array([
                        VES1DFWD(
                            r_synthetic,
                            t_synthetic,
                            spacing
                        )
                        for spacing in AB2_synthetic
                    ])

                    df_display = pd.DataFrame({
                        "AB_2": AB2_synthetic,
                        "rho_app_obs": rho_app_synthetic
                    })

                    st.session_state.AB2 = df_display["AB_2"].values
                    st.session_state.Rho_app_clean = df_display["rho_app_obs"].values
                    st.session_state.df_display = df_display
                    st.session_state.data_loaded = True

                    st.session_state.synthetic_n_layers_locked = n_layers_synthetic
                    st.session_state.synthetic_true_model_r = r_synthetic
                    st.session_state.synthetic_true_model_t = t_synthetic

                    st.session_state.synthetic_curve_type = curve_type
                    st.session_state.synthetic_ab2_min_used = ab2_min_used
                    st.session_state.synthetic_ab2_max_used = ab2_max_used

                    clear_inversion_result()

                    curve_type_text = (
                        f" Basic 3-layer curve type: {curve_type}."
                        if curve_type is not None
                        else ""
                    )

                    sampling_text = (
                        f" AB/2 used: {ab2_min_used:g}–{ab2_max_used:g} m."
                    )

                    auto_text = ""

                    if st.session_state.get("synthetic_auto_sampling", True):
                        if (
                            ab2_min_used != float(ab2_min)
                            or ab2_max_used != float(ab2_max)
                        ):
                            auto_text = (
                                " Auto sampling was applied to better display "
                                "the full synthetic curve response."
                            )

                    set_load_status(
                        True,
                        f"{len(df_display)} synthetic data points generated successfully "
                        f"from {n_layers_synthetic} layers."
                        f"{curve_type_text}"
                        f"{sampling_text}"
                        f"{auto_text}"
                    )

            except Exception as e:
                st.session_state.data_loaded = False
                clear_inversion_result()
                set_load_status(
                    False,
                    f"Failed to generate synthetic data. Error: {e}"
                )

        st.markdown("</div>", unsafe_allow_html=True)

        render_load_status()

        if (
            st.session_state.get("data_loaded")
            and st.session_state.get("df_display") is not None
        ):
            df = st.session_state.df_display
            render_data_preview(df)

    # -------------------------------------------------------------------------
    # MODE: MANUAL VES POINT
    # -------------------------------------------------------------------------
    elif st.session_state.data_mode == MODE_MANUAL:
        st.info(
            "MANUAL VES POINT — Enter or paste field VES data directly "
            "into the dataset table."
        )

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        if "manual_table" not in st.session_state:
            st.session_state.manual_table = pd.DataFrame({
                "AB/2": [""] * 30,
                "Rho_app_obs": [""] * 30
            })

        row_col1, row_col2 = st.columns([2, 1])

        with row_col1:
            n_rows_prepare = st.number_input(
                "Number of table rows",
                min_value=1,
                max_value=2000,
                value=max(len(st.session_state.manual_table), 30),
                step=1,
                help="Prepare the number of rows before pasting a large dataset."
            )

        with row_col2:
            st.markdown(
                "<div style='height:1.7rem;'></div>",
                unsafe_allow_html=True
            )

            if st.button(
                "Apply",
                use_container_width=True,
                key="prepare_manual_rows"
            ):
                st.session_state.manual_table = resize_manual_table(
                    st.session_state.manual_table,
                    int(n_rows_prepare)
                )

                if "manual_editor" in st.session_state:
                    del st.session_state["manual_editor"]

                st.rerun()

        edited_table = st.data_editor(
            st.session_state.manual_table,
            num_rows="dynamic",
            use_container_width=True,
            height=280,
            hide_index=False,
            column_config={
                "AB/2": st.column_config.TextColumn(
                    "AB/2",
                    help="AB/2 value in meters."
                ),
                "Rho_app_obs": st.column_config.TextColumn(
                    "Rho Apparent",
                    help="Apparent resistivity value in Ohm.m."
                )
            },
            key="manual_editor"
        )

        st.session_state.manual_table = edited_table.copy()

        if st.button(
            "LOAD DATA",
            type="primary",
            use_container_width=True,
            key="load_manual_data"
        ):
            try:
                df_temp = edited_table.copy()

                df_temp["AB/2_num"] = df_temp["AB/2"].apply(parse_float_value)
                df_temp["Rho_app_obs_num"] = df_temp["Rho_app_obs"].apply(parse_float_value)

                df_valid = df_temp.dropna(
                    subset=["AB/2_num", "Rho_app_obs_num"]
                ).copy()

                if len(df_valid) == 0:
                    st.session_state.data_loaded = False
                    clear_inversion_result()
                    set_load_status(
                        False,
                        "Failed to load data. The table is empty or contains invalid numeric values."
                    )

                elif (df_valid["AB/2_num"] <= 0).any():
                    st.session_state.data_loaded = False
                    clear_inversion_result()
                    set_load_status(
                        False,
                        "Failed to load data. All AB/2 values must be greater than zero."
                    )

                elif (df_valid["Rho_app_obs_num"] <= 0).any():
                    st.session_state.data_loaded = False
                    clear_inversion_result()
                    set_load_status(
                        False,
                        "Failed to load data. All apparent resistivity values must be greater than zero."
                    )

                else:
                    df_valid = df_valid.reset_index(drop=True)

                    df_display = pd.DataFrame({
                        "AB_2": df_valid["AB/2_num"].astype(float).values,
                        "rho_app_obs": df_valid["Rho_app_obs_num"].astype(float).values
                    })

                    st.session_state.AB2 = df_display["AB_2"].values
                    st.session_state.Rho_app_clean = df_display["rho_app_obs"].values
                    st.session_state.df_display = df_display
                    st.session_state.data_loaded = True

                    clear_inversion_result()

                    set_load_status(
                        True,
                        f"{len(df_display)} data points loaded successfully."
                    )

            except Exception as e:
                st.session_state.data_loaded = False
                clear_inversion_result()
                set_load_status(
                    False,
                    f"Failed to load data. Error: {e}"
                )

        st.markdown("</div>", unsafe_allow_html=True)

        render_load_status()

        if (
            st.session_state.get("data_loaded")
            and st.session_state.get("df_display") is not None
        ):
            df = st.session_state.df_display
            render_data_preview(df)

    # -------------------------------------------------------------------------
    # MODE: UPLOAD VES POINT
    # -------------------------------------------------------------------------
    elif st.session_state.data_mode == MODE_UPLOAD:
        st.warning(
            "UPLOAD VES POINT — Import an Excel file containing AB/2 "
            "and apparent resistivity columns."
        )

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Excel file",
            type=["xlsx", "xls"],
            label_visibility="collapsed",
            key="data_uploader"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                df.columns = [col.strip().lower() for col in df.columns]

                col_ab2 = None
                col_rho = None

                for col in df.columns:
                    if "ab" in col:
                        col_ab2 = col
                    if "rho" in col or "res" in col:
                        col_rho = col

                if col_ab2 is None or col_rho is None:
                    st.session_state.data_loaded = False
                    clear_inversion_result()
                    set_load_status(
                        False,
                        "Failed to load data. AB/2 or Rho/Res column was not found."
                    )

                else:
                    df_clean = df[[col_ab2, col_rho]].dropna().reset_index(drop=True)

                    df_display = pd.DataFrame({
                        "AB_2": df_clean[col_ab2].values.astype(float),
                        "rho_app_obs": df_clean[col_rho].values.astype(float)
                    })

                    if (df_display["AB_2"] <= 0).any():
                        st.session_state.data_loaded = False
                        clear_inversion_result()
                        set_load_status(
                            False,
                            "Failed to load data. All AB/2 values must be greater than zero."
                        )

                    elif (df_display["rho_app_obs"] <= 0).any():
                        st.session_state.data_loaded = False
                        clear_inversion_result()
                        set_load_status(
                            False,
                            "Failed to load data. All apparent resistivity values must be greater than zero."
                        )

                    else:
                        st.session_state.AB2 = df_display["AB_2"].values
                        st.session_state.Rho_app_clean = df_display["rho_app_obs"].values
                        st.session_state.df_display = df_display
                        st.session_state.data_loaded = True

                        clear_inversion_result()

                        set_load_status(
                            True,
                            f"{len(df_display)} data points loaded successfully."
                        )

            except Exception as e:
                st.session_state.data_loaded = False
                clear_inversion_result()
                set_load_status(
                    False,
                    f"Failed to load data. Error: {e}"
                )

        st.markdown("</div>", unsafe_allow_html=True)

        render_load_status()

        if (
            st.session_state.get("data_loaded")
            and st.session_state.get("df_display") is not None
        ):
            df = st.session_state.df_display
            render_data_preview(df)

        elif (
            not st.session_state.get("data_loaded")
            and st.session_state.get("load_status") is None
        ):
            st.warning("No file has been uploaded. Please upload an Excel file to continue.")        

# =============================================================================
# COLUMN 2: INVERSION METHODS
# =============================================================================
with col_method:
    st.markdown(
        '<p class="section-header">Inversion Methods</p>',
        unsafe_allow_html=True
    )

    if st.session_state.data_loaded:
        st.markdown(
            '<p class="sublabel">Inversion Method</p>',
            unsafe_allow_html=True
        )

        method = st.selectbox(
            "method",
            options=["SVD", "LM"],
            index=0,
            label_visibility="collapsed",
            help="SVD: Singular Value Decomposition | LM: Levenberg-Marquardt"
        )

        st.markdown(
            '<p class="sublabel">Reference Model</p>',
            unsafe_allow_html=True
        )

        current_data_mode = st.session_state.get("data_mode")

        reference_model_options = REFERENCE_MODEL_OPTIONS_BY_MODE.get(
            current_data_mode,
            REFERENCE_MODEL_OPTIONS
        )

        default_reference_model = st.session_state.get("reference_model_choice")

        if default_reference_model not in reference_model_options:
            default_reference_model = DEFAULT_REFERENCE_MODEL_BY_MODE.get(
                current_data_mode,
                REF_MODEL_HOM_100
            )

        default_reference_index = (
            reference_model_options.index(default_reference_model)
            if default_reference_model in reference_model_options
            else 0
        )

        model_choice = st.selectbox(
            "model",
            options=reference_model_options,
            index=default_reference_index,
            label_visibility="collapsed",
            disabled=False,
            help=(
                "Select the homogeneous reference model used as the initial "
                "resistivity condition for inversion."
            ),
            key="reference_model_selectbox"
        )

        st.session_state.reference_model_choice = model_choice

        st.markdown(
            '<p class="sublabel">Noise Test</p>',
            unsafe_allow_html=True
        )

        noise_choice = st.selectbox(
            "noise",
            options=[
                "No Noise (0%)",
                "Low Noise (5%)",
                "Medium Noise (10%)",
                "High Noise (15%)"
            ],
            index=0,
            label_visibility="collapsed"
        )

        noise_map = {
            "No Noise (0%)": 0,
            "Low Noise (5%)": 5,
            "Medium Noise (10%)": 10,
            "High Noise (15%)": 15
        }

        noise_level = noise_map[noise_choice]

        st.markdown(
            '<p class="sublabel">Number of Iterations</p>',
            unsafe_allow_html=True
        )

        max_iter = st.number_input(
            "iter",
            min_value=1,
            max_value=1000,
            value=50,
            step=1,
            label_visibility="collapsed"
        )

        config = MODEL_CONFIG[model_choice]

        st.markdown(
            '<p class="sublabel">Number of Layers</p>',
            unsafe_allow_html=True
        )

        if (
            st.session_state.data_mode == MODE_SYNTHETIC
            and st.session_state.get("synthetic_n_layers_locked") is not None
        ):
            default_n_layers = int(st.session_state.synthetic_n_layers_locked)
        else:
            default_n_layers = 3

        n_layers_input = st.number_input(
            "n_layer",
            min_value=2,
            max_value=10,
            value=default_n_layers,
            step=1,
            label_visibility="collapsed",
            help="Jumlah lapisan untuk model inversi. Dapat diubah bebas, "
                 "termasuk pada mode New Synthetic Data."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        run_button = st.button(
            "RUN INVERSION",
            use_container_width=True
        )

        reference_rho = config.get("reference_rho", None)

        if reference_rho is not None:
            st.success(
                f"Selected reference model: homogeneous {reference_rho:g} Ohm.m. "
                "This value is used as the initial homogeneous resistivity model."
            )
        else:
            st.info(
                "Homogeneous reference model is active. "
                "Initial model is generated automatically."
            )

    else:
        st.info("Load a dataset to enable inversion parameters.")

        run_button = False
        method = None
        model_choice = REF_MODEL_HOM_100
        noise_level = 0
        max_iter = 0
        n_layers_input = 3


# =============================================================================
# INVERSION EXECUTION
# =============================================================================
if run_button and st.session_state.data_loaded:
    config = MODEL_CONFIG[model_choice]
    reference_rho = config.get("reference_rho", None)

    AB2 = st.session_state.AB2
    Rho_clean = st.session_state.Rho_app_clean

    Rho_app = add_noise_svd(
        Rho_clean,
        noise_percent=noise_level,
        seed=42
    )

    with col_result:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        progress_bar = progress_placeholder.progress(0)

        status_text = status_placeholder.info(
            f"Starting {method} inversion..."
        )

        def update_progress(it, max_it, rms):
            pct = min(it / max_it, 1.0)
            progress_bar.progress(pct)

            status_placeholder.info(
                f"{method} iteration {it}/{max_it} | RMS: {rms:.4f}%"
            )

        if method == "SVD":
            result = run_svd_inversion(
                AB2=AB2,
                Rho_app=Rho_app,
                n_layers=n_layers_input,
                max_iter=max_iter,
                reference_rho=reference_rho,
                progress_callback=update_progress,
                target_rms=noise_level,
            )

        elif method == "LM":
            result = run_lm_inversion(
                AB2=AB2,
                Rho_app=Rho_app,
                n_layers=n_layers_input,
                max_iter=max_iter,
                reference_rho=reference_rho,
                progress_callback=update_progress,
                target_rms=noise_level,
            )

        else:
            st.error(f"Unknown inversion method: {method}")
            st.stop()

        progress_placeholder.empty()
        status_placeholder.empty()

    st.session_state.result = result
    st.session_state.Rho_app_noisy = Rho_app
    st.session_state.noise_level = noise_level
    st.session_state.model_name = model_choice
    st.session_state.reference_rho = reference_rho

    if st.session_state.data_mode == MODE_SYNTHETIC:
        st.session_state.true_model_r = st.session_state.get(
            "synthetic_true_model_r"
        )
        st.session_state.true_model_t = st.session_state.get(
            "synthetic_true_model_t"
        )
    else:
        st.session_state.true_model_r = config["true_model_r"]
        st.session_state.true_model_t = config["true_model_t"]

    st.session_state.method_used = method
    st.session_state.inversion_done = True


# =============================================================================
# COLUMN 3: RESULTS
# =============================================================================
with col_result:
    if st.session_state.inversion_done:
        tab_inversion, tab_model = st.tabs([
            "Inversion Result",
            "Model Result"
        ])

        result = st.session_state.result
        AB2 = st.session_state.AB2
        Rho_noisy = st.session_state.Rho_app_noisy
        noise_lv = st.session_state.noise_level
        model_nm = st.session_state.model_name
        reference_rho_result = st.session_state.get("reference_rho", None)
        true_r = st.session_state.true_model_r
        true_t = st.session_state.true_model_t
        method_nm = st.session_state.method_used

        r_final = result["r_final"]
        t_final = result["t_final"]
        rho_final = result["rho_final"]

        # ---------------------------------------------------------------------
        # TAB 1: INVERSION RESULT
        # ---------------------------------------------------------------------
        with tab_inversion:
            row_plots = st.columns([1, 1, 1], gap="small")

            with row_plots[0]:
                st.markdown("### Resistivity Curve")

                fig1, ax1 = plt.subplots(figsize=(5.4, 4.3))

                ax1.loglog(
                    AB2,
                    Rho_noisy,
                    "o",
                    color="darkred",
                    markersize=5,
                    markerfacecolor="red",
                    markeredgewidth=1.2,
                    linewidth=0,
                    label="Observed"
                )

                ax1.loglog(
                    AB2,
                    rho_final,
                    "-",
                    color="blue",
                    linewidth=2.0,
                    label="Calculated"
                )

                xmin, xmax = adaptive_log_limits(
                    AB2,
                    pad_fraction=0.06,
                    min_decades=0.5
                )

                ax1.set_xlim(xmin, xmax)

                ymin, ymax, y_ticks = adaptive_log_decade_axis(
                    Rho_noisy,
                    rho_final,
                    extra_decades=0,
                    edge_margin_decades=0.05,
                    max_ticks=12
                )

                ax1.set_ylim(ymin, ymax)
                ax1.set_yticks(y_ticks)

                ax1.yaxis.set_major_formatter(
                    mticker.LogFormatterMathtext(base=10)
                )

                ax1.yaxis.set_minor_locator(
                    mticker.LogLocator(
                        base=10,
                        subs=np.arange(2, 10) * 0.1,
                        numticks=100
                    )
                )

                ax1.yaxis.set_minor_formatter(
                    mticker.NullFormatter()
                )

                title_str = (
                    f"{method_nm}\n"
                    f"Noise: {noise_lv}% | "
                    f"RMS: {result['best_rms']:.2f}% | "
                    f"Best Iteration: {result['best_iter']}"
                )

                ax1.set_title(
                    title_str,
                    fontsize=9,
                    fontweight="bold",
                    pad=10
                )

                ax1.set_xlabel(
                    "AB/2 (m)",
                    fontsize=9,
                    fontweight="bold"
                )

                ax1.set_ylabel(
                    "Apparent Resistivity (Ohm.m)",
                    fontsize=9,
                    fontweight="bold"
                )

                style_loglog_plot(ax1)
                ax1.legend(loc="best", fontsize=8)

                plt.tight_layout()
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)

            with row_plots[1]:
                st.markdown("### Earth Model")

                fig2, ax2 = plt.subplots(figsize=(5.4, 4.3))

                has_true = (true_r is not None) and (true_t is not None)

                if has_true:
                    zmax, depth_interfaces = adaptive_depth_max_for_earth_model(
                        t_final,
                        true_t,
                        pad_fraction=0.20,
                        halfspace_extension_fraction=0.60,
                        fallback_depth=10.0
                    )
                else:
                    zmax, depth_interfaces = adaptive_depth_max_for_earth_model(
                        t_final,
                        pad_fraction=0.20,
                        halfspace_extension_fraction=0.60,
                        fallback_depth=10.0
                    )

                zmax_axis, depth_ticks, depth_tick_step = regular_depth_axis(
                    zmax,
                    preferred_step=10.0,
                    max_ticks=11
                )

                def build_layer_step(res, thick, zmax_plot):
                    z_interfaces = np.concatenate(([0.0], np.cumsum(thick)))

                    x = [res[0]]
                    z = [0.0]

                    for i in range(len(thick)):
                        x.append(res[i])
                        z.append(z_interfaces[i + 1])

                        x.append(res[i + 1])
                        z.append(z_interfaces[i + 1])

                    x.append(res[-1])
                    z.append(zmax_plot)

                    return np.array(x), np.array(z)

                x_inv, z_inv = build_layer_step(
                    r_final,
                    t_final,
                    zmax_axis
                )

                if has_true:
                    x_true, z_true = build_layer_step(
                        true_r,
                        true_t,
                        zmax_axis
                    )

                    ax2.plot(
                        x_true,
                        z_true,
                        "--",
                        color="red",
                        linewidth=2.0,
                        label="True Model"
                    )

                ax2.plot(
                    x_inv,
                    z_inv,
                    "-",
                    color="blue",
                    linewidth=2.5,
                    label=method_nm
                )

                ax2.set_xscale("log")

                if has_true:
                    all_r = np.concatenate([true_r, r_final])
                else:
                    all_r = r_final

                rxmin, rxmax = expanded_decade_log_limits(
                    all_r,
                    extra_decades=1,
                    lower_bound=1e-3,
                    upper_bound=1e9
                )

                ax2.set_xlim(rxmin, rxmax)

                ax2.set_ylim(zmax_axis, 0)

                ax2.set_yticks(depth_ticks)
                ax2.set_yticklabels([
                    format_regular_depth_tick(depth) for depth in depth_ticks
                ])

                ax2.tick_params(
                    axis="y",
                    labelsize=8,
                    pad=2
                )

                for depth in depth_ticks:
                    if depth > 0 and depth < zmax_axis:
                        ax2.axhline(
                            y=depth,
                            color="#e2e8f0",
                            linestyle="--",
                            linewidth=0.7,
                            alpha=0.8,
                            zorder=0
                        )

                ax2.set_xlabel(
                    "Resistivity (Ohm.m)",
                    fontsize=9,
                    fontweight="bold"
                )

                ax2.set_ylabel(
                    "Depth (m)",
                    fontsize=9,
                    fontweight="bold"
                )

                ax2.set_title(
                    method_nm,
                    fontsize=10,
                    fontweight="bold",
                    pad=10
                )

                style_loglog_plot(ax2)
                ax2.legend(loc="best", fontsize=8)

                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            with row_plots[2]:
                st.markdown("### RMS Error")

                b = result["rms_history"]

                if len(b) > 0 and b.shape[0] > 1:
                    fig3, ax3 = plt.subplots(figsize=(5.4, 4.0))

                    ax3.plot(
                        b[:, 1],
                        b[:, 0],
                        "r-o",
                        linewidth=2,
                        markersize=4
                    )

                    x_min, x_max = adaptive_linear_limits(
                        b[:, 1],
                        pad_fraction=0.05,
                        min_span=1.0,
                        lower_bound=0.0
                    )

                    y_min, y_max = adaptive_linear_limits(
                        b[:, 0],
                        pad_fraction=0.12,
                        min_span=max(result["best_rms"] * 0.1, 0.01),
                        lower_bound=0.0
                    )

                    ax3.set_xlim(x_min, x_max)
                    ax3.set_ylim(y_min, y_max)

                    ax3.set_title(
                        f"RMS: {result['best_rms']:.2f}%",
                        fontsize=10,
                        fontweight="bold",
                        pad=10
                    )

                    ax3.set_xlabel(
                        "Iteration",
                        fontsize=9,
                        fontweight="bold"
                    )

                    ax3.set_ylabel(
                        "RMS Error (%)",
                        fontsize=9,
                        fontweight="bold"
                    )

                    style_linear_plot(ax3)

                    ax3.axhline(
                        y=result["best_rms"],
                        color="green",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Best: {result['best_rms']:.2f}%"
                    )

                    ax3.legend(fontsize=8)

                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                    plt.close(fig3)

                else:
                    st.warning("RMS history is not available.")

            st.divider()

            metric_col1, metric_col2, metric_col3 = st.columns(
                [1, 2, 1],
                gap="small"
            )

            with metric_col1:
                st.empty()

            with metric_col2:
                b = result["rms_history"]

                if len(b) > 0 and b.shape[0] > 1:
                    col_metrics1, col_metrics2 = st.columns(2)

                    with col_metrics1:
                        st.metric(
                            "Initial RMS",
                            f"{b[0, 0]:.2f}%"
                        )

                    with col_metrics2:
                        reduction = (
                            (b[0, 0] - result["best_rms"])
                            / max(b[0, 0], 1e-12)
                            * 100
                        )

                        st.metric(
                            "Reduction",
                            f"{reduction:.1f}%"
                        )

                else:
                    st.info("RMS metrics are not available.")

            with metric_col3:
                st.empty()

        # ---------------------------------------------------------------------
        # TAB 2: MODEL RESULT
        # ---------------------------------------------------------------------
        with tab_model:
            n_layers = result["n_layers"]
            lt = result["lt"]

            inv_col1, inv_col2 = st.columns([1.4, 1], gap="large")

            with inv_col1:
                st.markdown("### Final Inversion Model")

                table_rows = []

                for i in range(n_layers):
                    table_rows.append({
                        "Layer": i + 1,
                        "Resistivity (Ohm.m)": f"{r_final[i]:.2f}",
                        "Thickness (m)": f"{t_final[i]:.2f}" if i < lt else INFINITY_SYMBOL
                    })

                df_result = pd.DataFrame(table_rows)

                st.dataframe(
                    df_result,
                    use_container_width=True,
                    hide_index=True
                )

                if true_r is not None:
                    st.markdown("### Comparison with True Model")

                    compare_rows = []
                    lt_true = len(true_t) if true_t is not None else 0
                    n_compare = min(n_layers, len(true_r))

                    for i in range(n_compare):
                        compare_rows.append({
                            "Layer": i + 1,
                            "True Rho": f"{true_r[i]:.2f}",
                            "Inverted Rho": f"{r_final[i]:.2f}",
                            "True Thickness": f"{true_t[i]:.2f}" if i < lt_true else INFINITY_SYMBOL,
                            "Inverted Thickness": f"{t_final[i]:.2f}" if i < lt else INFINITY_SYMBOL
                        })

                    df_compare = pd.DataFrame(compare_rows)

                    st.dataframe(
                        df_compare,
                        use_container_width=True,
                        hide_index=True
                    )

            with inv_col2:
                st.markdown("### Configuration")

                reference_rho_text = (
                    f"{reference_rho_result:g} Ohm.m"
                    if reference_rho_result is not None
                    else "-"
                )

                st.info(
                    f"Method: {method_nm}\n\n"
                    f"Reference model: {model_nm}\n\n"
                    f"Reference rho: {reference_rho_text}\n\n"
                    f"Noise: {noise_lv}%\n\n"
                    f"Number of layers: {n_layers}\n\n"
                    f"Model parameters: {n_layers + (n_layers - 1)}"
                )

                st.markdown("### Statistics")

                st.metric(
                    "Lowest RMS",
                    f"{result['best_rms']:.10f}%"
                )

                sc1, sc2 = st.columns(2)

                with sc1:
                    st.metric(
                        "Run Iteration",
                        f"{result['iteration_done']}"
                    )

                with sc2:
                    st.metric(
                        "Best Iteration",
                        f"{result['best_iter']}"
                    )

                if result["iteration_done"] < result["max_iter"]:
                    st.success("Converged")
                else:
                    st.warning("Maximum iteration reached")

                if len(result["log_messages"]) > 0:
                    with st.expander("Process Log"):
                        for msg in result["log_messages"]:
                            st.text(msg)

    elif st.session_state.data_loaded:
        st.info(
            "Set the inversion parameters in the middle column, select "
            "SVD or LM, then click RUN INVERSION to start the inversion process."
        )

    else:
        st.warning(
            "Load a VES dataset from the left column to start the inversion process."
        )                