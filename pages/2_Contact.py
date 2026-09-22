"""
===============================================================================
CONTACT PAGE - INVERSA-1D
Development team and academic contact information.
===============================================================================
"""

from html import escape
from textwrap import dedent

import streamlit as st


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Contact | INVERSA-1D",
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

ARTICLE_TITLE = (
    "INVERSA-1D: A Web-based Adaptive SVD and Levenberg\u2013Marquardt "
    "Framework for 1D Resistivity Inversion"
)
ARTICLE_AUTHORS = (
    "Jonathan Bilian Putra, Selvi Misnia Irawati, Wahyu Eko Junian, "
    "Risky Martin Antosia, Asido Saputra Sigalingging"
)
JOURNAL = "Phi: Jurnal Pendidikan Fisika dan Terapan"
JOURNAL_ISSN = "Print ISSN 2460-4348 \u00b7 Online ISSN 2549-7162"
ARTICLE_STATUS = "Manuscript in the publication process"

CONTACT_EMAIL_DEVELOPER = "jonathan.bilian14@gmail.com"
CONTACT_EMAIL_SUPERVISOR_1 = "selvi.irawati@tg.itera.ac.id"
CONTACT_EMAIL_SUPERVISOR_2 = "wahyu.junian@tg.itera.ac.id"
CONTACT_EMAIL_EXAMINER_1 = "martin.antosia@tg.itera.ac.id"
CONTACT_EMAIL_EXAMINER_2 = "asido.saputra@tg.itera.ac.id"

# =============================================================================
# DEVELOPMENT TEAM
# =============================================================================
# All members are affiliated with the same study program and institution, so the
# affiliation is rendered from the constants above rather than repeated here.
# Email addresses are listed only for the members available for correspondence.

TEAM = [
    {
        "name": "Jonathan Bilian Putra",
        "role": "Student",
        "email": CONTACT_EMAIL_DEVELOPER,
    },
    {
        "name": "Selvi Misnia Irawati",
        "role": "Lecturer",
        "email": CONTACT_EMAIL_SUPERVISOR_1,
    },
    {
        "name": "Wahyu Eko Junian",
        "role": "Lecturer",
        "email": CONTACT_EMAIL_SUPERVISOR_2,
    },
    {
        "name": "Risky Martin Antosia",
        "role": "Lecturer",
        "email": CONTACT_EMAIL_EXAMINER_1,
    },
    {
        "name": "Asido Saputra Sigalingging",
        "role": "Lecturer",
        "email": CONTACT_EMAIL_EXAMINER_2,
    },
]


# =============================================================================
# CUSTOM CSS  (layout only; colours follow the active Streamlit theme)
# =============================================================================

render_html("""
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
    .contact-title {
        color: #e74c3c;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.03em;
    }

    .contact-subtitle {
        color: #64748b;
        font-size: 0.98rem;
        line-height: 1.75;
        text-align: justify;
        max-width: 900px;
        margin-bottom: 1.3rem;
    }

    .contact-rule {
        width: 100%;
        height: 1px;
        background: linear-gradient(
            90deg, transparent 0%, #e2e8f0 12%, #e2e8f0 88%, transparent 100%
        );
        margin: 0.4rem 0 1.5rem 0;
    }

    /* ---- Section titles ---- */
    .section-title {
        font-size: 1.28rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 1.9rem 0 0.55rem 0;
        padding-bottom: 0.45rem;
        border-bottom: 2px solid #e74c3c;
    }

    /* ---- Body prose ---- */
    .contact-body {
        font-size: 0.95rem;
        line-height: 1.8;
        text-align: justify;
        max-width: 900px;
        margin-bottom: 0.6rem;
    }

    /* ---- Team grid ---- */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
        gap: 0.9rem;
        margin-top: 0.9rem;
    }

    .team-card {
        display: flex;
        flex-direction: column;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #e74c3c;
        border-radius: 10px;
        padding: 1.05rem 1.15rem 1.1rem 1.15rem;
    }

    .team-name {
        font-size: 1.02rem;
        font-weight: 800;
        line-height: 1.35;
    }

    .team-role {
        align-self: flex-start;
        color: #b91c1c;
        background: #fef2f2;
        border: 1px solid #fecaca;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.14rem 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
    }

    .team-affil {
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.6;
        margin-top: 0.6rem;
    }

    .team-email {
        margin-top: auto;
        padding-top: 0.7rem;
        font-size: 0.85rem;
        word-break: break-word;
    }

    .team-email-label {
        display: block;
        color: #94a3b8;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.15rem;
    }

    .team-email a {
        text-decoration: none;
        font-weight: 700;
    }

    .team-email a:hover {
        text-decoration: underline;
    }

    /* ---- Reference box ---- */
    .reference-box {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        max-width: 900px;
        margin-top: 0.7rem;
    }

    .reference-title {
        font-size: 1.0rem;
        font-weight: 700;
        line-height: 1.5;
    }

    .reference-authors {
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    .reference-meta {
        color: #64748b;
        font-size: 0.86rem;
        line-height: 1.6;
        margin-top: 0.45rem;
    }

    .reference-status {
        display: inline-block;
        color: #1e40af;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        padding: 0.16rem 0.55rem;
        border-radius: 5px;
        margin-top: 0.75rem;
    }

    @media screen and (max-width: 900px) {
        .block-container {
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }
        .contact-title {
            font-size: 2.1rem;
        }
        .contact-subtitle,
        .contact-body {
            text-align: left;
        }
    }
</style>
""")


# =============================================================================
# PAGE HEADING
# =============================================================================

render_html(f"""
<div class="contact-title">Contact</div>
<div class="contact-subtitle">
    Development team and academic contact information for <b>{escape(APP_NAME)}</b>.
    A description of the application and the methodological formulation on which it
    is based is provided on the <b>About</b> page.
</div>
<div class="contact-rule"></div>
""")


# =============================================================================
# DEVELOPMENT TEAM
# =============================================================================

render_html('<div class="section-title">Development Team</div>')

render_html(f"""
<div class="contact-body">
    {escape(APP_NAME)} was developed within the {escape(PROGRAM)} program at
    {escape(INSTITUTION)}, {escape(LOCATION)}. Any questions regarding the
    application or the accompanying research may be directed to the contacts
    listed below.
</div>
""")

team_cards = []
for member in TEAM:
    name_html = escape(member["name"])
    role_html = escape(member["role"])

    email = member.get("email")
    if email:
        safe_email = escape(email)
        email_html = (
            f'<div class="team-email">'
            f'<span class="team-email-label">Email</span>'
            f'<a href="mailto:{safe_email}">{safe_email}</a>'
            f'</div>'
        )
    else:
        email_html = ""

    team_cards.append(dedent(f"""
        <div class="team-card">
            <div class="team-name">{name_html}</div>
            <div class="team-role">{role_html}</div>
            <div class="team-affil">
                {escape(PROGRAM)}<br>{escape(INSTITUTION)}
            </div>
            {email_html}
        </div>
    """).strip())

render_html('<div class="team-grid">' + "".join(team_cards) + "</div>")


# =============================================================================
# RESEARCH REFERENCE
# =============================================================================

render_html('<div class="section-title">Research Reference</div>')

render_html(f"""
<div class="contact-body">
    This application accompanies the authors' research article, which is currently
    in the publication process at {escape(JOURNAL)}.
</div>
<div class="reference-box">
    <div class="reference-title">{escape(ARTICLE_TITLE)}</div>
    <div class="reference-authors">{escape(ARTICLE_AUTHORS)}</div>
    <div class="reference-meta">
        {escape(PROGRAM)}, {escape(INSTITUTION)}, {escape(LOCATION)}<br>
        {escape(JOURNAL)} &middot; {escape(JOURNAL_ISSN)}
    </div>
    <div class="reference-status">{escape(ARTICLE_STATUS)}</div>
</div>
""")