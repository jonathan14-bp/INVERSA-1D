# INVERSA-1D

A web-based application for one-dimensional inversion of geoelectrical
resistivity data (Vertical Electrical Sounding, Schlumberger configuration)
using two inversion methods: **Damped Singular Value Decomposition (SVD)**
and **Levenberg-Marquardt (LM)**.

Built with Python and Streamlit, and containerised with Docker so that it runs
consistently across Windows, macOS, and Linux.

---

## Background

The inversion of 1D VES data is a non-linear and frequently ill-posed problem,
which makes regularisation necessary. This application implements two distinct
regularisation strategies and allows them to be compared directly on the same
dataset under identical conditions.

## Methodology

### Forward Modelling
Apparent resistivity is computed through a recursive resistivity transform
propagated from the deepest layer upward to the surface, followed by
convolution with the **11-point Ghosh digital filter** for the Schlumberger
configuration.

### Jacobian Matrix
The sensitivity matrix is evaluated numerically using a finite-difference
approximation with a parameter perturbation of 0.1 percent.

### Damped SVD Inversion
Singular value decomposition is applied to the Jacobian matrix, and a damping
factor is introduced to stabilise the solution against small singular values
that would otherwise amplify noise in the observed data.

### Levenberg-Marquardt Inversion
The algorithm adaptively interpolates between the Gauss-Newton method and
steepest descent. At each iteration the optimal damping factor is determined
by a **Golden Section Search** performed in logarithmic space.

### Error Evaluation
Misfit is quantified as the root-mean-square error in logarithmic space, which
is standard practice for resistivity data spanning several orders of magnitude.

---

## Features

- Three built-in synthetic models (homogeneous 10, 100, and 1000 Ohm.m)
- Synthetic model editor and manual data entry
- Field data import from Excel files (.xlsx and .xls)
- Synthetic noise injection with a user-defined percentage
- Automatic or manually specified initial models
- Visualisation of sounding curves, layered models, and RMS convergence history
- Stopping criteria based on a target RMS or a maximum iteration count

---

## Running with Docker (recommended)

Prerequisite: Docker Desktop installed and running.

```bash
git clone https://github.com/jonathan14-bp/INVERSA-1D.git
cd INVERSA-1D
docker compose up --build -d
```

Open `http://localhost:8501` in a browser.

To stop the application:

```bash
docker compose down
```

---

## Running without Docker

Prerequisite: Python 3.12.

```bash
pip install -r requirements.txt
streamlit run Home.py
```

---

## Input Data Format

An Excel file containing two columns:

| Column | Description |
|--------|-------------|
| AB/2   | Half the current electrode spacing, in metres |
| Rho    | Measured apparent resistivity, in Ohm.m |

Column headers are detected automatically as long as the first contains the
string "AB" and the second contains either "rho" or "res".

---

## Repository Structure
INVERSA-1D/
├── Home.py Main interface and application workflow
├── svd_core.py Forward modelling, Jacobian, and Damped SVD inversion
├── lm_core.py Forward modelling, Jacobian, and Levenberg-Marquardt inversion
├── pages/
│ ├── 1_About.py Method description and theoretical background
│ └── 2_Contact.py Contact information
├── .streamlit/
│ └── config.toml Streamlit configuration
├── Dockerfile Image build recipe
├── docker-compose.yml Container runtime configuration
└── requirements.txt Dependency list with pinned versions

---

## Reproducibility

All dependencies are pinned to exact versions in `requirements.txt`.

Within the container, the linear algebra backend is restricted to a single
thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).
This prevents the summation order of floating-point reductions from varying
with the number of available CPU cores, so that inversion results remain
identical across different hardware. This is essential for the reproducibility
of the numerical values reported in the accompanying publication.

---

## License

Released under the MIT License. See the `LICENSE` file for details.

## Author

Jonathan Bilian Putra
