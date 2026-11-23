# %% [markdown]
"""# 15 MV square fields: pyRadMC Monte Carlo vs. the SVD pencil-beam engine.

A made-up 15 MV machine (Ali & Rogers 2012 Varian 15 MV spectrum, the generic machine's
primary fluence, no off-axis softening, no extra-focal source, no contaminant electrons)
was simulated in pyRadMC:

- a pencil kernel in water, fitted to the SVD model of pyRadPlan's SVDPB engine
  (``15MV_svd_fit.npz``: m, three betas and lateral kernels per SSD);
- square fields through a jawless dual-layer MLC, SSD 90 cm, 25 cm of air above the
  water (``15MV_machine_dose_<field>mm.npz``: 3D dose per unit primary fluence on the axis
  at the isocentre plane, MeV/g per photon/cm²).

Workflow:

1. One SVDPB machine for all fields: the fitted kernel, the primary fluence of the MC
   files (identical in all of them), SAD 100 cm.
2. The penumbra (Gaussian FWHM at the isocentre) is fitted once, to the MC 80/20
   penumbrae in x and y at d_max, 5, 10 and 20 cm of the 40, 100 and 150 mm fields.
3. One calibration factor c = D_MC / D_SVDPB, on the axis at 10 cm depth of the 100 mm
   field only, then held fixed: any field-size dependence stays visible.
4. Forward dose of every field on a water phantom on the MC grid (surface at SSD 90 cm),
   then depth doses, profiles at d_max, 5, 10 and 20 cm, widths, penumbrae, a 2 %/2 mm
   gamma inside the field and output factors (axis, 10 cm depth, relative to 100 mm;
   SSD 90 cm instead of the usual SSD 95 cm / 5 cm).

Only fields of at least 40 mm are compared; smaller MC files are skipped. The build-up
region (depth < d_max, where the 3-component fit itself is off by 2-4 %) is reported
separately from the region beyond d_max.

Frames: the MC has the focal spot at the origin and z along the beam, x the leaf-travel
(crossplane) and y the leaf-side (inplane) direction. In pyRadPlan, gantry 0 sends the beam
along LPS +y; the machine crossplane axis is BEV x (= LPS x) and the inplane axis BEV z
(= LPS z). Hence MC (x, y, z) <-> LPS (x, z, y + SAD). This mapping is a reflection, so
an asymmetric MC feature would appear mirrored in y; the open fields are symmetric.

Run from the repository root::

    python examples/mc_pyradmc_vs_svdpb_15MV.py [--data-dir DIR] [--penumbra FWHM_MM]
"""

# %%
from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.io import savemat

from pyRadPlan import PhotonPlan, calc_dose_forward, generate_stf
from pyRadPlan.ct import create_ct
from pyRadPlan.cst import create_cst, create_voi
from pyRadPlan.machines import add_machine_file, load_from_name, load_machine
from pyRadPlan.machines.photons import PhotonSVDKernel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("mc_vs_svdpb")
logger.setLevel(logging.INFO)

# pyRadMC checked out next to pyRadPlan
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "pyRadMC" / "examples" / "15MV_results"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "15MV_svdpb_results"
ENERGY = 15.0
MACHINE_NAME = "MadeUp15MV"
AIR_VOXELS = 4  # air in front of the water, so the ray tracer sees the surface inside the CT
REF_FIELD_MM = 100
REF_DEPTH_MM = 100.0
OF_HALF_DEPTH_MM = 10.0  # output factors: depth doses averaged over 9-11 cm
CALIB_HALF_MM = 2.5  # calibration box: voxel centres within +-2.5 mm (3 x 3 x 2 voxels)
MIN_FIELD_MM = 40  # smaller fields are not compared
AVG_HALF_MM = 2.5  # depth doses and profiles: voxel centres within +-2.5 mm of the axis
FIT_FIELDS_MM = (40, 100, 150)
FIT_PENUMBRAS_MM = (2.0, 3.0, 4.0, 5.0, 6.0)
PROFILE_DEPTHS_MM = (None, 50.0, 100.0, 200.0)  # None = the field's MC d_max
PDD_DEPTHS_MM = (50.0, 100.0, 200.0, 300.0)
PDD_COMPARE_MAX_MM = 300.0
GAMMA_DOSE = 0.02  # global, fraction of the field's MC maximum
GAMMA_DTA_MM = 2.0
GAMMA_LOW_DOSE = 0.10  # voxels below 10 % of the field's MC maximum are not evaluated

# Default categorical slots 1-4 (validated palette), one per profile depth, in fixed order
DEPTH_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MC_COLOR = "#3d3d3a"
SVD_COLOR = "#2a78d6"
MUTED = "#8a8a85"


# %% [markdown]
# ## Input
# %%
@dataclass
class MCDose:
    """MC dose of one square field, in the water only."""

    field_mm: int
    dose: np.ndarray  # (x, y, depth), MeV/g per photon/cm²
    x_mm: np.ndarray
    y_mm: np.ndarray
    depth_mm: np.ndarray
    spacing_mm: float
    ssd_mm: float
    sad_mm: float
    primary_fluence: np.ndarray


def load_mc(path: Path) -> MCDose:
    """Read one MC field file and keep its water region."""
    d = np.load(path)
    origin = 10.0 * d["origin_cm"]
    spacing = 10.0 * d["spacing_cm"]
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("The comparison assumes isotropic MC voxels.")
    field = 10.0 * d["field_cm"]
    if field[0] != field[1]:
        raise ValueError(f"{path.name}: only square fields are supported")
    shape = d["dose_per_fluence"].shape
    centres = [origin[a] + (np.arange(shape[a]) + 0.5) * spacing[a] for a in range(3)]
    surface = 10.0 * float(d["surface_z_cm"])
    in_water = centres[2] > surface
    return MCDose(
        field_mm=int(round(field[0])),
        dose=np.asarray(d["dose_per_fluence"][:, :, in_water], dtype=np.float64),
        x_mm=centres[0],
        y_mm=centres[1],
        depth_mm=centres[2][in_water] - surface,
        spacing_mm=float(spacing[0]),
        ssd_mm=10.0 * float(d["ssd_cm"]),
        sad_mm=10.0 * float(d["sad_cm"]),
        primary_fluence=np.asarray(d["primary_fluence_mm"], dtype=np.float64),
    )


def find_fields(data_dir: Path) -> dict[int, Path]:
    """The MC field files in *data_dir*, by field size in mm."""
    files = {}
    for path in data_dir.glob("15MV_machine_dose_*mm.npz"):
        files[int(path.stem.removeprefix("15MV_machine_dose_").removesuffix("mm"))] = path
    if not files:
        raise FileNotFoundError(f"No 15MV_machine_dose_*mm.npz in {data_dir}")
    return dict(sorted(files.items()))


def build_machine(fit_path: Path, primary_fluence: np.ndarray, penumbra_mm: float):
    """Replace the generic photon LINAC's 6 MV kernel by the fitted 15 MV kernel."""
    fit = np.load(fit_path)
    kernel = PhotonSVDKernel(
        energy=ENERGY,
        m=float(fit["m_per_mm"]),
        penumbra=penumbra_mm,
        kernel_betas=np.asarray(fit["betas_per_mm"], dtype=np.float64),
        kernel_ssds=np.asarray(fit["kernel_ssds_mm"], dtype=np.float64),
        kernel_pos=np.asarray(fit["kernel_pos_mm"], dtype=np.float64),
        # Already in the version-2 (1/mm^2) convention: no legacy factor 4
        kernel_data=np.asarray(fit["kernel_data"], dtype=np.float64),
        primary_fluence=primary_fluence,
    )
    generic = load_from_name("photons", "Generic")
    if generic.version < 2:
        raise RuntimeError("Expected a version-2 machine (kernels in the 1/mm^2 convention).")
    return generic.model_copy(
        update={
            "name": MACHINE_NAME,
            "sad": float(fit["sad_mm"]),
            "energies": np.array([ENERGY]),
            "pb_kernels": {ENERGY: kernel},
            "description": "Made-up 15 MV machine, SVD kernel fitted to pyRadMC",
        }
    )


def save_machine_mat(machine, path: Path, note: str = "") -> None:
    """Write a photon SVD machine as a matRad-style ``machine`` struct (``.mat``).

    ``version = 2`` must be stored: the loader treats a machine without it as version 1
    and multiplies the kernels by 4.
    """
    kernel = machine.pb_kernels[machine.energies[0]]
    n_ssd, n_comp, _ = kernel.kernel_data.shape
    # matRad layout: data.kernel is a struct array over SSDs with fields SSD, kernel1..n
    kernel_struct = np.empty(
        (1, n_ssd), dtype=[("SSD", "O")] + [(f"kernel{i + 1}", "O") for i in range(n_comp)]
    )
    for s in range(n_ssd):
        kernel_struct[0, s]["SSD"] = float(kernel.kernel_ssds[s])
        for i in range(n_comp):
            kernel_struct[0, s][f"kernel{i + 1}"] = kernel.kernel_data[s, i].reshape(1, -1)
    meta = {
        "radiationMode": machine.radiation_mode,
        "dataType": "-",
        "name": machine.name,
        "version": 2,
        "SAD": float(machine.sad),
        "SCD": float(machine.scd),
        "created_on": time.strftime("%d-%b-%Y"),
        "created_by": "mc_pyradmc_vs_svdpb_15MV.py",
        "description": f"{machine.description}. Kernels in 1/mm^2 (version 2). {note}".strip(),
    }
    data = {
        "energy": float(kernel.energy),
        "m": float(kernel.m),
        "betas": kernel.kernel_betas.reshape(1, -1),
        "penumbraFWHMatIso": float(kernel.penumbra),
        "primaryFluence": kernel.primary_fluence,
        "kernelPos": kernel.kernel_pos.reshape(1, -1),
        "kernel": kernel_struct,
    }
    savemat(path, {"machine": {"meta": meta, "data": data}}, do_compression=True)

    reloaded = load_machine(str(path))
    k2 = reloaded.pb_kernels[float(kernel.energy)]
    if not (
        reloaded.version == 2
        and np.array_equal(k2.kernel_data, kernel.kernel_data)
        and np.array_equal(k2.kernel_ssds, kernel.kernel_ssds)
        and np.array_equal(k2.primary_fluence, kernel.primary_fluence)
        and k2.penumbra == kernel.penumbra
    ):
        raise RuntimeError(f"Saved machine {path} does not read back unchanged.")


# %% [markdown]
# ## Water phantom on the MC grid and SVDPB forward dose
# LPS x = MC x, LPS z = MC y, LPS y = depth axis. The isocentre is at the LPS origin.
# %%
def build_phantom(mc: MCDose):
    """Water CT and structures on the MC grid of one field."""
    s = mc.spacing_mm
    nx, nz = mc.x_mm.size, mc.y_mm.size
    ny = AIR_VOXELS + mc.depth_mm.size
    y_first_water = mc.ssd_mm - mc.sad_mm + mc.depth_mm[0]

    hu = np.zeros((nz, ny, nx), dtype=np.int16)  # sitk array order (z, y, x)
    hu[:, :AIR_VOXELS, :] = -1000
    cube = sitk.GetImageFromArray(hu)
    cube.SetSpacing((s, s, s))
    cube.SetOrigin((mc.x_mm[0], y_first_water - AIR_VOXELS * s, mc.y_mm[0]))
    ct = create_ct(cube_hu=cube)

    zz, yy, xx = np.meshgrid(
        mc.y_mm, cube.GetOrigin()[1] + s * np.arange(ny), mc.x_mm, indexing="ij"
    )
    body = hu == 0
    target = body & (np.abs(xx) <= 5.0) & (np.abs(yy) <= 5.0) & (np.abs(zz) <= 5.0)

    def voi(name, voi_type, mask):
        img = sitk.GetImageFromArray(mask.astype(np.uint8))
        img.CopyInformation(cube)
        return create_voi(name=name, voi_type=voi_type, mask=img, ct_image=ct)

    cst = create_cst([voi("WATER", "OAR", body), voi("ISO", "TARGET", target)], ct=ct)
    return ct, cst


def run_svdpb(machine, penumbra_mm: float, mc: MCDose, phantom) -> np.ndarray:
    """Uncalibrated SVDPB dose of one open field, as (x, y, depth) on the MC water grid."""
    ct, cst = phantom
    pln = PhotonPlan(radiation_mode="photons")
    pln.machine = machine
    pln.prop_stf = {
        "generator": "photonOpenFields",
        "gantry_angles": [0.0],
        "couch_angles": [0.0],
        "bixel_width": float(mc.field_mm),  # the field width of the single open-field ray
        "energy": ENERGY,
        "iso_center": np.zeros((1, 3)),
    }
    pln.prop_dose_calc = {
        "engine": "SVDPB",
        "dose_grid": {"resolution": [mc.spacing_mm] * 3},
        "use_custom_primary_photon_fluence": True,  # field x primary fluence, then penumbra
        "force_penumbra": penumbra_mm,
        "enable_dij_sampling": False,
        # + half diagonal = radius at iso; stays inside the kernel's 299.5 mm range
        "geometric_lateral_cutoff": 250.0,
    }
    stf = generate_stf(ct, cst, pln)

    t0 = time.perf_counter()
    result = calc_dose_forward(ct, cst, stf, pln, weights=np.ones(1))
    logger.info(
        "SVDPB %3d mm, FWHM %.2f mm: %.1f s", mc.field_mm, penumbra_mm, time.perf_counter() - t0
    )

    dose = result["physical_dose"]
    dose = sitk.GetArrayFromImage(dose) if isinstance(dose, sitk.Image) else np.asarray(dose)
    if dose.shape != (mc.y_mm.size, AIR_VOXELS + mc.depth_mm.size, mc.x_mm.size):
        raise RuntimeError(f"Unexpected dose grid {dose.shape}")
    # (LPS z, LPS y, LPS x) = (MC y, depth, MC x) -> (MC x, MC y, depth)
    return np.transpose(dose, (2, 0, 1))[:, :, AIR_VOXELS:].astype(np.float64)


# %% [markdown]
# ## Analysis helpers
# %%
def window(coords: np.ndarray, centre: float, half: float) -> np.ndarray:
    """Voxel centres within *half* of *centre* (at least the nearest one)."""
    sel = np.abs(coords - centre) <= half + 1e-6
    if not sel.any():
        sel[np.argmin(np.abs(coords - centre))] = True
    return sel


def depth_window(mc: MCDose, depth_mm: float) -> np.ndarray:
    """The two voxels around a depth (depths of interest lie on voxel boundaries)."""
    return window(mc.depth_mm, depth_mm, 0.5 * mc.spacing_mm)


def pdd_column(d: np.ndarray, mc: MCDose) -> np.ndarray:
    ix, iy = window(mc.x_mm, 0.0, AVG_HALF_MM), window(mc.y_mm, 0.0, AVG_HALF_MM)
    return d[np.ix_(ix, iy)].mean(axis=(0, 1))


def profile(d: np.ndarray, mc: MCDose, depth_mm: float, axis: str) -> np.ndarray:
    iz = depth_window(mc, depth_mm)
    if axis == "x":
        return d[:, window(mc.y_mm, 0.0, AVG_HALF_MM)][:, :, iz].mean(axis=(1, 2))
    return d[window(mc.x_mm, 0.0, AVG_HALF_MM)][:, :, iz].mean(axis=(0, 2))


def d_max(depth: np.ndarray, pdd: np.ndarray) -> tuple[float, float]:
    """Depth and value of the maximum: parabola through +-1.5 cm around the peak."""
    i = int(np.argmax(ndimage.uniform_filter1d(pdd, 3)))
    sel = window(depth, depth[i], 15.0)
    a, b, c = np.polyfit(depth[sel], pdd[sel], 2)
    z = float(np.clip(-b / (2 * a), depth[sel][0], depth[sel][-1])) if a < 0 else depth[i]
    return z, float(np.polyval((a, b, c), z))


def crossing(pos: np.ndarray, prof: np.ndarray, level: float) -> float:
    """Position where a profile (going outward from the axis) first falls below *level*."""
    below = np.nonzero(prof < level)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    j = below[0]
    return float(np.interp(level, [prof[j], prof[j - 1]], [pos[j], pos[j - 1]]))


def field_metrics(pos: np.ndarray, prof: np.ndarray) -> dict[str, float]:
    """Width at 50 % and mean 80/20 penumbra, relative to the axis value."""
    centre = int(np.argmin(np.abs(pos)))
    rel = prof / prof[window(pos, 0.0, AVG_HALF_MM)].mean()
    sides = []
    for sl in (slice(centre, None), slice(centre, None, -1)):
        p, r = np.abs(pos[sl]), rel[sl]
        sides.append({lvl: crossing(p, r, lvl) for lvl in (0.8, 0.5, 0.2)})
    return {
        "width": sides[0][0.5] + sides[1][0.5],
        "penumbra": 0.5 * sum(s[0.2] - s[0.8] for s in sides),
    }


def box_mean(d: np.ndarray, mc: MCDose, depth_mm: float, half_mm: float) -> float:
    ix, iy = window(mc.x_mm, 0.0, half_mm), window(mc.y_mm, 0.0, half_mm)
    return float(d[np.ix_(ix, iy, depth_window(mc, depth_mm))].mean())


def profile_depths(zmax_mm: float) -> list[tuple[str, float]]:
    return [
        ("d_max" if depth is None else f"{depth / 10:g}cm", zmax_mm if depth is None else depth)
        for depth in PROFILE_DEPTHS_MM
    ]


def penumbrae(d: np.ndarray, mc: MCDose, zmax_mm: float) -> dict[tuple[str, str], float]:
    """80/20 penumbrae in x and y at the profile depths."""
    out = {}
    for label, depth in profile_depths(zmax_mm):
        for axis, pos in (("x", mc.x_mm), ("y", mc.y_mm)):
            prof = profile(d, mc, depth, axis)
            out[(label, axis)] = field_metrics(pos, prof)["penumbra"]
    return out


def gamma_pass(ref: np.ndarray, ev: np.ndarray, roi: np.ndarray, spacing: float) -> np.ndarray:
    """Global 3D gamma of *ev* against *ref* on the voxels in *roi*.

    Brute force: the evaluated dose is shifted by every offset on a half-voxel grid within
    the search radius (linear interpolation) and the minimum gamma is kept per voxel.
    """
    dd = GAMMA_DOSE * ref.max()
    step = 0.5 * spacing
    n = int(np.ceil(2 * GAMMA_DTA_MM / step))
    idx = np.nonzero(roi)
    lo = [max(int(i.min()) - n, 0) for i in idx]
    hi = [min(int(i.max()) + n + 1, s) for i, s in zip(idx, ref.shape)]
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    ref_c, ev_c, roi_c = ref[crop], ev[crop], roi[crop]

    gamma2 = np.full(ref_c.shape, np.inf)
    offsets = np.arange(-n, n + 1) * step
    for ox in offsets:
        for oy in offsets:
            for oz in offsets:
                r2 = (ox**2 + oy**2 + oz**2) / GAMMA_DTA_MM**2
                if r2 > 4.0:
                    continue
                shifted = ndimage.shift(
                    ev_c, (-ox / spacing, -oy / spacing, -oz / spacing), order=1, mode="nearest"
                )
                np.minimum(gamma2, r2 + ((shifted - ref_c) / dd) ** 2, out=gamma2)
    return np.sqrt(gamma2[roi_c])


# %% [markdown]
# ## Penumbra fit
# %%
def fit_penumbra(machine, mcs, phantoms, out_dir: Path) -> float:
    """FWHM minimizing the squared 80/20 penumbra differences over fields, depths, x and y.

    SVDPB is evaluated at a few FWHMs; each penumbra is interpolated linearly in between.
    """
    targets, scanned = {}, {}
    for f in FIT_FIELDS_MM:
        mc = mcs[f]
        zmax = d_max(mc.depth_mm, pdd_column(mc.dose, mc))[0]
        targets[f] = penumbrae(mc.dose, mc, zmax)
        scanned[f] = []
        for fwhm in FIT_PENUMBRAS_MM:
            d = run_svdpb(machine, fwhm, mc, phantoms[f])
            scanned[f].append(penumbrae(d, mc, zmax))

    grid = np.linspace(FIT_PENUMBRAS_MM[0], FIT_PENUMBRAS_MM[-1], 801)
    cost = np.zeros_like(grid)
    for f in FIT_FIELDS_MM:
        for key, target in targets[f].items():
            values = [s[key] for s in scanned[f]]
            cost += (np.interp(grid, FIT_PENUMBRAS_MM, values) - target) ** 2
    n_terms = sum(len(t) for t in targets.values())
    best = float(grid[np.argmin(cost)])
    if best in (grid[0], grid[-1]):
        logger.warning("Penumbra fit at the edge of the scanned range (%.2f mm)", best)
    logger.info(
        "Fitted FWHM %.2f mm, rms 80/20 penumbra difference %.2f mm over %d profiles",
        best,
        np.sqrt(cost.min() / n_terms),
        n_terms,
    )

    fig, axes = plt.subplots(1, len(FIT_FIELDS_MM), figsize=(4.2 * len(FIT_FIELDS_MM), 4))
    for ax, f in zip(axes, FIT_FIELDS_MM):
        labels = list(dict.fromkeys(label for label, _ in targets[f]))
        for color, label in zip(DEPTH_COLORS, labels):
            svd = [0.5 * (s[(label, "x")] + s[(label, "y")]) for s in scanned[f]]
            ax.plot(FIT_PENUMBRAS_MM, svd, "-", color=color, lw=2, label=label)
            for axis, marker in (("x", "o"), ("y", "s")):
                ax.plot(best, targets[f][(label, axis)], marker, color=color, ms=8, mfc="none")
        ax.axvline(best, color=MUTED, lw=1, ls=":")
        ax.set_title(f"{f} mm field")
        ax.set_xlabel("SVDPB FWHM at iso [mm]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("80/20 penumbra [mm]")
    axes[0].legend(title="SVDPB (line); MC x o, y □", fontsize=8, title_fontsize=8)
    fig.suptitle(f"Penumbra fit: FWHM = {best:.2f} mm")
    fig.tight_layout()
    fig.savefig(out_dir / "penumbra_fit.png", dpi=150)
    plt.close(fig)
    return best


# %% [markdown]
# ## Per-field comparison
# %%
def compare_field(mc: MCDose, svd: np.ndarray, do_gamma: bool, out_dir: Path) -> dict:
    """Metrics of one field (SVDPB already calibrated) and its profile figure."""
    z = mc.depth_mm
    pdd_mc, pdd_svd = pdd_column(mc.dose, mc), pdd_column(svd, mc)
    zmax_mc, dmax_mc = d_max(z, pdd_mc)
    zmax_svd, dmax_svd = d_max(z, pdd_svd)

    rows = {"d_max_cm": (zmax_mc / 10, zmax_svd / 10)}
    for depth in PDD_DEPTHS_MM:
        rows[f"PDD{depth / 10:g}_%"] = (
            100 * float(np.interp(depth, z, pdd_mc)) / dmax_mc,
            100 * float(np.interp(depth, z, pdd_svd)) / dmax_svd,
        )

    # Absolute (calibrated) axis differences, % of the field's MC maximum: these include
    # any output-factor error. Build-up and the region beyond d_max are judged separately.
    diff = 100 * (pdd_svd - pdd_mc) / dmax_mc
    buildup = z < zmax_mc
    beyond = (z >= zmax_mc) & (z <= PDD_COMPARE_MAX_MM)
    stats = {
        "pdd_diff_buildup_rms_%": np.sqrt(np.mean(diff[buildup] ** 2)),
        "pdd_diff_buildup_maxabs_%": np.abs(diff[buildup]).max(),
        "pdd_diff_beyond_mean_%": diff[beyond].mean(),
        "pdd_diff_beyond_rms_%": np.sqrt(np.mean(diff[beyond] ** 2)),
        "pdd_diff_beyond_maxabs_%": np.abs(diff[beyond]).max(),
    }

    norm = mc.dose.max()
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7), sharex=True, height_ratios=(3, 1), squeeze=False
    )
    half_view = 0.5 * mc.field_mm * 1.2 + 40.0
    for col, (axis, pos) in enumerate((("x", mc.x_mm), ("y", mc.y_mm))):
        for color, (label, depth) in zip(DEPTH_COLORS, profile_depths(zmax_mc)):
            p_mc, p_svd = profile(mc.dose, mc, depth, axis), profile(svd, mc, depth, axis)
            m_mc = field_metrics(pos, p_mc)
            m_svd = field_metrics(pos, p_svd)
            rows[f"width_{axis}_{label}_mm"] = (m_mc["width"], m_svd["width"])
            rows[f"pen8020_{axis}_{label}_mm"] = (m_mc["penumbra"], m_svd["penumbra"])
            core = np.abs(pos) <= 0.4 * mc.field_mm * (mc.ssd_mm + depth) / mc.sad_mm
            if core.any():
                stats[f"profile_{axis}_{label}_core_rms_%"] = np.sqrt(
                    np.mean((100 * (p_svd - p_mc)[core] / norm) ** 2)
                )
            name = f"{label} ({depth / 10:.1f} cm)" if label == "d_max" else label
            axes[0, col].plot(pos / 10, 100 * p_mc / norm, "o", color=color, ms=2.5)
            axes[0, col].plot(pos / 10, 100 * p_svd / norm, "-", color=color, lw=2, label=name)
            axes[1, col].plot(pos / 10, 100 * (p_svd - p_mc) / norm, "-", color=color, lw=1.5)
        axes[0, col].set_title(f"{axis} profiles")
        axes[0, col].set_ylabel("dose [% of MC max]")
        axes[1, col].axhline(0, color=MUTED, lw=0.8)
        axes[1, col].set_ylim(-8, 8)
        axes[1, col].set_xlim(-half_view / 10, half_view / 10)
        axes[1, col].set_xlabel(f"{axis} at depth [cm]")
        axes[1, col].set_ylabel("SVDPB − MC [% max]")
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
    axes[0, 0].legend(title="depth (MC dots, SVDPB lines)", fontsize=8, title_fontsize=8)
    fig.suptitle(f"{mc.field_mm} × {mc.field_mm} mm², SSD 90 cm")
    fig.tight_layout()
    fig.savefig(out_dir / f"profiles_{mc.field_mm}mm.png", dpi=130)
    plt.close(fig)

    if do_gamma:
        xx, yy, zz = np.meshgrid(mc.x_mm, mc.y_mm, z, indexing="ij")
        half_field = np.maximum(
            0.8 * 0.5 * mc.field_mm * (mc.ssd_mm + zz) / mc.sad_mm, 0.5 * mc.spacing_mm
        )
        roi = (np.abs(xx) <= half_field) & (np.abs(yy) <= half_field)
        roi &= mc.dose >= GAMMA_LOW_DOSE * norm
        g = gamma_pass(mc.dose, svd, roi, mc.spacing_mm)
        stats["gamma_pass_all_%"] = 100 * np.mean(g <= 1)
        stats["gamma_pass_beyond_dmax_%"] = 100 * np.mean(g[zz[roi] >= zmax_mc] <= 1)
        stats["gamma_voxels"] = g.size

    return {
        "rows": rows,
        "stats": stats,
        "pdd": (z, pdd_mc, pdd_svd, dmax_mc, zmax_mc),
        "axis_dose": (axis_dose(pdd_mc, mc), axis_dose(pdd_svd, mc)),
    }


def axis_dose(pdd: np.ndarray, mc: MCDose) -> float:
    """Output-factor dose: the depth-dose column averaged over +-1 cm around 10 cm.

    A single voxel is dominated by MC noise; the depth dose is nearly linear there, so the
    average stands for the dose at 10 cm (and is taken the same way for both engines).
    """
    return float(pdd[window(mc.depth_mm, REF_DEPTH_MM, OF_HALF_DEPTH_MM)].mean())


def plot_pdds(results: dict[int, dict], c: float, path: Path) -> None:
    """Small multiples: one depth dose (with its difference below) per field."""
    fields = list(results)
    ncol = 3
    nrow = int(np.ceil(len(fields) / ncol))
    fig = plt.figure(figsize=(4.4 * ncol, 4.2 * nrow))
    # a spacer row after each field row keeps titles clear of the tick labels above
    gs = fig.add_gridspec(3 * nrow, ncol, height_ratios=[3, 1, 0.55] * nrow, hspace=0.08)
    for i, f in enumerate(fields):
        z, pdd_mc, pdd_svd, dmax_mc, zmax = results[f]["pdd"]
        r, col = divmod(i, ncol)
        ax = fig.add_subplot(gs[3 * r, col])
        axd = fig.add_subplot(gs[3 * r + 1, col], sharex=ax)
        for a in (ax, axd):
            a.axvspan(0, zmax / 10, color="#ecebe7", lw=0)
            a.grid(alpha=0.25)
        ax.plot(z / 10, 100 * pdd_mc / dmax_mc, "o", color=MC_COLOR, ms=2, label="MC")
        ax.plot(z / 10, 100 * pdd_svd / dmax_mc, "-", color=SVD_COLOR, lw=2, label="SVDPB")
        ax.set_title(f"{f} mm", fontsize=10, pad=3)
        ax.set_ylim(0, 115)
        ax.tick_params(labelbottom=False)
        axd.plot(z / 10, 100 * (pdd_svd - pdd_mc) / dmax_mc, "-", color=MC_COLOR, lw=1.2)
        axd.axhline(0, color=MUTED, lw=0.8)
        axd.set_ylim(-6, 6)
        if col == 0:
            ax.set_ylabel("% of MC d_max")
            axd.set_ylabel("Δ [%]")
        if r == nrow - 1:
            axd.set_xlabel("depth [cm]")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Axis depth doses, SSD 90 cm (calibrated once on 100 mm, c = {c:.5g}); "
        "shaded: build-up; Δ = SVDPB − MC",
        y=0.995,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_output_factors(fields, of_mc, of_svd, path: Path) -> None:
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, height_ratios=(3, 1.3))
    ax.plot(fields, of_mc, "o", color=MC_COLOR, ms=8, label="MC")
    ax.plot(fields, of_svd, "-", color=SVD_COLOR, lw=2, marker="s", ms=5, label="SVDPB")
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_ylabel(f"output factor (axis, {REF_DEPTH_MM / 10:g} cm)")
    ax.set_title(f"Output factors relative to {REF_FIELD_MM} mm, SSD 90 cm")
    ax.legend()
    axr.plot(fields, 100 * (np.asarray(of_svd) / np.asarray(of_mc) - 1), "-o", color=MC_COLOR)
    axr.axhline(0, color=MUTED, lw=0.8)
    axr.set_ylabel("SVDPB / MC − 1 [%]")
    axr.set_xlabel("field size at iso [mm]")
    for a in (ax, axr):
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# %% [markdown]
# ## Main
# %%
def main() -> None:  # noqa: PLR0915
    """Fit, calibrate and compare all fields."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fields", type=int, nargs="*", help="field sizes [mm], default all")
    parser.add_argument("--penumbra", type=float, default=None, help="SVDPB FWHM at iso [mm]")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-gamma", action="store_true")
    parser.add_argument(
        "--register-machine",
        action="store_true",
        help="copy the saved machine into pyRadPlan's user machine folder (load by name)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = find_fields(args.data_dir)
    fields = args.fields or list(files)
    skipped = [f for f in fields if f < MIN_FIELD_MM]
    if skipped:
        logger.info("Skipping fields below %d mm: %s", MIN_FIELD_MM, skipped)
    fields = [f for f in fields if f >= MIN_FIELD_MM]
    needed = sorted(
        set(fields) | {REF_FIELD_MM} | (set() if args.penumbra else set(FIT_FIELDS_MM))
    )
    missing = [f for f in needed if f not in files]
    if missing:
        raise FileNotFoundError(f"No MC file for field(s) {missing} mm in {args.data_dir}")
    mcs = {f: load_mc(files[f]) for f in needed}

    primary_fluence = mcs[REF_FIELD_MM].primary_fluence
    for f, mc in mcs.items():
        if not np.array_equal(mc.primary_fluence, primary_fluence):
            raise ValueError(
                f"The {f} mm file has a different primary fluence; one machine needs one table."
            )
    phantoms = {f: build_phantom(mc) for f, mc in mcs.items()}
    fit_path = args.data_dir / "15MV_svd_fit.npz"
    machine = build_machine(fit_path, primary_fluence, 5.0)

    penumbra = args.penumbra or fit_penumbra(machine, mcs, phantoms, args.out_dir)
    machine = build_machine(fit_path, primary_fluence, penumbra)

    # One calibration, from the reference field only
    ref = mcs[REF_FIELD_MM]
    svd_ref = run_svdpb(machine, penumbra, ref, phantoms[REF_FIELD_MM])
    c = box_mean(ref.dose, ref, REF_DEPTH_MM, CALIB_HALF_MM) / box_mean(
        svd_ref, ref, REF_DEPTH_MM, CALIB_HALF_MM
    )
    print(f"\nPenumbra FWHM at iso: {penumbra:.2f} mm")
    print(
        f"Calibration c = D_MC / D_SVDPB = {c:.6g} ({REF_FIELD_MM} mm field, axis, "
        f"{REF_DEPTH_MM / 10:g} cm, 3x3x2 voxels); 0.01 = kernel scale exact "
        "(MC per photon/cm^2, kernel per photon/mm^2)."
    )

    results = {}
    for f in fields:
        mc = mcs[f]
        svd = svd_ref if f == REF_FIELD_MM else run_svdpb(machine, penumbra, mc, phantoms[f])
        results[f] = compare_field(mc, c * svd, not args.no_gamma, args.out_dir)
        if f != REF_FIELD_MM:
            mc.dose = None  # free the grid; the reference field is kept for the factors

    # Output factors: axis, 10 cm depth (see axis_dose), relative to the reference field
    ref_mc, ref_svd = (
        results[REF_FIELD_MM]["axis_dose"]
        if REF_FIELD_MM in results
        else (axis_dose(pdd_column(ref.dose, ref), ref), axis_dose(pdd_column(svd_ref, ref), ref))
    )
    of_mc = [results[f]["axis_dose"][0] / ref_mc for f in fields]
    of_svd = [results[f]["axis_dose"][1] / ref_svd for f in fields]

    # Report
    with open(args.out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field_mm", "quantity", "MC", "SVDPB", "SVDPB-MC"])
        w.writerow(["all", "penumbra_fwhm_iso_mm", "", f"{penumbra:.3f}", ""])
        w.writerow(["all", "calibration_c", "", f"{c:.6g}", ""])
        for f, om, osv in zip(fields, of_mc, of_svd):
            for key, (a, b) in results[f]["rows"].items():
                w.writerow([f, key, f"{a:.4f}", f"{b:.4f}", f"{b - a:.4f}"])
            w.writerow([f, "output_factor", f"{om:.5f}", f"{osv:.5f}", f"{osv - om:.5f}"])
            for key, v in results[f]["stats"].items():
                w.writerow([f, key, "", f"{v:.4f}", ""])
    with open(args.out_dir / "output_factors.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field_mm", "OF_MC", "OF_SVDPB", "SVDPB/MC-1_%"])
        for f, om, osv in zip(fields, of_mc, of_svd):
            w.writerow([f, f"{om:.5f}", f"{osv:.5f}", f"{100 * (osv / om - 1):.3f}"])

    head = (
        f"{'field':>6} {'d_max MC/SV':>12} {'PDD10 MC/SV':>12} {'width10 x/y MC|SV':>22} "
        f"{'pen10 x/y MC|SV':>18} {'PDD diff build-up':>15} {'PDD diff >d_max':>13} "
        f"{'gamma':>6} {'OF MC/SV':>14}"
    )
    print("\n" + head + "\n" + "-" * len(head))
    for f, om, osv in zip(fields, of_mc, of_svd):
        r, s = results[f]["rows"], results[f]["stats"]
        print(
            f"{f:>4}mm"
            f" {r['d_max_cm'][0]:>5.2f}/{r['d_max_cm'][1]:<5.2f}"
            f"  {r['PDD10_%'][0]:>5.1f}/{r['PDD10_%'][1]:<5.1f}"
            f" {r['width_x_10cm_mm'][0]:>5.1f}/{r['width_y_10cm_mm'][0]:<5.1f}|"
            f"{r['width_x_10cm_mm'][1]:<5.1f}"
            f"    {r['pen8020_x_10cm_mm'][0]:>4.1f}/{r['pen8020_y_10cm_mm'][0]:<4.1f}|"
            f"{r['pen8020_x_10cm_mm'][1]:<4.1f}"
            f"  rms {s['pdd_diff_buildup_rms_%']:>4.1f} %"
            f"   rms {s['pdd_diff_beyond_rms_%']:>4.1f} %"
            f" {s.get('gamma_pass_all_%', np.nan):>5.1f}%"
            f"  {om:>6.4f}/{osv:<6.4f}"
        )
    print(
        "PDD diff: calibrated SVDPB - MC on the axis, % of the field's MC d_max dose "
        f"(beyond d_max up to {PDD_COMPARE_MAX_MM / 10:g} cm); gamma 2 %/2 mm in 80 % of "
        "the field."
    )

    plot_pdds(results, c, args.out_dir / "pdd_all_fields.png")
    plot_output_factors(fields, of_mc, of_svd, args.out_dir / "output_factors.png")

    machine_file = args.out_dir / f"photons_{MACHINE_NAME}.mat"
    save_machine_mat(
        machine,
        machine_file,
        note=f"Penumbra FWHM fitted to MC fields {list(FIT_FIELDS_MM)} mm: {penumbra:.3f} mm. "
        f"MC calibration ({REF_FIELD_MM} mm, axis, 10 cm, SSD 90 cm): "
        f"D_MC per photon/cm^2 = {c:.6g} x D_SVDPB.",
    )
    print(f"\nMachine saved to {machine_file.resolve()} (reads back unchanged)")
    if args.register_machine:
        registered = add_machine_file(machine_file, overwrite=True)
        print(f"Registered as photons/{MACHINE_NAME}: {registered}")
    print(f"Outputs in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
