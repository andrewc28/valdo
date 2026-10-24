"""Map files and structure helpers for the valdo-easy report.

Every map here is computed with gemmi from an amplitude column and a phase
column of an MTZ file, put on a grid of about d_min/3 spacing (``sample_rate``
3, as upstream's blob search), normalised to mean 0 and RMS 1 over the whole
unit cell (so map values are in "sigma" units), and written in CCP4 format,
optionally cut to a box around a point. A cut map keeps the full-cell
statistics in its header (mean 0, RMS 1), so "3.0" means 3 sigma of the whole
map in every viewer that uses absolute contour levels.

The difference map is built from exactly the coefficients and phases that
upstream's blob search used (``valdo.blobs.blob_helper``: read the phased MTZ
with reciprocalspaceship, keep ``[diff_col, phase_col]``, drop NaN rows,
``transform_f_phi_to_map``, ``normalize``), but it is kept signed and is not
blurred. The blob search itself works on |map| after a Gaussian blur, so a
blob can come from negative difference density (something that left), not
only from positive density (something that arrived).

This module also mirrors two upstream file-lookup rules so that the report
reads the same files the pipeline used:

* :func:`locate_phase_file` = ``valdo.helper.find_phase_file`` with
  ``parser=None`` (exact basename, then ``*{stem}*.mtz`` preferring
  ``*_001.mtz``, then ``basename[0:4] + '.mtz'``);
* :func:`locate_model` = the model lookup in ``valdo.blobs.blob_helper`` and
  ``valdo.tag`` (``{stem}.pdb``, then ``*{stem}*.pdb``).

Nothing here imports torch.
"""

import glob
import itertools
import os

import numpy as np

# Amplitude/phase pairs of 2mFo-DFc-style ("context") map coefficients, in order
# of preference: PHENIX, XChem/PTP1B files, refmac/dimple.
TWO_FOFC_PAIRS = (
    ("2FOFCWT", "PH2FOFCWT"),
    ("2FOFCWT", "PHI2FOFCWT"),
    ("FWT", "PHWT"),
    ("2FOFCWT_fill", "PH2FOFCWT_fill"),
    ("2FOFCWT_no_fill", "PH2FOFCWT_no_fill"),
    ("FWT", "PHFWT"),
)

# Amplitude/phase pairs of mFo-DFc difference map coefficients.
FOFC_PAIRS = (
    ("FOFCWT", "PHFOFCWT"),
    ("FOFCWT", "PHIFOFCWT"),
    ("DELFWT", "PHDELWT"),
    ("DELFWT", "PHDELFWT"),
)

DEFAULT_SAMPLE_RATE = 3.0  # upstream valdo.blobs.blob_helper default (the pipeline does not change it)

# Residue names never reported as "nearest residue" by default.
WATER_NAMES = frozenset({"HOH", "WAT", "H2O", "DOD", "D2O", "SOL", "TIP", "TIP3"})


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def mtz_column_types(path_or_mtz):
    """Return {label: MTZ column type} for an MTZ path or a gemmi.Mtz."""
    import gemmi

    mtz = gemmi.read_mtz_file(path_or_mtz) if isinstance(path_or_mtz, str) else path_or_mtz
    return {c.label: c.type for c in mtz.columns}


def find_coefficient_pairs(columns, kind="2fofc", preferred_phase=None):
    """Return the map-coefficient (amplitude, phase) pairs present in ``columns``.

    Parameters
    ----------
    columns : dict or iterable of str
        Column labels (a dict label -> type is also accepted; when types are
        given the amplitude must be type F and the phase type P).
    kind : {"2fofc", "fofc"}
        Which kind of coefficients to look for.
    preferred_phase : str, optional
        If a pair uses this phase column it is listed first.

    Returns
    -------
    list of (str, str), possibly empty, best first.
    """
    pairs = {"2fofc": TWO_FOFC_PAIRS, "fofc": FOFC_PAIRS}[kind]
    types = dict(columns) if isinstance(columns, dict) else {c: None for c in columns}
    found = []
    for f, phi in pairs:
        if f in types and phi in types:
            if types[f] not in (None, "F") or types[phi] not in (None, "P"):
                continue
            found.append((f, phi))
    if preferred_phase:
        found.sort(key=lambda p: p[1] != preferred_phase)
    return found


# ---------------------------------------------------------------------------
# File lookup rules mirrored from upstream
# ---------------------------------------------------------------------------

def locate_phase_file(mtz_path, phasing_path):
    """Find the phase MTZ upstream's add_phases used for ``mtz_path``.

    Mirrors ``valdo.helper.find_phase_file(file, apo_mtzs_path, parser=None)``
    (helper.py:253-305) without importing valdo.helper (which imports torch).

    Returns (path or None, handler) with upstream's handler numbers:
    2 exact basename, 3 ``*{stem}*.mtz`` (``*_001.mtz`` preferred),
    4 ``basename[0:4] + '.mtz'``, -1 nothing found.
    """
    base = os.path.basename(mtz_path)
    exact = os.path.join(phasing_path, base)
    if os.path.isfile(exact):
        return exact, 2
    stem = os.path.splitext(base)[0]
    matches = sorted(glob.glob(os.path.join(phasing_path, f"*{stem}*.mtz")))
    if matches:
        refined = [m for m in matches if m.endswith("_001.mtz")]
        candidate = refined[0] if refined else matches[0]
        return (candidate, 3) if os.path.isfile(candidate) else (None, 0)
    fallback = os.path.join(phasing_path, base[0:4] + ".mtz")
    if os.path.isfile(fallback):
        return fallback, 4
    return None, -1


def locate_model(sample, model_folder):
    """Find the model PDB upstream's blob search and tagging used for ``sample``.

    ``{model_folder}/{sample}.pdb`` first, then the first of
    ``sorted(glob('*{sample}*.pdb'))`` (blobs.py:104-106; tag.py:334). Upstream
    takes ``glob(...)[0]`` unsorted; sorting only matters when several files
    match, which the valdo-easy symlink farm never produces.
    """
    if not model_folder:
        return None
    exact = os.path.join(model_folder, f"{sample}.pdb")
    if os.path.isfile(exact):
        return exact
    matches = sorted(glob.glob(os.path.join(model_folder, f"*{sample}*.pdb")))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def map_from_mtz(path, f_col, phi_col, sample_rate=DEFAULT_SAMPLE_RATE, normalize=True):
    """Compute a map from one amplitude and one phase column of an MTZ file.

    Rows where either column is missing are dropped first, exactly as
    ``valdo.blobs.blob_helper`` does. With ``normalize`` the grid is scaled to
    mean 0 and RMS 1 over the unit cell (sigma units).

    Returns a gemmi.FloatGrid covering the full unit cell.
    """
    import reciprocalspaceship as rs

    ds = rs.read_mtz(path)
    missing = [c for c in (f_col, phi_col) if c not in ds.columns]
    if missing:
        raise KeyError(
            f"{os.path.basename(path)} has no column(s) {', '.join(missing)}; "
            f"columns present: {', '.join(map(str, ds.columns))}"
        )
    ds = ds[[f_col, phi_col]].dropna()
    if len(ds) == 0:
        raise ValueError(f"{os.path.basename(path)}: no reflection has both {f_col} and {phi_col}")
    mtz = ds.to_gemmi()
    grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=sample_rate)
    if normalize:
        grid.normalize()
    return grid


def grid_rms(grid):
    """Mean and RMS deviation of a grid's values (over the whole grid)."""
    arr = np.array(grid, copy=False)
    return float(arr.mean()), float(arr.std())


def box_around(cell, center, radius):
    """gemmi.FractionalBox enclosing a cube of half-width ``radius`` (Angstrom)."""
    import gemmi

    box = gemmi.FractionalBox()
    for dx, dy, dz in itertools.product((-radius, radius), repeat=3):
        box.extend(cell.fractionalize(gemmi.Position(center.x + dx, center.y + dy, center.z + dz)))
    return box


def write_ccp4(grid, path, center=None, radius=None):
    """Write a grid as a CCP4 map, optionally cut to a box around ``center``.

    The header keeps the statistics of the full grid (for a normalised grid:
    mean 0, RMS 1), so contour levels mean the same in the cut and the full
    map. The file is written to ``path + '.tmp'`` and renamed.

    Returns a dict describing the file (path, grid shape, spacing, box).
    """
    import gemmi

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header(2, True)
    info = {
        "path": path,
        "full_cell_shape": list(grid.shape),
        "spacing_A": [round(float(s), 4) for s in grid.spacing],
        "boxed": False,
    }
    if center is not None and radius:
        ccp4.set_extent(box_around(grid.unit_cell, center, float(radius)))
        info["boxed"] = True
        info["box_radius_A"] = float(radius)
        info["box_center"] = [round(center.x, 3), round(center.y, 3), round(center.z, 3)]
    info["shape"] = list(ccp4.grid.shape)
    tmp = path + ".tmp"
    ccp4.write_ccp4_map(tmp)
    os.replace(tmp, path)
    return info


def values_near(grid, center, radius):
    """Map values around a point.

    Returns a dict with the interpolated value at ``center`` and the maximum,
    minimum and largest-magnitude value over grid points within ``radius``
    Angstrom of ``center`` (periodic in the unit cell).
    """
    import gemmi

    cell = grid.unit_cell
    frac = cell.fractionalize(center)
    dims = np.array([grid.nu, grid.nv, grid.nw])
    spacing = np.array(grid.spacing, dtype=float)
    n = np.ceil(radius / spacing).astype(int) + 1
    base = np.floor(np.array([frac.x, frac.y, frac.z]) * dims).astype(int)
    arr = np.array(grid, copy=False)
    vmax, vmin = -np.inf, np.inf
    for i in range(-n[0], n[0] + 1):
        for j in range(-n[1], n[1] + 1):
            for k in range(-n[2], n[2] + 1):
                u, v, w = base[0] + i, base[1] + j, base[2] + k
                pos = cell.orthogonalize(gemmi.Fractional(u / dims[0], v / dims[1], w / dims[2]))
                if pos.dist(center) > radius:
                    continue
                val = float(arr[u % dims[0], v % dims[1], w % dims[2]])
                vmax = max(vmax, val)
                vmin = min(vmin, val)
    out = {"at_center": float(grid.interpolate_value(center))}
    if np.isfinite(vmax):
        out.update(max=vmax, min=vmin, max_abs=vmax if abs(vmax) >= abs(vmin) else vmin)
    else:
        out.update(max=out["at_center"], min=out["at_center"], max_abs=out["at_center"])
    return out


# ---------------------------------------------------------------------------
# Structure helpers (symmetry-aware)
# ---------------------------------------------------------------------------

def symmetry_image_near(cell, spacegroup, pos, target):
    """Return the image of ``pos`` (symmetry + lattice) closest to ``target``.

    Blob centroids from gemmi's flood fill lie in the unit cell, which is often
    not where the deposited model sits; viewers need the copy of the point
    next to the molecule. Returns (gemmi.Position, distance in Angstrom).
    """
    import gemmi

    f = cell.fractionalize(pos)
    t = cell.fractionalize(target)
    t_arr = np.array([t.x, t.y, t.z])
    best = (None, np.inf)
    for op in spacegroup.operations():
        g = np.array(op.apply_to_xyz([f.x, f.y, f.z]))
        base = g - np.round(g - t_arr)
        for shift in itertools.product((-1, 0, 1), repeat=3):
            cand = base + np.array(shift)
            cart = cell.orthogonalize(gemmi.Fractional(*cand))
            d = cart.dist(target)
            if d < best[1]:
                best = (cart, d)
    return best


def is_water(residue):
    return residue.name.upper() in WATER_NAMES


def nearest_residue(structure, pos, max_dist=12.0, skip_waters=True):
    """Nearest atom (and its residue) to ``pos``, including symmetry mates.

    Uses gemmi.NeighborSearch over the first model with the structure's cell,
    so atoms of symmetry-related molecules count (as in upstream tag.py).
    Returns a dict (chain, resname, seqid, icode, atom, dist, label, atom_pos,
    display_pos), where ``display_pos`` is the image of ``pos`` next to the
    deposited atom (use it to centre a viewer), or None when no atom is within
    ``max_dist``.
    """
    import gemmi

    model = structure[0]
    ns = gemmi.NeighborSearch(model, structure.cell, max_dist).populate()
    best = None
    for mark in ns.find_atoms(pos, "\0", radius=max_dist):
        cra = mark.to_cra(model)
        if skip_waters and is_water(cra.residue):
            continue
        d = structure.cell.find_nearest_pbc_image(pos, cra.atom.pos, mark.image_idx).dist()
        if best is None or d < best[0]:
            best = (d, cra, mark.image_idx)
    if best is None:
        return None
    d, cra, image_idx = best
    # The copy of `pos` next to the deposited atom is exactly as far from it as `pos` is from the
    # atom's image; if gemmi's helper does not give that, search the images explicitly.
    display = structure.cell.find_nearest_pbc_position(cra.atom.pos, pos, image_idx, True)
    if abs(display.dist(cra.atom.pos) - d) > 0.01:
        sg = structure.find_spacegroup() or gemmi.SpaceGroup("P 1")
        display = symmetry_image_near(structure.cell, sg, pos, cra.atom.pos)[0]
    seqid = cra.residue.seqid
    icode = seqid.icode.strip() if seqid.icode not in (" ", "\0") else ""
    return {
        "chain": cra.chain.name,
        "resname": cra.residue.name,
        "seqid": int(seqid.num),
        "icode": icode,
        "atom": cra.atom.name,
        "dist": float(d),
        "label": f"{cra.chain.name}/{cra.residue.name}{seqid.num}{icode}",
        "atom_pos": cra.atom.pos,
        "display_pos": display,
    }


def nearest_water_distance(structure, pos, max_dist=4.0):
    """Distance from ``pos`` to the nearest water oxygen (symmetry included), or None."""
    import gemmi

    model = structure[0]
    ns = gemmi.NeighborSearch(model, structure.cell, max_dist).populate()
    best = None
    for mark in ns.find_atoms(pos, "\0", radius=max_dist):
        cra = mark.to_cra(model)
        if not is_water(cra.residue):
            continue
        d = structure.cell.find_nearest_pbc_image(pos, cra.atom.pos, mark.image_idx).dist()
        best = d if best is None else min(best, d)
    return best


def residues_within(structure, pos, radius, resnames=None, hetatm_only=False):
    """Residues with any atom within ``radius`` of ``pos`` (symmetry included).

    ``resnames``: optional set of residue names to keep. Returns a sorted list
    of "chain/RESnum" labels.
    """
    import gemmi

    model = structure[0]
    ns = gemmi.NeighborSearch(model, structure.cell, max(radius, 1.0)).populate()
    hits = set()
    for mark in ns.find_atoms(pos, "\0", radius=radius):
        cra = mark.to_cra(model)
        if resnames is not None and cra.residue.name not in resnames:
            continue
        if hetatm_only and cra.residue.het_flag != "H":
            continue
        d = structure.cell.find_nearest_pbc_image(pos, cra.atom.pos, mark.image_idx).dist()
        if d <= radius:
            hits.add(f"{cra.chain.name}/{cra.residue.name}{cra.residue.seqid.num}")
    return sorted(hits)
