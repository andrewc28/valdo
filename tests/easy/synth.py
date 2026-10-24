"""A toy fragment screen for the valdo-easy tests.

`make_toy_screen(out_dir)` writes a small trigonal crystal form (P 3 2 1, a = b = 36 A,
c = 44 A, 2.4-2.8 A data) with, per dataset: an MTZ file in the PTP1B/XChem layout (F-obs,
SIGF-obs, R-free-flags and 2FOFCWT/PHI2FOFCWT, FOFCWT/PHIFOFCWT from the apo model), a model
(PDB) in the dataset's own frame, and for the 'bound' datasets a ligand (residue LIG) that is
present in the data but not in the model, plus a bound-state model. A few datasets are
written in the alternative indexing setting (data, map coefficients, model and bound model
all transformed consistently), and optionally a few broken files. truth.json records which.

Structure factors are computed by direct summation with Gaussian atoms; the point is to have
realistic plumbing (columns, indexing, phases, models), not realistic crystallography.
"""

import json
import os

import numpy as np

SPACE_GROUP = "P 3 2 1"
CELL = (36.0, 36.0, 44.0, 90.0, 90.0, 120.0)
CENTER = (1.0 / 3.0, 0.0, 0.25)   # fractional; the molecule is built around it
N_RESIDUES = 40
FOCAL_SEQID = 10          # a CYS, like Cys215 in PTP1B
ATOM_Z = {"N": 7.0, "C": 6.0, "O": 8.0, "S": 16.0}


def _orth(cell):
    import gemmi
    return np.array(gemmi.UnitCell(*cell).orth.mat.tolist())   # (orthogonalization_matrix is gone in gemmi 0.7)


def _ops(sg_name):
    import gemmi
    out = []
    for op in gemmi.SpaceGroup(sg_name).operations():
        rot = np.array(op.rot, dtype=float) / op.DEN
        tran = np.array(op.tran, dtype=float) / op.DEN
        out.append((rot, tran))
    return out


def _asu_hkl(cell, sg_name, d_min):
    import gemmi
    uc = gemmi.UnitCell(*cell)
    sg = gemmi.SpaceGroup(sg_name)
    asu = gemmi.ReciprocalAsu(sg)
    gops = sg.operations()
    hmax = [int(np.ceil(x / d_min)) + 1 for x in cell[:3]]
    out = []
    for h in range(-hmax[0], hmax[0] + 1):
        for k in range(-hmax[1], hmax[1] + 1):
            for l in range(-hmax[2], hmax[2] + 1):
                if (h, k, l) == (0, 0, 0) or not asu.is_in([h, k, l]) or gops.is_systematically_absent([h, k, l]):
                    continue
                if uc.calculate_d([h, k, l]) >= d_min:
                    out.append((h, k, l))
    return np.array(out, dtype=int)


def _stol2(cell, hkl):
    import gemmi
    uc = gemmi.UnitCell(*cell)
    return np.array([0.25 / uc.calculate_d(list(map(int, h))) ** 2 for h in hkl])


def structure_factors(hkl, atoms, cell=CELL, sg_name=SPACE_GROUP):
    """Complex F(hkl) for atoms [(element, cartesian xyz, B, occupancy)] by direct summation."""
    frac = np.linalg.inv(_orth(cell))
    xyz = np.array([a[1] for a in atoms]) @ frac.T
    z = np.array([ATOM_Z[a[0]] for a in atoms])
    b = np.array([a[2] for a in atoms])
    occ = np.array([a[3] for a in atoms])
    s2 = _stol2(cell, hkl)
    f = occ[None, :] * z[None, :] * np.exp(-(b[None, :] + 8.0) * s2[:, None])
    total = np.zeros(len(hkl), dtype=complex)
    for rot, tran in _ops(sg_name):
        x = xyz @ rot.T + tran
        total += (f * np.exp(2j * np.pi * (hkl @ x.T))).sum(axis=1)
    return total


def contact_distances(points, others, same_copy=True, cell=CELL, sg_name=SPACE_GROUP):
    """Shortest distance from each point to any crystal copy (symmetry operation plus lattice
    translation) of the `others`. same_copy=False leaves out the `others` themselves (identity,
    no translation): the contacts of a molecule with its neighbours in the crystal, including
    an atom with its own image when it sits near a symmetry axis."""
    orth = _orth(cell)
    frac = np.linalg.inv(orth)
    fp = np.asarray(points, dtype=float).reshape(-1, 3) @ frac.T
    fo = np.asarray(others, dtype=float).reshape(-1, 3) @ frac.T
    local = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    best = np.full(len(fp), np.inf)
    for rot, tran in _ops(sg_name):
        identity = np.allclose(rot, np.eye(3)) and np.allclose(tran, 0.0)
        raw = (fo @ rot.T + tran)[None, :, :] - fp[:, None, :]
        base = -np.round(raw)
        for t in local:
            shift = base + t
            d = np.linalg.norm((raw + shift) @ orth.T, axis=-1)
            if identity and not same_copy:
                d = np.where(np.all(shift == 0, axis=-1), np.inf, d)
            best = np.minimum(best, d.min(axis=1))
    return best


def toy_protein(rng, min_contact=3.5):
    """[(resname, seqid, [(atom name, element, xyz)])]: a compact poly-ALA chain with a CYS.

    The chain is a random walk around CENTER (12 A from the 3-fold axes, 11 A from the 2-fold
    planes). Residues that would come closer than `min_contact` to a crystal neighbour are left
    out (gaps in the chain are fine here), so no atom sits on or near a symmetry axis: such atoms
    are counted several times by the structure-factor sum and give artificial density peaks.
    """
    orth = _orth(CELL)
    center = np.array(CENTER) @ orth.T
    for _attempt in range(100):
        walk = []
        ca = center.copy()
        direction = rng.normal(size=3)
        for seqid in range(1, N_RESIDUES + 1):
            direction = direction / np.linalg.norm(direction)
            step = direction + 0.8 * rng.normal(size=3)
            step -= 0.08 * (ca - center)          # stay compact
            ca = ca + 3.8 * step / np.linalg.norm(step)
            direction = step
            offs = rng.normal(size=(4, 3))
            offs = 1.5 * offs / np.linalg.norm(offs, axis=1, keepdims=True)
            atoms = [("N", "N", ca + offs[0]), ("CA", "C", ca), ("C", "C", ca + offs[1]),
                     ("O", "O", ca + 1.6 * offs[1] / 1.5 + 0.6 * offs[2] / 1.5), ("CB", "C", ca + offs[3])]
            name = "ALA"
            if seqid == FOCAL_SEQID:
                name = "CYS"
                atoms.append(("SG", "S", ca + 2.0 * offs[3] / 1.5 + 0.5))
            walk.append((name, seqid, atoms))
        # the focal residue first, then the chain in order; a residue is kept when none of its
        # atoms comes too close to a crystal neighbour of the residues kept so far (itself included)
        kept, kept_xyz = [], []
        for res in sorted(walk, key=lambda r: (r[1] != FOCAL_SEQID, r[1])):
            xyz = [x for _, _, x in res[2]]
            if contact_distances(xyz, kept_xyz + xyz, same_copy=False).min() >= min_contact:
                kept.append(res)
                kept_xyz += xyz
        if kept and kept[0][1] == FOCAL_SEQID and len(kept) >= 0.75 * N_RESIDUES:
            return sorted(kept, key=lambda r: r[1])
    raise RuntimeError("could not build a toy protein without symmetry clashes")


def _ring(center):
    ring = []
    for i in range(6):
        t = 2 * np.pi * i / 6
        ring.append((f"C{i + 1}", "C", center + 1.4 * np.array([np.cos(t), np.sin(t), 0.0])))
    ring.append(("O1", "O", center + np.array([0.0, 0.0, 1.4])))
    return ring


def ligand_sites(residues, rng, n):
    """n ligand rings [(seqid of the nearest residue, center, atoms)] on the protein surface:
    3.3-4.5 A from the nearest protein atom (any crystal copy), >= 9 A from the focal residue,
    >= 4 A from their own symmetry images and >= 8 A from each other."""
    protein = [x for _, _, atoms in residues for _, _, x in atoms]
    owner = [seqid for _, seqid, atoms in residues for _ in atoms]
    focal = [x for _, seqid, atoms in residues if seqid == FOCAL_SEQID for _, _, x in atoms]
    xyz_protein = np.array(protein)
    lo, hi = xyz_protein.min(axis=0) - 5.0, xyz_protein.max(axis=0) + 5.0
    sites = []
    for _ in range(20000):
        center = rng.uniform(lo, hi)
        ring = _ring(center)
        xyz = [x for _, _, x in ring]
        closest = contact_distances(xyz, protein).min()
        if not 3.3 <= closest <= 4.5 or contact_distances([center], focal).min() < 9.0:
            continue
        if contact_distances(xyz, xyz, same_copy=False).min() < 4.0:
            continue
        if any(contact_distances([center], [other]).min() < 8.0 for _, other, _ in sites):
            continue
        near = owner[int(np.argmin(np.linalg.norm(xyz_protein - center, axis=1)))]
        sites.append((int(near), center, ring))
        if len(sites) == n:
            return sites
    raise RuntimeError("could not place the toy ligands")


def write_model(path, residues, ligand=None, b_iso=20.0, cell=CELL, sg_name=SPACE_GROUP):
    import gemmi
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*cell)
    st.spacegroup_hm = sg_name
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    serial = 0
    for name, seqid, atoms in residues + ([("LIG", 101, ligand)] if ligand else []):
        res = gemmi.Residue()
        res.name = name
        res.seqid = gemmi.SeqId(seqid, " ")
        if name == "LIG":
            res.het_flag = "H"
        for aname, element, xyz in atoms:
            serial += 1
            atom = gemmi.Atom()
            atom.name = aname
            atom.element = gemmi.Element(element)
            atom.pos = gemmi.Position(*map(float, xyz))
            atom.b_iso = b_iso
            atom.occ = 1.0
            atom.serial = serial
            res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    st.write_pdb(path)


def _transform_residues(residues, op_triplet):
    """Coordinates for data reindexed with `op` (h' = h R): x' = R^-1 x (fractional)."""
    import gemmi
    if op_triplet in (None, "x,y,z"):
        return residues
    inv = gemmi.Op(op_triplet).inverse()
    rot = np.array(inv.rot, dtype=float) / gemmi.Op.DEN
    orth = _orth(CELL)
    m = orth @ rot @ np.linalg.inv(orth)
    return [(n, s, [(a, e, m @ np.asarray(x)) for a, e, x in atoms]) for n, s, atoms in residues]


def _map_coefficients(fobs, fcalc):
    """2mFo-DFc and mFo-DFc (m = D = 1) with the model phases; amplitudes kept positive."""
    scale = np.sum(fobs * np.abs(fcalc)) / np.sum(np.abs(fcalc) ** 2)
    fo = fobs / scale
    phic = np.degrees(np.angle(fcalc))
    out = {}
    for label, value in (("2FOFC", 2 * fo - np.abs(fcalc)), ("FOFC", fo - np.abs(fcalc))):
        amp = np.abs(value)
        phase = np.where(value < 0, phic + 180.0, phic)
        out[label] = (amp, ((phase + 180.0) % 360.0) - 180.0)
    return out


def make_toy_screen(out_dir, n_datasets=24, n_bound=4, n_flipped=3, n_broken=1, n_bad=0, seed=7,
                    amplitude="F-obs", sigma="SIGF-obs", with_phases=True, names="TOY-x{:04d}"):
    """Write the toy screen into out_dir; returns the truth dict (also out_dir/truth.json).

    n_bad datasets (never the first, which becomes the reference) get their amplitudes shuffled
    over the reflections: they read and scale fine, but do not correlate with the others, so the
    filter stage drops them (min_cc)."""
    import gemmi
    import reciprocalspaceship as rs

    rng = np.random.default_rng(seed)
    for sub in ("data", "models", "bound"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    residues = toy_protein(rng)
    write_model(os.path.join(out_dir, "apo.pdb"), residues)
    twin = gemmi.find_twin_laws(gemmi.UnitCell(*CELL), gemmi.SpaceGroup(SPACE_GROUP), 3, False)
    flip_op = twin[0].triplet() if len(twin) else None
    hkl_all = _asu_hkl(CELL, SPACE_GROUP, 2.4)
    uc = gemmi.UnitCell(*CELL)
    d_all = np.array([uc.calculate_d(list(map(int, h))) for h in hkl_all])
    order = rng.permutation(n_datasets)
    bound_order = order[:n_bound].tolist()
    sites = ligand_sites(residues, rng, n_bound) if n_bound else []
    bound_idx = set(bound_order)
    # one bound dataset is also in the alternative setting: its ligand must still be found
    flipped_idx = set(([bound_order[0]] if n_bound and n_flipped else []) +
                      order[n_bound:n_bound + max(0, n_flipped - 1)].tolist()) if flip_op else set()
    used = set(bound_idx) | flipped_idx | {0}
    bad_idx = set([int(i) for i in order if int(i) not in used][:n_bad])
    truth = {"space_group": SPACE_GROUP, "cell": CELL, "flip_op": flip_op, "focal_seqid": FOCAL_SEQID,
             "datasets": [], "bound": [], "flipped": [], "broken": [], "bad": [], "ligands": {}}
    free = rng.integers(0, 20, size=len(hkl_all))
    for i in range(n_datasets):
        name = names.format(i + 1)
        d_min = float(rng.uniform(2.4, 2.8))
        keep = d_all >= d_min
        hkl = hkl_all[keep]
        noisy = [(n, s, [(a, e, x + rng.normal(0, 0.05, 3)) for a, e, x in atoms]) for n, s, atoms in residues]
        b = 18.0 + rng.normal(0, 1.5)
        atoms = [(e, x, b, 1.0) for _, _, at in noisy for _, e, x in at]
        bound = i in bound_idx
        lig = None
        if bound:
            near, _, lig = sites[bound_order.index(i)]
            atoms += [(e, x, b, 1.0) for _, e, x in lig]
            truth["ligands"][name] = {"near": int(near), "center": np.mean([a[2] for a in lig], axis=0).tolist()}
        ftrue = structure_factors(hkl, atoms)
        fcalc = structure_factors(hkl, [(e, x, b, 1.0) for _, _, at in noisy for _, e, x in at])
        s2 = _stol2(CELL, hkl)
        k = rng.uniform(0.7, 1.3)
        famp = k * np.abs(ftrue) * np.exp(-rng.uniform(-2, 2) * s2)
        sig = 0.03 * famp + 0.02 * famp.mean()
        fobs = np.abs(famp + rng.normal(0, 1, len(famp)) * sig) + 1e-3
        if i in bad_idx:
            shuffle = rng.permutation(len(fobs))
            fobs, sig = fobs[shuffle], sig[shuffle]
            truth["bad"].append(name)
        mc = _map_coefficients(fobs, fcalc)
        ds = rs.DataSet({c: rs.DataSeries(hkl[:, j], dtype="HKL") for j, c in enumerate("HKL")},
                        cell=gemmi.UnitCell(*CELL), spacegroup=gemmi.SpaceGroup(SPACE_GROUP)).set_index(["H", "K", "L"])
        ds[amplitude] = rs.DataSeries(fobs, dtype="SFAmplitude", index=ds.index)
        ds[sigma] = rs.DataSeries(sig, dtype="Stddev", index=ds.index)
        ds["R-free-flags"] = rs.DataSeries(free[keep], dtype="MTZInt", index=ds.index)
        if with_phases:
            ds["2FOFCWT"] = rs.DataSeries(mc["2FOFC"][0], dtype="SFAmplitude", index=ds.index)
            ds["PHI2FOFCWT"] = rs.DataSeries(mc["2FOFC"][1], dtype="Phase", index=ds.index)
            ds["FOFCWT"] = rs.DataSeries(mc["FOFC"][0], dtype="SFAmplitude", index=ds.index)
            ds["PHIFOFCWT"] = rs.DataSeries(mc["FOFC"][1], dtype="Phase", index=ds.index)
        ds.merged = True
        op = flip_op if i in flipped_idx else None
        if op:
            ds = ds.apply_symop(gemmi.Op(op)).hkl_to_asu()
            if "M/ISYM" in ds.columns:
                ds = ds.drop(columns=["M/ISYM"])
        mtz_path = os.path.join(out_dir, "data", f"{name}.mtz")
        ds.write_mtz(mtz_path)
        own = _transform_residues(noisy, op)
        write_model(os.path.join(out_dir, "models", f"{name}-pandda-input.pdb"), own)
        if bound:
            lig_own = _transform_residues([("LIG", 101, lig)], op)[0][2]
            write_model(os.path.join(out_dir, "bound", f"{name}_bound-state.pdb"), own, ligand=lig_own)
            truth["bound"].append(name)
        if op:
            truth["flipped"].append(name)
        truth["datasets"].append({"name": name, "d_min": d_min, "bound": bound, "flipped": bool(op),
                                  "bad": i in bad_idx})
    for j in range(n_broken):
        name = names.format(n_datasets + j + 1)
        with open(os.path.join(out_dir, "data", f"{name}.mtz"), "wb") as handle:
            handle.write(b"MTZ \x00\x00\x00\x00 truncated file")
        truth["broken"].append(name)
    with open(os.path.join(out_dir, "truth.json"), "w") as handle:
        json.dump(truth, handle, indent=1)
    return truth
