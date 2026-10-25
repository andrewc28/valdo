"""Twin laws for twin refinement (``phases.twin_law`` in valdo.yaml).

A twin law is written in hkl notation, the way PHENIX's ``xray_data.twin_law`` parameter and
REFMAC's output write it: ``-h,-k,l`` means that reflection (h, k, l) of one twin domain lies on
(-h, -k, l) of the other. As a matrix, the new indices are ``h' = h R`` (row vector times R),
the convention of :func:`valdo.easy.scan.hkl_triplet`, gemmi's ``Op.apply_to_hkl`` and
reciprocalspaceship's ``apply_symop``. Upper-case letters and spaces are accepted
(REFMAC writes ``-K, -H, -L``).

Twin laws that differ only by a symmetry operation of the crystal (or by -1: amplitudes obey
Friedel's law) describe the same twinning: for P 31 2 1, ``-h,-k,l`` and REFMAC's
``-K, -H, -L`` are the same law. :func:`same_law` compares laws this way, and
:func:`possible_laws` lists one law per such family for a cell and space group (the lattice
symmetry gemmi finds with ``find_twin_laws``, obliquity up to 3 degrees, as upstream reindex
uses it).

``phases.twin_law: auto`` means "the twin law of this crystal form": :func:`check_law` resolves it
to the single law the reference lattice allows, and refuses it when the lattice allows none or
several (then the human chooses, from what phenix.xtriage or the processing program reports).

Only :func:`possible_laws` and :func:`check_law` need gemmi; parsing, formatting and the
output parsers do not.
"""

import re
from fractions import Fraction

MAX_OBLIQUITY = 3.0          # degrees; the value upstream reindex passes to gemmi.find_twin_laws
AUTO = "auto"                # phases.twin_law: auto = the only twin law the reference lattice allows
IDENTITY = ((Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)))

_TERM = re.compile(r"([+-]?)(\d+(?:\.\d+)?(?:/\d+)?)?\*?([hkl])(?:/(\d+))?", re.I)


class TwinLawError(ValueError):
    """A twin law that cannot be read (the message says why and what is expected)."""


# ---------------------------------------------------------------------------
# Matrices (3x3 tuples of Fractions, row form: h' = h R)
# ---------------------------------------------------------------------------

def _mat(rows):
    return tuple(tuple(Fraction(x) for x in row) for row in rows)


def matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def neg(a):
    return tuple(tuple(-x for x in row) for row in a)


def det(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def inverse(a):
    d = det(a)
    if d == 0:
        raise TwinLawError("the matrix is singular")
    cof = [[(a[(i + 1) % 3][(j + 1) % 3] * a[(i + 2) % 3][(j + 2) % 3]
             - a[(i + 1) % 3][(j + 2) % 3] * a[(i + 2) % 3][(j + 1) % 3]) for j in range(3)] for i in range(3)]
    return tuple(tuple(cof[j][i] / d for j in range(3)) for i in range(3))


def op_matrix(op):
    """Row-form hkl matrix of a gemmi.Op (h' = h R with R = op.rot / DEN, as gemmi's apply_to_hkl)."""
    return tuple(tuple(Fraction(int(op.rot[i][j]), int(op.DEN)) for j in range(3)) for i in range(3))


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def parse_law(text):
    """The matrix of a twin law written in hkl notation (``-h,-k,l``, ``-K, -H, -L``, ``h+k,-k,-l``,
    ``1/2*h+3/2*k,...``). Raises TwinLawError with a plain-language reason."""
    if text is None or not str(text).strip():
        raise TwinLawError("no twin law given")
    raw = str(text).strip()
    s = raw.strip("[]()").replace(" ", "").replace("\t", "")
    if re.search(r"[xyzXYZ]", s):
        raise TwinLawError(f"'{raw}' uses x, y, z; write twin laws with h, k and l as PHENIX and REFMAC do "
                           "(e.g. -h,-k,l)")
    parts = s.split(",")
    if len(parts) != 3 or not all(parts):
        raise TwinLawError(f"'{raw}' is not a twin law: give three parts separated by commas, e.g. -h,-k,l")
    rows = [[Fraction(0)] * 3 for _ in range(3)]
    for j, part in enumerate(parts):
        pos = 0
        while pos < len(part):
            m = _TERM.match(part, pos)
            if not m or m.end() == pos or (pos and not m.group(1)):
                raise TwinLawError(f"'{raw}': cannot read '{part}' (expected terms such as -h, k, h+k or 2*l)")
            sign, coef, letter, div = m.groups()
            c = Fraction(coef) if coef else Fraction(1)
            if div:
                c /= int(div)
            if sign == "-":
                c = -c
            rows["hkl".index(letter.lower())][j] += c
            pos = m.end()
    return _mat(rows)


def format_law(matrix):
    """hkl text of a row-form matrix (the inverse of :func:`parse_law`), e.g. '-h,-k,l'."""
    out = []
    for j in range(3):
        terms = ""
        for i, letter in enumerate("hkl"):
            c = Fraction(matrix[i][j])
            if c == 0:
                continue
            sign = "-" if c < 0 else ("+" if terms else "")
            mag = abs(c)
            if mag == 1:
                coef = ""
            elif mag.denominator == 1:
                coef = f"{mag.numerator}*"
            else:
                coef = f"{mag.numerator}/{mag.denominator}*"
            terms += f"{sign}{coef}{letter}"
        out.append(terms or "0")
    return ",".join(out)


def normalise_text(text):
    """The canonical spelling of a twin law ('-K, -H, -L' -> '-k,-h,-l'); raises TwinLawError."""
    return format_law(parse_law(text))


def is_auto(text):
    """True for the setting ``auto`` (any case): the twin law is taken from the reference lattice."""
    return isinstance(text, str) and text.strip().lower() == AUTO


def normalise_setting(text):
    """phases.twin_law as written in valdo.yaml or given to --twin-law: 'auto', or a law's canonical
    spelling. Raises TwinLawError."""
    return AUTO if is_auto(text) else normalise_text(text)


# ---------------------------------------------------------------------------
# Lattice and symmetry (gemmi)
# ---------------------------------------------------------------------------

def rotations(space_group):
    """Rotation parts (hkl row form) of the space group's symmetry operations."""
    import gemmi
    sg = space_group if not isinstance(space_group, str) else gemmi.SpaceGroup(space_group)
    out = []
    for op in sg.operations().sym_ops:
        m = op_matrix(op)
        if m not in out:
            out.append(m)
    return out


def same_law(a, b, rots):
    """True if twin laws ``a`` and ``b`` (matrices) describe the same twinning: a = +-b g or +-g b for a
    symmetry rotation g (amplitudes are the same for every member of that family)."""
    for g in rots:
        for m in (matmul(b, g), matmul(g, b)):
            if a == m or a == neg(m):
                return True
    return False


def is_symmetry(matrix, rots):
    """True if the 'law' is a symmetry operation of the crystal (no twinning)."""
    return same_law(matrix, IDENTITY, rots)


def _nice(matrix, rots, preferred=None):
    """The simplest proper member of a law's family: fewest non-zero terms, then gemmi's own
    representative, then fewest minus signs."""
    cands = []
    for g in rots:
        for m in (matmul(matrix, g), matmul(g, matrix)):
            for c in (m, neg(m)):
                if det(c) == 1 and c not in cands:
                    cands.append(c)

    def key(c):
        flat = [x for row in c for x in row]
        return (sum(1 for x in flat if x != 0), 0 if c == preferred else 1, sum(1 for x in flat if x < 0),
                sum(1 for x in flat if x.denominator != 1), format_law(c))
    return min(cands, key=key) if cands else matrix


def possible_laws(cell, space_group, max_obliquity=MAX_OBLIQUITY):
    """One twin law per family that the lattice allows: [{"law": text, "matrix": matrix}], [] if none.

    ``cell``: six numbers (or a gemmi.UnitCell); ``space_group``: name (or gemmi.SpaceGroup)."""
    import gemmi
    uc = cell if isinstance(cell, gemmi.UnitCell) else gemmi.UnitCell(*[float(x) for x in cell])
    sg = space_group if not isinstance(space_group, str) else gemmi.SpaceGroup(space_group)
    rots = rotations(sg)
    out = []
    for op in gemmi.find_twin_laws(uc, sg, max_obliquity, False):
        m = op_matrix(op)
        nice = _nice(m, rots, preferred=m)
        out.append({"law": format_law(nice), "matrix": nice})
    return out


def conjugate(matrix, op):
    """A law found in another indexing setting, expressed in this one.

    ``op`` (gemmi.Op or matrix) maps the other setting's indices onto this one (h_this = h_other op,
    as find_frame_op returns it); the law becomes op^-1 L op."""
    o = op if isinstance(op, tuple) else op_matrix(op)
    return matmul(matmul(inverse(o), matrix), o)


def law_for_variant(law, best_op, variant_op, space_group):
    """The twin law to refine in one indexing variant of a dataset.

    ``law``: the law in the reference indexing (text); ``best_op`` / ``variant_op``: the re-indexing
    operators (hkl text, as the reindex stage applies them to the dataset's own indexing) of the
    variant that matches the reference and of the variant being refined. The variant's indices are
    h_v = h_best B^-1 V, so the law there is M^-1 L M with M = B^-1 V. For the twin laws of most
    lattices (P 3, P 31 2 1, P 41, ...) this is the same law; only lattices with several twin laws
    that do not commute (pseudo-cubic ones) need another law in some variants. Returns the law's
    text (``law`` itself when the twinning is the same)."""
    rots = rotations(space_group)
    m = matmul(inverse(parse_law(best_op)), parse_law(variant_op))
    matrix = parse_law(law)
    moved = conjugate(matrix, m)
    if same_law(moved, matrix, rots):
        return law
    return format_law(_nice(moved, rots))


def check_law(text, cell, space_group):
    """Is ``text`` a usable twin law for this cell and space group?

    ``text`` may also be ``auto``: the only twin law the lattice allows. Returns {"ok", "code",
    "given", "law" (canonical text to use), "equivalent" (the family's simplest law), "possible"
    ([texts]), "message", "fix", "note"}. Codes when not ok: twin_law_syntax, twin_law_is_symmetry,
    twin_law_none_possible, twin_law_invalid, twin_law_ambiguous (auto, several laws possible)."""
    import gemmi
    sg = space_group if not isinstance(space_group, str) else gemmi.SpaceGroup(space_group)
    sg_name = sg.hm
    out = {"ok": False, "code": None, "given": text, "law": None, "equivalent": None, "possible": [],
           "message": "", "fix": "", "note": None}
    try:
        possible = possible_laws(cell, sg)
    except Exception as exc:  # exotic cell or space group
        out.update(code="twin_law_invalid", message=f"the twin laws of {sg_name} could not be determined ({exc})",
                   fix="Check the reference dataset's cell and space group.")
        return out
    out["possible"] = [p["law"] for p in possible]
    options = (f"possible here: {', '.join(out['possible'])}" if possible else
               f"{sg_name} with this cell allows no twin law")
    if is_auto(text):
        if not possible:
            out.update(code="twin_law_none_possible",
                       message=f"twin_law auto: {sg_name} with the reference cell cannot be merohedrally twinned",
                       fix="Remove phases.twin_law (set it to null). If you expected twinning, check the space "
                           "group and cell of your data.")
        elif len(possible) > 1:
            out.update(code="twin_law_ambiguous",
                       message=f"twin_law auto: {sg_name} with the reference cell allows several twin laws "
                               f"({', '.join(out['possible'])})",
                       fix="Set phases.twin_law to the one phenix.xtriage or your processing program reports.")
        else:
            out.update(ok=True, law=out["possible"][0], equivalent=out["possible"][0],
                       message=f"twin law {out['possible'][0]} (auto: the only twin law of {sg_name} with the "
                               "reference cell)")
        return out
    try:
        matrix = parse_law(text)
    except TwinLawError as exc:
        out.update(code="twin_law_syntax", message=str(exc), fix=f"Write it like -h,-k,l ({options}).")
        return out
    d = det(matrix)
    if abs(d) != 1:
        out.update(code="twin_law_invalid",
                   message=f"'{text}' is not a symmetry operation of any lattice (determinant {d})",
                   fix=f"Use one of the laws {options}." if possible else
                   "Remove phases.twin_law: crystals of this form cannot be twinned by (pseudo-)merohedry.")
        return out
    inverted = d == -1
    if inverted:
        matrix = neg(matrix)
        out["note"] = (f"'{text}' includes an inversion; for amplitudes it is the same twin law as "
                       f"{format_law(matrix)}, which is used")
    rots = rotations(sg)
    if is_symmetry(matrix, rots):
        if matrix == IDENTITY:      # -h,-k,-l: h,k,l and -h,-k,-l have the same amplitude (Friedel's law)
            why = f"'{text}' only relates Friedel mates (h,k,l and -h,-k,-l have the same amplitude)"
        elif inverted:
            why = (f"'{text}' is a symmetry operation of {sg_name} combined with an inversion: it maps every "
                   "reflection onto a Friedel mate of a symmetry-equivalent one (the same amplitude)")
        else:
            why = f"'{text}' is a symmetry operation of {sg_name}: it maps the crystal onto itself"
        out.update(code="twin_law_is_symmetry", message=why + ", so it describes no twinning",
                   fix=(f"Use a twin law of this crystal form ({options})" if possible else
                        "Remove phases.twin_law (this crystal form cannot be twinned by (pseudo-)merohedry)") + ".")
        return out
    if not possible:
        out.update(code="twin_law_none_possible",
                   message=f"{sg_name} with the reference cell allows no (pseudo-)merohedral twinning, so no twin "
                           "law applies",
                   fix="Remove phases.twin_law (set it to null). If you expected twinning, check the space group "
                       "and cell of your data.")
        return out
    match = [p for p in possible if same_law(matrix, p["matrix"], rots)]
    if not match:
        out.update(code="twin_law_invalid",
                   message=f"'{text}' is not a twin law of {sg_name} with the reference cell (it is not a "
                           "symmetry of the lattice)",
                   fix=f"Use one of the laws {options}.")
        return out
    out.update(ok=True, law=format_law(matrix), equivalent=match[0]["law"],
               message=f"twin law {format_law(matrix)} ({sg_name}"
                       + (f"; the same law as {match[0]['law']}" if match[0]["law"] != format_law(matrix) else "")
                       + ")")
    return out


def twin_in_extra_args(program, extra_args):
    """The extra command-line argument that already asks for twin refinement, if any
    (phenix: ``xray_data.twin_law=...`` or ``twin_law=...``; dimple: ``--refmac-key twin...``)."""
    args = [str(a) for a in extra_args or []]
    for i, arg in enumerate(args):
        if program == "phenix" and re.search(r"(^|\.)twin_law\s*=", arg):
            return arg
        if program == "dimple":
            if arg == "--refmac-key" and i + 1 < len(args) and args[i + 1].strip().lower().startswith("twin"):
                return f"{arg} {args[i + 1]}"
            if arg.startswith("--refmac-key=") and arg.split("=", 1)[1].strip().lower().startswith("twin"):
                return arg
    return None


# ---------------------------------------------------------------------------
# What the refinement programs report
# ---------------------------------------------------------------------------

def parse_refmac_twin(pdb_text="", cif_text=""):
    """Twin domains in a REFMAC model: [{"law": text, "fraction": float}] (identity domain included),
    [] when REFMAC kept no twin domain, None when the file has no twin information.

    REFMAC writes (verbatim from refmacat in CCP4 9, dimple 2.7 with --refmac-key twin)::

        REMARK   3  TWIN DETAILS
        REMARK   3   NUMBER OF TWIN DOMAINS  :    2
        REMARK   3      TWIN DOMAIN   :    1
        REMARK   3      TWIN OPERATOR :  H,  K,  L
        REMARK   3      TWIN FRACTION : 0.751
        REMARK   3      TWIN DOMAIN   :    2
        REMARK   3      TWIN OPERATOR : -K, -H, -L
        REMARK   3      TWIN FRACTION : 0.249

    (``NUMBER OF TWIN DOMAINS  : NULL`` when no twin domain was kept), and in mmCIF a
    ``_pdbx_reflns_twin`` loop (operator, fraction to 4 decimals), which is preferred when given.
    """
    domains = _cif_twin(cif_text) if cif_text else None
    if domains:
        return domains
    if cif_text and re.search(r"^_ccp4_reflns_twin\.number_of_domains\s+1\s*$", cif_text, re.M) and \
            not re.search(r"^REMARK   3\s+TWIN OPERATOR", pdb_text or "", re.M):
        return []
    if not pdb_text or not re.search(r"^REMARK   3\s+TWIN DETAILS", pdb_text, re.M):
        return None
    ops = re.findall(r"^REMARK   3\s+TWIN OPERATOR\s*:\s*(.+?)\s*$", pdb_text, re.M)
    fractions = re.findall(r"^REMARK   3\s+TWIN FRACTION\s*:\s*([0-9.]+)", pdb_text, re.M)
    out = []
    for op, frac in zip(ops, fractions):
        try:
            out.append({"law": normalise_text(op), "fraction": float(frac)})
        except (TwinLawError, ValueError):
            continue
    return out


def parse_phenix_twin(cif_text="", pdb_text="", log_text=""):
    """Twin law and fraction from phenix.refine output: {"law", "fraction"} or None.

    The formats are those of real twin refinements (PHENIX 2.2.1-6174 and 1.21.2-5419,
    ``xray_data.twin_law=-h,-k,l``, S2 benchmark dataset 0053), in this order of preference:

    * mmCIF model (only when PHENIX writes one): ``_pdbx_reflns_twin.operator -h,-k,l`` and
      ``_pdbx_reflns_twin.fraction 0.2600``;
    * PDB model: ``REMARK   3  TWINNING INFORMATION.`` / ``REMARK   3   FRACTION: 0.260`` /
      ``REMARK   3   OPERATOR: -h,-k,l``;
    * log: the last ``r_work=0.0663 r_free=0.0748 twin_fraction=0.26 twin_law=-h,-k,l`` line (the
      final model), else the last ``| twin fraction: 0.26  twin operator: -h,-k,l |`` box, which
      PHENIX prints during the macro-cycles and is not always the final value.
    """
    found = _cif_twin(cif_text) if cif_text else None
    if found:
        twin = [d for d in found if d["law"] != format_law(IDENTITY)] or found
        return {"law": twin[-1]["law"], "fraction": twin[-1]["fraction"]}
    if pdb_text and re.search(r"^REMARK   3\s+TWINNING INFORMATION", pdb_text, re.M):
        frac = re.findall(r"^REMARK   3\s+FRACTION\s*:\s*([0-9.]+)", pdb_text, re.M)
        ops = re.findall(r"^REMARK   3\s+OPERATOR\s*:\s*(\S+)", pdb_text, re.M)
        if frac and ops:
            try:
                return {"law": normalise_text(ops[-1]), "fraction": float(frac[-1])}
            except (TwinLawError, ValueError):
                pass
    for pattern in (r"twin_fraction=([0-9.]+)\s+twin_law=(\S+)",
                    r"twin fraction:\s*([0-9.]+)\s+twin operator:\s*([^\s|]+)"):
        hits = re.findall(pattern, log_text or "")
        if hits:
            try:
                return {"law": normalise_text(hits[-1][1]), "fraction": float(hits[-1][0])}
            except (TwinLawError, ValueError):
                continue
    return None


def _cif_twin(text):
    """[{"law", "fraction"}] from _pdbx_reflns_twin (loop or single values), or None."""
    ops = fractions = None
    try:
        import gemmi
        block = gemmi.cif.read_string(text).sole_block()
        table = block.find("_pdbx_reflns_twin.", ["operator", "fraction"])
        if len(table):
            ops = [gemmi.cif.as_string(row[0]) for row in table]
            fractions = [row[1] for row in table]
    except Exception:
        ops = None
    if ops is None:   # without gemmi, or text that is not a whole mmCIF file
        op1 = re.findall(r"^\s*_pdbx_reflns_twin\.operator\s+(['\"]?)(.+?)\1\s*$", text, re.M)
        fr1 = re.findall(r"^\s*_pdbx_reflns_twin\.fraction\s+([0-9.]+)", text, re.M)
        if op1 and fr1:
            ops, fractions = [op1[-1][1]], [fr1[-1]]
    if not ops:
        return None
    out = []
    for op, frac in zip(ops, fractions):
        try:
            out.append({"law": normalise_text(op), "fraction": float(frac)})
        except (TwinLawError, ValueError):
            continue
    return out or None


def summarise_domains(domains, configured=None, op=None, rots=None):
    """Twin result of one REFMAC refinement in the input's indexing.

    ``domains``: :func:`parse_refmac_twin` output (None = no information); ``configured``: the law
    from valdo.yaml (text); ``op``: the operator that maps the refined data's indexing onto the input's
    (dimple re-indexed the data), or None; ``rots``: symmetry rotations (for comparing laws).
    Returns (law text, fraction or None, note)."""
    if domains is None:
        return "", None, "no twin information in the refined model"
    others = []
    for d in domains:
        m = parse_law(d["law"])
        if op is not None:
            m = conjugate(m, op)
        if m == IDENTITY or (rots and is_symmetry(m, rots)):
            continue
        others.append((m, d["fraction"]))
    if not others:
        return "", 0.0, ("REFMAC kept no twin domain (its tests found twinning unlikely or the twin fraction "
                         "below 0.07): refined as untwinned")
    cfg = parse_law(configured) if configured else None
    texts, notes = [], []
    for m, frac in others:
        if cfg is not None and rots is not None and same_law(m, cfg, rots):
            texts.append(format_law(cfg))
        else:
            texts.append(format_law(_nice(m, rots) if rots else m))
            if cfg is not None:
                notes.append(f"REFMAC kept twin law {texts[-1]} (fraction {frac:.3f}), not the configured "
                             f"{format_law(cfg)}")
    total = round(sum(f for _, f in others), 4)
    if len(others) > 1:
        notes.append("several twin domains: " + ", ".join(f"{t} {f:.3f}" for t, (_, f) in zip(texts, others)))
    return " + ".join(texts), total, "; ".join(notes)


def zero_fraction_text(program, n):
    """How the stage message and the report describe ``n`` refinements with a twin fraction of 0.

    dimple: REFMAC dropped the twin law and refined those datasets as untwinned. phenix.refine always
    refines with the law; its fraction can come out as 0 (an untwinned crystal)."""
    if program == "dimple":
        return f"{n} refined as untwinned (REFMAC kept no twin domain)"
    return f"{n} with a refined twin fraction of 0"
