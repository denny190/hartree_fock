#!/usr/bin/env python3
import os
import json
import math
import argparse
import numpy as np
from scipy.linalg import fractional_matrix_power

# -------------------------
# I/O
# -------------------------

# Minimal periodic table for common light elements (expand as needed)
SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10
}

ANGSTROM_PER_BOHR = 0.5291772109  # 1 bohr = 0.529177... Å

def read_input(input_file):
    """
    Very lightweight reader for the original educational input format.
    Expected layout (example):

    <title/ignored>
    2
    <ignored>
    0
    H  0.000000  0.000000  -0.700000
    H  0.000000  0.000000   0.700000

    Line 2 (index 1): natoms
    Line 4 (index 3): charge
    Lines 5..(4+natoms-1): symbol x y z  [Å]
    """
    with open(input_file, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 4:
        raise ValueError("Input file too short / malformed.")

    natoms = int(lines[1].split()[0])
    charge = int(lines[3].split()[0])

    labels = []
    Z = []
    coords = np.zeros((natoms, 3))
    for i in range(natoms):
        parts = lines[4 + i].split()
        sym = parts[0]
        if sym not in SYMBOL_TO_Z:
            raise ValueError(f"Element '{sym}' not in SYMBOL_TO_Z (add it if needed).")
        Z.append(SYMBOL_TO_Z[sym])
        labels.append(sym)
        # Convert Å -> bohr by dividing by 0.529177...
        coords[i, 0] = float(parts[1]) / ANGSTROM_PER_BOHR
        coords[i, 1] = float(parts[2]) / ANGSTROM_PER_BOHR
        coords[i, 2] = float(parts[3]) / ANGSTROM_PER_BOHR

    return natoms, charge, labels, Z, coords


# -------------------------
# Basis
# -------------------------

def build_basis(natoms, atomic_numbers, coordinates, basis_set_name, basis_directory="basis/"):
    """
    Build basis from BSE-style JSON files located in basis_directory.
    Educational note: current integrals assume *s-type* (l=0) GTOs only.

    Returns
    -------
    nb : int
        Number of basis functions.
    basis_set : list
        For each BF:
        [ bf_index, atom_index, n_primitives, exponents, coefficients, angular_momentum ]
    """
    bfno = 0
    basis_set = []

    basis_file = os.path.join(basis_directory, f"{basis_set_name}.json")
    if not os.path.exists(basis_file):
        raise FileNotFoundError(f"Basis set file not found: {basis_file}")

    with open(basis_file, "r") as f:
        basis_data = json.load(f)

    for atom_idx, Z in enumerate(atomic_numbers):
        key = str(Z)
        if key not in basis_data["elements"]:
            raise ValueError(f"Basis set not defined for Z={Z} in {basis_file}")

        element_basis = basis_data["elements"][key]
        for shell in element_basis["electron_shells"]:
            am_list = shell["angular_momentum"]
            exponents = [float(e) for e in shell["exponents"]]
            coefficients_sets = [[float(c) for c in coeffs] for coeffs in shell["coefficients"]]

            # EDUCATIONAL SCOPE: s-type only (l=0)
            for am_idx, am in enumerate(am_list):
                if am != 0:
                    raise NotImplementedError(
                        "This educational implementation supports s-type shells only. "
                        f"Found angular momentum l={am}. "
                        "Use an s-only basis JSON or extend the integrals."
                    )
                basis_set.append([
                    bfno,
                    atom_idx,
                    len(exponents),
                    exponents,
                    coefficients_sets[am_idx],
                    am
                ])
                bfno += 1

    return bfno, basis_set


# -------------------------
# Boys function (F0)
# -------------------------

def boys_F0(x):
    """Boys F0(x) with a small-x limit."""
    xsmall = 1.0e-6
    if x < xsmall:
        return 1.0
    return 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))


# -------------------------
# One- and Two-electron integrals (s-type)
# -------------------------

def compute_one_electron_integrals(num_atoms, num_basis, atomic_numbers, coordinates, basis_set):
    """
    Overlap S, kinetic T, nuclear attraction V for contracted s-GTOs.
    """
    S = np.zeros((num_basis, num_basis))
    T = np.zeros((num_basis, num_basis))
    V = np.zeros((num_basis, num_basis))

    for i in range(num_basis):
        ai = basis_set[i][1]
        Ai = coordinates[ai, :]
        npi = basis_set[i][2]
        exp_i = basis_set[i][3]
        cof_i = basis_set[i][4]

        for j in range(num_basis):
            aj = basis_set[j][1]
            Aj = coordinates[aj, :]
            npj = basis_set[j][2]
            exp_j = basis_set[j][3]
            cof_j = basis_set[j][4]

            Rij2 = np.dot(Ai - Aj, Ai - Aj)

            S_ij = 0.0
            T_ij = 0.0
            V_ij = 0.0

            for pi in range(npi):
                zi = exp_i[pi]
                ci = cof_i[pi]
                for pj in range(npj):
                    zj = exp_j[pj]
                    cj = cof_j[pj]

                    p = zi + zj
                    q = zi * zj / p
                    P = (zi * Ai + zj * Aj) / p

                    # Overlap contribution
                    pref = (math.pi / p) ** (1.5) * math.exp(-q * Rij2)
                    S_prim = ci * cj * pref
                    S_ij += S_prim

                    # Kinetic contribution (s-type closed form)
                    T_prim = q * (3.0 - 2.0 * q * Rij2) * S_prim
                    T_ij += T_prim

                    # Nuclear attraction over all nuclei
                    V_sum = 0.0
                    for A in range(num_atoms):
                        RA = coordinates[A, :]
                        RPA2 = np.dot(P - RA, P - RA)
                        V_sum += atomic_numbers[A] * boys_F0(p * RPA2)
                    V_prim = -2.0 * math.sqrt(p / math.pi) * S_prim * V_sum
                    V_ij += V_prim

            S[i, j] = S_ij
            T[i, j] = T_ij
            V[i, j] = V_ij

    return S, T, V


def compute_two_electron_integrals(num_basis, coordinates, basis_set):
    """
    Two-electron integrals (ij|kl) for contracted s-GTOs.
    Naive O(n^4 * prim^4) educational implementation with symmetry filling.
    """
    TEI = np.zeros((num_basis, num_basis, num_basis, num_basis))
    norm = math.sqrt(2.0) * (math.pi ** 1.25)

    for i in range(num_basis):
        ai = basis_set[i][1]; Ai = coordinates[ai, :]
        npi = basis_set[i][2]; exp_i = basis_set[i][3]; cof_i = basis_set[i][4]

        for j in range(i + 1):
            aj = basis_set[j][1]; Aj = coordinates[aj, :]
            npj = basis_set[j][2]; exp_j = basis_set[j][3]; cof_j = basis_set[j][4]
            Rij2 = np.dot(Ai - Aj, Ai - Aj)

            for k in range(num_basis):
                ak = basis_set[k][1]; Ak = coordinates[ak, :]
                npk = basis_set[k][2]; exp_k = basis_set[k][3]; cof_k = basis_set[k][4]

                for l in range(k + 1):
                    al = basis_set[l][1]; Al = coordinates[al, :]
                    npl = basis_set[l][2]; exp_l = basis_set[l][3]; cof_l = basis_set[l][4]
                    Rkl2 = np.dot(Ak - Al, Ak - Al)

                    # Index symmetry pruning
                    if i * (i + 1) // 2 + j < k * (k + 1) // 2 + l:
                        continue

                    val = 0.0
                    for pi in range(npi):
                        zi = exp_i[pi]; ci = cof_i[pi]
                        for pj in range(npj):
                            zj = exp_j[pj]; cj = cof_j[pj]

                            p = zi + zj
                            q = zi * zj / p
                            P = (zi * Ai + zj * Aj) / p
                            f12 = norm * math.exp(-q * Rij2) / p

                            for pk in range(npk):
                                zk = exp_k[pk]; ck = cof_k[pk]
                                for pl in range(npl):
                                    zl = exp_l[pl]; cl = cof_l[pl]

                                    r = zk + zl
                                    s = zk * zl / r
                                    Q = (zk * Ak + zl * Al) / r
                                    f34 = norm * math.exp(-s * Rkl2) / r

                                    RPQ2 = np.dot(P - Q, P - Q)
                                    rho = p * r / (p + r)
                                    F0 = boys_F0(rho * RPQ2)

                                    val += F0 * f12 * f34 * (ci * cj * ck * cl) / math.sqrt(p + r)

                    TEI[i, j, k, l] = val
                    TEI[j, i, k, l] = val
                    TEI[i, j, l, k] = val
                    TEI[j, i, l, k] = val
                    TEI[k, l, i, j] = val
                    TEI[l, k, i, j] = val
                    TEI[k, l, j, i] = val
                    TEI[l, k, j, i] = val

    return TEI


# -------------------------
# SCF building blocks
# -------------------------

def H_core(T, V):
    return T + V

def S_orthogonalized(S):
    """Symmetric orthogonalization: S^{-1/2}."""
    return fractional_matrix_power(S, -0.5)

def init_density(nbf):
    return np.zeros((nbf, nbf))

def construct_F(H, P, TEI):
    """Fock matrix from H, density P and two-electron integrals (s-type)."""
    nbf = H.shape[0]
    F = H.copy()
    # Educational (explicit) O(n^4) construction
    for i in range(nbf):
        for j in range(nbf):
            s = 0.0
            x = 0.0
            for k in range(nbf):
                for l in range(nbf):
                    s += P[k, l] * TEI[i, j, k, l]      # Coulomb (J)
                    x += P[k, l] * TEI[i, k, j, l]      # Exchange (K)
            F[i, j] += s - 0.5 * x
    return F

def diagonalize_F(F, S_half_inv):
    """Solve F' C' = C' e  with F' = S^{-1/2} F S^{-1/2}; return C in AO basis."""
    Fp = S_half_inv @ F @ S_half_inv
    eigvals, eigvecs = np.linalg.eigh(Fp)
    C = S_half_inv @ eigvecs
    return C, eigvals

def construct_P(C, occ_orbitals):
    """RHF closed-shell density from MO coefficients."""
    nbf = C.shape[0]
    P = np.zeros((nbf, nbf))
    for a in range(occ_orbitals):
        va = C[:, a]
        P += 2.0 * np.outer(va, va)
    return P

def electronic_energy(P, H, F):
    """E_elec = 0.5 * sum_ij P_ij (H_ij + F_ij)."""
    return 0.5 * float(np.sum(P * (H + F)))

def nuclear_repulsion_energy(Z, coords):
    E = 0.0
    for i in range(len(Z)):
        for j in range(i + 1, len(Z)):
            rij = np.linalg.norm(coords[i] - coords[j])
            E += Z[i] * Z[j] / rij
    return E

def energy_decomposition(P, T, V, TEI, Z, coords):
    """Return (ET, EV, EJ, EK, ENuc, ETot)."""
    nbf = P.shape[0]
    ET = float(np.sum(P * T))
    EV = float(np.sum(P * V))

    EJ = 0.0
    EK = 0.0
    for i in range(nbf):
        for j in range(nbf):
            for k in range(nbf):
                for l in range(nbf):
                    EJ += P[i, j] * P[k, l] * TEI[i, j, k, l]
                    EK -= 0.5 * P[i, j] * P[k, l] * TEI[i, k, j, l]
    EJ *= 0.5
    EK *= 0.5

    ENuc = nuclear_repulsion_energy(Z, coords)
    ETot = ET + EV + EJ + EK + ENuc
    return ET, EV, EJ, EK, ENuc, ETot


def scf(T, V, S, TEI, Z, charge, max_iter=100, conv_thresh=1e-6, verbose=True):
    """
    Simple RHF SCF for closed-shell systems (educational).
    Returns (E_elec, P, eigvals, C)
    """
    nbf = S.shape[0]
    H = H_core(T, V)
    S_half_inv = S_orthogonalized(S)
    P = init_density(nbf)

    nelec = int(sum(Z) - charge)
    if nelec % 2 != 0:
        raise NotImplementedError("This educational SCF handles closed-shell (even electron) only.")
    occ_orbs = nelec // 2

    E_old = 0.0
    for it in range(1, max_iter + 1):
        F = construct_F(H, P, TEI)
        C, eps = diagonalize_F(F, S_half_inv)
        P_new = construct_P(C, occ_orbs)
        E_elec = electronic_energy(P_new, H, F)

        if verbose:
            print(f"Iter {it:3d}: E_elec = {E_elec:.12f}  dE = {E_elec - E_old:.3e}")

        if abs(E_elec - E_old) < conv_thresh:
            if verbose:
                print(f"SCF converged in {it} iterations.")
            return E_elec, P_new, eps, C

        P = P_new
        E_old = E_elec

    print("WARNING: SCF did not converge within max_iter.")
    return E_elec, P, eps, C


# -------------------------
# Command-line interface
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Educational RHF (s-GTO) — run from terminal with dynamic input & basis."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to molecule input file.")
    parser.add_argument("--basis", "-b", required=True, help="Basis set name (JSON file basename).")
    parser.add_argument("--basis-dir", default="basis", help="Directory containing basis JSON files.")
    parser.add_argument("--max-iter", type=int, default=100, help="SCF max iterations.")
    parser.add_argument("--conv", type=float, default=1e-6, help="SCF energy convergence threshold (Hartree).")
    parser.add_argument("--quiet", action="store_true", help="Reduce SCF iteration printing.")
    args = parser.parse_args()

    # Read molecule
    natoms, charge, labels, Z, coords = read_input(args.input)
    nelec = int(sum(Z) - charge)

    print("== Molecule ==")
    print(f"  Atoms: {natoms}   Charge: {charge}   Electrons: {nelec}")
    for a, (sym, r) in enumerate(zip(labels, coords)):
        print(f"   {a:2d}  {sym:>2s}   {r[0]: .6f}  {r[1]: .6f}  {r[2]: .6f}  (bohr)")
    print()

    # Basis & integrals
    nb, basis = build_basis(natoms, Z, coords, args.basis, basis_directory=args.basis_dir)
    print(f"== Basis ==\n  Name: {args.basis}   Functions: {nb}\n")

    S, T, V = compute_one_electron_integrals(natoms, nb, Z, coords, basis)
    TEI = compute_two_electron_integrals(nb, coords, basis)

    # SCF
    E_elec, P, eps, C = scf(
        T, V, S, TEI, Z, charge,
        max_iter=args.max_iter,
        conv_thresh=args.conv,
        verbose=not args.quiet
    )

    # Energy breakdown
    ET, EV, EJ, EK, ENuc, ETot = energy_decomposition(P, T, V, TEI, Z, coords)

    print("\n== Energies (Hartree) ==")
    print(f"  E_elec            : {E_elec:.12f}")
    print(f"  Nuclear repulsion : {ENuc:.12f}")
    print(f"  E(RHF) = E_elec + E_nuc : {(E_elec + ENuc):.12f}")
    print("\n-- Decomposition --")
    print(f"  ET (kinetic)      : {ET:.12f}")
    print(f"  EV (nuc attract.) : {EV:.12f}")
    print(f"  EJ (Coulomb)      : {EJ:.12f}")
    print(f"  EK (Exchange)     : {EK:.12f}")
    print(f"  Total RHF         : {ETot:.12f}")

if __name__ == "__main__":
    main()
