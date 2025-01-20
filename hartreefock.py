import numpy as np
import scipy as sp
import os, sys
import math
import json

def ReadInput(input_file):
    input=open(input_file,"r")
    angtobohr = 0.5291772109
    natoms=0
    z=[]
    i=0
    labels=[]
    charge=0

    for index,line in enumerate(input): 
        if index==1:
            natoms=int(line.split()[0]) 
            coords=np.zeros((natoms,3))
        if index==3:
            charge=int(line.split()[0])
        if index > 3 and index < (natoms+4):
            aux=line.split()
            atomic_name=aux[0]     #read and store the name of each atom
            if atomic_name == "H":
                atomic_num = 1
            if atomic_name == "He":
                atomic_num = 2
            coord=[float(aux[1])/angtobohr,float(aux[2])/angtobohr,float(aux[3])/angtobohr]
            labels.append(atomic_name) #we add the data (atom names) to a list
            z.append(atomic_num) # we add the data (atomic number) to a list
            coords[i,:]=coord  #we add the coordinates of each atom
            i=i+1
    return natoms,charge,labels,z,coords

def BuildBasis(natoms, atomic_numbers, coordinates, basis_set_name, basis_directory="basis/"):
    """
    Build basis from basis sets in JSON files located the basis dir.
    Basis sets sourced from the Basis Set Exchange.

    Args:
        natoms (int): Number of atoms.
        atomic_numbers (list): List of atomic numbers for each atom.
        coordinates (np.ndarray): Atomic coordinates.
        basis_directory (str): Directory containing JSON files for basis sets.

    Returns:
        int: Number of basis functions.
        list: Basis set information.
    """
    bfno = 0
    basis_set = []

    basis_file = os.path.join(basis_directory, f"{basis_set_name}.json")
    if not os.path.exists(basis_file):
        raise FileNotFoundError(f"Basis set file not found: {basis_file}")

    with open(basis_file, "r") as file:
        basis_data = json.load(file)

    # Iteration over atoms loaded in input
    # basis set JSON's contain all info about atom constants that we need, so no need to maintain a separate periodic table
    for atom_idx, atomic_number in enumerate(atomic_numbers):
        if str(atomic_number) not in basis_data["elements"]:
            raise ValueError(f"Basis set not defined for atomic number {atomic_number}")

        element_basis = basis_data["elements"][str(atomic_number)]

        # Parse electron shells for the current atom
        for shell in element_basis["electron_shells"]:
            angular_momentum = shell["angular_momentum"]
            exponents = [float(e) for e in shell["exponents"]]
            coefficients = [[float(c) for c in coeff_set] for coeff_set in shell["coefficients"]]

            # Add basis functions
            for ang_mom_idx, ang_mom in enumerate(angular_momentum):
                basis_set.append(
                    [
                        bfno,  # Basis function index
                        atom_idx,  # Atom index
                        len(exponents),  # Number of primitives
                        exponents,  # Primitive exponents
                        coefficients[ang_mom_idx],  # Primitive coefficients
                        ang_mom,  # Angular momentum (e.g., s, p, d, ...)
                    ]
                )
                bfno += 1

    return bfno, basis_set

def BoysFunc(r):
   xsmall = 1.e-6
   if r < xsmall:
      b = 1.
   else:
      b = 0.5 * np.sqrt(np.pi/r) * math.erf(np.sqrt(r))
   return b 

def compute_one_electron_integrals(num_atoms, num_basis, atomic_numbers, coordinates, basis_set):
    """
    Compute one-electron integrals: overlap (S), kinetic energy (T), and nuclear attraction (V).
    """
    overlap_matrix = np.zeros((num_basis, num_basis))
    kinetic_energy_matrix = np.zeros((num_basis, num_basis))
    nuclear_attraction_matrix = np.zeros((num_basis, num_basis))

    for basis_fn1 in range(num_basis):
        atom1_idx = basis_set[basis_fn1][1]
        atom1_coords = coordinates[atom1_idx, :]

        for basis_fn2 in range(num_basis):
            atom2_idx = basis_set[basis_fn2][1]
            atom2_coords = coordinates[atom2_idx, :]

            for prim1_idx in range(basis_set[basis_fn1][2]):
                zeta1 = basis_set[basis_fn1][3][prim1_idx]
                coeff1 = basis_set[basis_fn1][4][prim1_idx]

                for prim2_idx in range(basis_set[basis_fn2][2]):
                    zeta2 = basis_set[basis_fn2][3][prim2_idx]
                    coeff2 = basis_set[basis_fn2][4][prim2_idx]

                    p = zeta1 + zeta2
                    q = zeta1 * zeta2 / p
                    gaussian_center = (zeta1 * atom1_coords + zeta2 * atom2_coords) / p

                    distance_squared = np.linalg.norm(atom2_coords - atom1_coords) ** 2
                    gaussian_overlap = (
                        coeff1
                        * coeff2
                        * (np.pi / p) ** (3 / 2)
                        * np.exp(-q * distance_squared)
                    )

                    overlap_matrix[basis_fn1, basis_fn2] += gaussian_overlap

                    kinetic_contribution = q * (3.0 - 2.0 * q * distance_squared)
                    kinetic_energy_matrix[basis_fn1, basis_fn2] += kinetic_contribution * gaussian_overlap

                    attraction_term = 0.0
                    for nucleus_idx in range(num_atoms):
                        nucleus_coords = coordinates[nucleus_idx, :]
                        nucleus_distance_squared = np.linalg.norm(
                            nucleus_coords - gaussian_center
                        ) ** 2
                        boys_factor = BoysFunc(p * nucleus_distance_squared)
                        attraction_term += boys_factor * atomic_numbers[nucleus_idx]

                    nuclear_attraction_matrix[basis_fn1, basis_fn2] -= (
                        2.0 * attraction_term * np.sqrt(p / np.pi) * gaussian_overlap
                    )

    return overlap_matrix, kinetic_energy_matrix, nuclear_attraction_matrix


def compute_two_electron_integrals(num_basis, coordinates, basis_set):
    """
    Compute two-electron integrals using the given basis set and coordinates.
    """
    two_electron_integrals = np.zeros((num_basis, num_basis, num_basis, num_basis))
    normalization_factor = np.sqrt(2.0) * (np.pi ** 1.25)

    for basis_fn1 in range(num_basis):
        atom1_idx = basis_set[basis_fn1][1]
        atom1_coords = coordinates[atom1_idx, :]

        for basis_fn2 in range(basis_fn1 + 1):
            atom2_idx = basis_set[basis_fn2][1]
            atom2_coords = coordinates[atom2_idx, :]
            distance12_squared = np.linalg.norm(atom2_coords - atom1_coords) ** 2

            for basis_fn3 in range(num_basis):
                atom3_idx = basis_set[basis_fn3][1]
                atom3_coords = coordinates[atom3_idx, :]

                for basis_fn4 in range(basis_fn3 + 1):
                    atom4_idx = basis_set[basis_fn4][1]
                    atom4_coords = coordinates[atom4_idx, :]
                    distance34_squared = np.linalg.norm(atom4_coords - atom3_coords) ** 2

                    if basis_fn1 * (basis_fn1 + 1) // 2 + basis_fn2 < basis_fn3 * (basis_fn3 + 1) // 2 + basis_fn4:
                        continue

                    integral_value = 0.0
                    for prim1_idx in range(basis_set[basis_fn1][2]):
                        zeta1 = basis_set[basis_fn1][3][prim1_idx]
                        coeff1 = basis_set[basis_fn1][4][prim1_idx]

                        for prim2_idx in range(basis_set[basis_fn2][2]):
                            zeta2 = basis_set[basis_fn2][3][prim2_idx]
                            coeff2 = basis_set[basis_fn2][4][prim2_idx]

                            p = zeta1 + zeta2
                            q = zeta1 * zeta2 / p
                            gaussian_center1 = (zeta1 * atom1_coords + zeta2 * atom2_coords) / p
                            factor12 = normalization_factor * np.exp(-q * distance12_squared) / p

                            for prim3_idx in range(basis_set[basis_fn3][2]):
                                zeta3 = basis_set[basis_fn3][3][prim3_idx]
                                coeff3 = basis_set[basis_fn3][4][prim3_idx]

                                for prim4_idx in range(basis_set[basis_fn4][2]):
                                    zeta4 = basis_set[basis_fn4][3][prim4_idx]
                                    coeff4 = basis_set[basis_fn4][4][prim4_idx]

                                    pk = zeta3 + zeta4
                                    qk = zeta3 * zeta4 / pk
                                    gaussian_center2 = (
                                        (zeta3 * atom3_coords + zeta4 * atom4_coords) / pk
                                    )
                                    factor34 = normalization_factor * np.exp(-qk * distance34_squared) / pk

                                    inter_center_distance_squared = np.linalg.norm(
                                        gaussian_center2 - gaussian_center1
                                    ) ** 2
                                    rho = p * pk / (p + pk)
                                    boys_factor = BoysFunc(rho * inter_center_distance_squared)

                                    integral_contribution = (
                                        boys_factor
                                        * factor12
                                        * factor34
                                        / np.sqrt(p + pk)
                                        * coeff1
                                        * coeff2
                                        * coeff3
                                        * coeff4
                                    )
                                    integral_value += integral_contribution

                    # Symmetry considerations
                    two_electron_integrals[basis_fn1, basis_fn2, basis_fn3, basis_fn4] = integral_value
                    two_electron_integrals[basis_fn2, basis_fn1, basis_fn3, basis_fn4] = integral_value
                    two_electron_integrals[basis_fn1, basis_fn2, basis_fn4, basis_fn3] = integral_value
                    two_electron_integrals[basis_fn2, basis_fn1, basis_fn4, basis_fn3] = integral_value
                    two_electron_integrals[basis_fn3, basis_fn4, basis_fn1, basis_fn2] = integral_value
                    two_electron_integrals[basis_fn4, basis_fn3, basis_fn1, basis_fn2] = integral_value
                    two_electron_integrals[basis_fn3, basis_fn4, basis_fn2, basis_fn1] = integral_value
                    two_electron_integrals[basis_fn4, basis_fn3, basis_fn2, basis_fn1] = integral_value

    return two_electron_integrals



def H_core(T, V):
    return T + V

def S_orthogonalized(S):
    '''
    Orthogonalization of the overlap matrix.
    '''
    from scipy.linalg import fractional_matrix_power
    return fractional_matrix_power(S, -0.5)

def init_P(nb):
    '''
    Initial guess for the density matrix:
    Just an zero matrix with the dimensions of the number of basis functions (nb)
    '''
    return np.zeros((nb, nb))

def _construct_F(H, P, TEI, nb):
    '''
    Using the TEI, H_core and P we construct the F matrix
    '''
    F = np.copy(H)
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for l in range(nb):
                    #adding contributions from repulsion
                    F[i, j] += P[k, l] * (2 * TEI[i, j, k, l] - TEI[i, k, j, l])
    return F

def construct_F(H, P, TEI, nb):
    '''
    Using the TEI, H_core and P we construct the F matrix. v2
    '''
    F = np.copy(H)
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for l in range(nb):
                    F[i, j] += P[k, l] * TEI[i, j, k, l] #coulomb term
                    F[i, j] -= 0.5 * P[k, l] * TEI[i, k, j, l] #exchange term
    return F

def diag_F(F, S_half_inv):
    '''
    Diagonalizing the F matrix
    '''
    F_prime = S_half_inv @ F @ S_half_inv #Transform F to orthogonalized basis - F'=<S|F|S>
    eigvals, eigvecs = np.linalg.eigh(F_prime) # Solve for eigvals and eigvecs
    C = S_half_inv @ eigvecs #Obtaining C - the MO coefficients
    return C, eigvals

def construct_P(C, nb, charge):
    '''
    Update of density matrix using the MO coefficients retrieved from the F diagonalization
    '''
    P_new = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            electrons = sum(z) - charge
            occ_orbitals = electrons // 2  # For closed-shell systems
            for a in range(occ_orbitals):
                P_new[i, j] += 2 * C[i, a] * C[j, a]
    return P_new
    

def electronic_E(P, H, F):
    '''
    Electronic energy calculated wth the core Hamiltonian and the iteratively updated F and P matrices.
    '''
    E_elec = 0
    #print("P*H:", np.sum(P * H))
    #print("P*F:", np.sum(P * F))
    for i in range(nb):
        for j in range(nb):
                E_elec += P[i, j] * (H[i,j] + F[i,j])
    return 0.5 * E_elec

def scf(T, V, S, TEI, nb, charge, max_iterations=100, convergence_threshold=1e-6):
    '''
    SCF procedure
    First H_core, S_orthogonalized and initial P matrix are initialized. 
    E_old is stores the energy from previous iteration for the convergence criterion

    In the iteration itself:
    1) the F matrix is constructed using the P matrix
    2) MO coefficents are retrieved from the diagonalization
    3) Mo coefficients are used to construct new density matrix
    4) E_elec is computed and convergence is checked
    5) Rinse and repeat
    '''
    H = H_core(T, V)
    So = S_orthogonalized(S)
    P = init_P(nb)
    E_old = 0 #For storing the energy from previous iteration to check convergence

    for iteration in range(max_iterations):
        F = construct_F(H, P, TEI, nb)
        C, eigvals = diag_F(F, So)
        P_new = construct_P(C, nb, charge)
        E_elec = electronic_E(P_new, H, F)

        print(f"Iteration {iteration + 1}: E_elec = {E_elec}")

        if abs(E_elec - E_old) < convergence_threshold:
            print(f"SCF converged after {iteration + 1} iterations")
            break

        P = P_new
        E_old = E_elec
    else:
        print("SCF did not converge")

    return E_elec, P

def nuclear_repulsion_energy(z, coords):
    '''
    Nuclear repulsion energy term, surprisingly
    '''
    e_repulsion = 0.0
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            r_ij = np.linalg.norm(coords[i] - coords[j])
            e_repulsion += z[i] * z[j] / r_ij
    return e_repulsion

def energy_decomposition(P, T, V, TEI, z, coords, nb):
    '''
    Using the density matrix from the SCF to recalculate various expectation values.
    E_elec from the cycle is not used here. Instead the final density matrix is applied to obtain the 1e and 2e energies again. 
    This redundancy was left in the code on purpose, as an 'error check'.
    '''
    ET = np.sum(P * T) #1e kinetic energy
    EV = np.sum(P * V) #1e - nuclear attraction

    EJ = 0.0 #2e repulsion
    EK = 0.0 #2e exchange
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for l in range(nb):
                    EJ += P[i, j] * P[k, l] * TEI[i, j, k, l]
                    EK -= 0.5 * P[i, j] * P[k, l] * TEI[i, k, j, l]
    EJ = EJ/2
    EK = EK/2

    ENuc = nuclear_repulsion_energy(z, coords)

    ETot = ET + EV + EJ + EK + ENuc
    return ET, EV, EJ, EK, ENuc, ETot
    

#####################################################

# Reading input
natoms, charge, labels, z, coords = ReadInput('inputs/h2.input')
#natoms, charge, labels, z, coords = ReadInput('heh+.input')
basis_set = "cc-pvqz"

# Building basis and computing 1 and 2 electron integrals
nb, basis = BuildBasis(natoms, z, coords, basis_set, basis_directory="basis/")
S, T, V = compute_one_electron_integrals(natoms,nb,z,coords,basis)
TEI = compute_two_electron_integrals(nb,coords,basis)
 
# Main execution
E_elec, P = scf(T, V, S, TEI, nb, charge)
ET, EV, EJ, EK, ENuc, ETot = energy_decomposition(P, T, V, TEI, z, coords, nb)

print("###########")
print("E_elec:", E_elec, "\n-----------")
print("ET:", ET, "\nEV:", EV, "\nEJ:", EJ, "\nEK:", EK, "\nENuc:", ENuc, "\nE_Elec + ENuc:", E_elec + ENuc, "\n>>> E(RHF):", ETot," <<<")
