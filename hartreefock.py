import numpy as np
import scipy as sp
import math

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

def BuildBasis(nat,z,coords):
    bfno = 0
    basis = []
    for i in range(nat):
        if z[i] == 1:
            zets = [18.7311370,2.8253937,0.6401217]
            dijs = [0.2149354183,0.3645711272,0.4150514277]
            basis.append([bfno,i,3,zets,dijs])
            bfno += 1
            zets = [0.1612778000]
            dijs = [0.1813806839]
            basis.append([bfno,i,1,zets,dijs])
            bfno += 1

        if z[i] == 2:
            zets = [38.421634000,5.7780300000,1.2417740000]
            dijs = [0.4414855700,0.6938965807,0.6649918170]
            basis.append([bfno,i,3,zets,dijs])
            bfno += 1
            zets = [0.2979640000]
            dijs = [0.2874305587]
            basis.append([bfno,i,1,zets,dijs])
            bfno += 1
    return bfno,basis

def BoysFunc(r):
   xsmall = 1.e-6
   if r < xsmall:
      b = 1.
   else:
      b = 0.5 * np.sqrt(np.pi/r) * math.erf(np.sqrt(r))
   return b 

def cmpt1e(nat,nb,z,coords,basis):
   S = np.zeros((nb,nb))
   T = np.zeros((nb,nb))
   V = np.zeros((nb,nb))
   for nao1 in range(nb):
      for nao2 in range(nb):
         for prim1 in range(basis[nao1][2]):
            zetb=basis[nao1][3][prim1]
            dijb=basis[nao1][4][prim1]
            Ab=coords[basis[nao1][1],:]
            for prim2 in range(basis[nao2][2]):
               zetk=basis[nao2][3][prim2]
               dijk=basis[nao2][4][prim2]
               Ak=coords[basis[nao2][1],:]
               p=zetb+zetk
               q=zetb*zetk/p
               Abk = (zetb * Ab + zetk * Ak) / p
               RAB=Ak - Ab
               RAB2=np.power(np.linalg.norm(RAB),2)
               s00=dijb*dijk*np.power(np.pi/p,3/2)*np.exp(-q*RAB2)
               S[nao1,nao2]+=s00
               k00 = q*(3.-2.*q*RAB2)
               T[nao1,nao2]+=k00*s00
               vtmp = 0.
               for nuc in range(nat):
                  RPC2 = np.power(np.linalg.norm(coords[nuc,:]-Abk),2)
                  bfunc = BoysFunc(p * RPC2)
                  vtmp += bfunc * z[nuc]
               V[nao1,nao2] -= 2. * vtmp * np.sqrt(p/np.pi) * s00
   return S, T, V

def cmpt2e(nb,coords,basis):
   TEI = np.zeros((nb,nb,nb,nb))
   kfac = np.sqrt(2.) * np.power(np.pi,1.25)
   for i in range(nb):
      orbi=basis[i]
      Ai=coords[orbi[1],:]
      for j in range(i+1):
          orbj=basis[j]
          Aj=coords[orbj[1],:]
          rij2=np.power(np.linalg.norm(Aj-Ai),2)
          ij = i*(i+1)/2 + j
          for k in range(nb):
              orbk=basis[k]
              Ak=coords[orbk[1],:]
              for l in range(k+1):
                  kl = k*(k+1)/2 + l
                  if (ij < kl):
                     continue
                  orbl=basis[l]
                  Al=coords[orbl[1],:]
                  rkl2=np.power(np.linalg.norm(Al-Ak),2)
                  intval  = 0.
                  for ii in range(orbi[2]):
                     zeti=orbi[3][ii]
                     diji=orbi[4][ii]
                     for jj in range(orbj[2]):
                         zetj=orbj[3][jj]
                         dijj=orbj[4][jj]
                         pb = zeti + zetj
                         qb = zeti * zetj / pb
                         pXb = (Ai * zeti + Aj * zetj) / pb
                         kij = kfac * np.exp(-qb * rij2) / pb
                         for kk in range(orbk[2]):
                             zetk=orbk[3][kk]
                             dijk=orbk[4][kk]
                             for ll in range(orbl[2]):
                                zetl=orbl[3][ll]
                                dijl=orbl[4][ll]
                                pk = zetk + zetl
                                qk = zetk * zetl / pk
                                pXk = (Ak * zetk + Al * zetl) / pk
                                kkl = kfac * np.exp(-qk * rkl2) / pk
                                rpp2=np.power(np.linalg.norm(pXk-pXb),2)
                                rho = pb * pk / (pb + pk)
                                bfunc = BoysFunc(rpp2 * rho)
                                fac = bfunc * kij * kkl / np.sqrt(pb + pk)
                                intval += fac * diji * dijj * dijk * dijl
                  TEI[i,j,k,l] = intval
                  TEI[j,i,k,l]=TEI[i,j,k,l]
                  TEI[i,j,l,k]=TEI[i,j,k,l]
                  TEI[j,i,l,k]=TEI[i,j,k,l]
                  TEI[k,l,i,j]=TEI[i,j,k,l]
                  TEI[k,l,j,i]=TEI[i,j,k,l]
                  TEI[l,k,i,j]=TEI[i,j,k,l]
                  TEI[l,k,j,i]=TEI[i,j,k,l]
   return TEI

def H_core(T, V):
    return T + V

def S_orthogonalized(S):
    '''
    Orthogonalization of the overlap matrix.
    Why tho?
    '''
    from scipy.linalg import fractional_matrix_power
    return fractional_matrix_power(S, -0.5)

def init_P(nb):
    '''
    Initial guess for the density matrix:
    Just an zero matrix with the dimensions of the number of basis functions (nb)
    '''
    return np.zeros((nb, nb))

def construct_F(H, P, TEI, nb):
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
    E_elec = 0
    print("P*H:", np.sum(P * H))
    print("P*F:", np.sum(P * F))
    for i in range(nb):
        for j in range(nb):
                E_elec += P[i, j] * (H[i,j] + F[i,j])
    return E_elec

def scf(T, V, S, TEI, nb, charge, max_iter=100, convergence_threshold=1e-6):
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
    E_old = 0

    for iteration in range(max_iter):
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
    e_repulsion = 0.0
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            r_ij = np.linalg.norm(coords[i] - coords[j])
            e_repulsion += z[i] * z[j] / r_ij
    return e_repulsion

def energy_decomposition(P, T, V, TEI, z, coords, nb):
    ET = np.sum(P * T)
    EV = np.sum(P * V)
    EJ = 0.0
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for l in range(nb):
                    EJ += P[i, j] * P[k, l] * TEI[i, j, k, l]
    EJ = EJ/2
    EK = 0.0
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for l in range(nb):
                    EK -= P[i, j] * P[k, l] * TEI[i, k, j, l]
    EK = EK/2

    ENuc = nuclear_repulsion_energy(z, coords)

    ETot = ET + EV + EJ + EK + ENuc
    return ET, EV, EJ, EK, ENuc, ETot
    

#####################################################

natoms, charge, labels, z, coords = ReadInput('/home/danb/hartree_fock/h2.input')
#natoms, charge, labels, z, coords = ReadInput('heh+.input')
nb, basis = BuildBasis(natoms,z,coords)
S, T, V = cmpt1e(natoms,nb,z,coords,basis)
TEI = cmpt2e(nb,coords,basis)

#print("Overlap integrals")
#print(S)
#print("Kinetic Energy integrals")
#print(T)
#print("Nuclear-Electronic Energy integrals")
#print(V)
#print("Two-electron Integrals")
#print(TEI)

 
# Main execution
H = H_core(T, V)
E_elec, P = scf(T, V, S, TEI, nb, charge)
ET, EV, EJ, EK, ENuc, ETot = energy_decomposition(P, T, V, TEI, z, coords, nb)

print("###########")
print("E_elec:", E_elec, "\n-----------")
print("ET:", ET, "\nEV:", EV, "\nEJ:", EJ, "\nEK:", EK, "\nENuc:", ENuc, "\n>>> E(RHF):", ETot," <<<")
print(E_elec + ENuc)
