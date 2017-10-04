#!/usr/bin/env python3

import argparse
import copy
import os
import sys
import time
import itertools
import multiprocessing
from contextlib import closing
from functools import partial

import numpy as np

class Cube(object):
    """
    Cube object to store volumetric data.
    """
    def __init__(self):
        """
        Initialize cube object.
        :rtype: object
        """
        self.grad = None
        self.Natom = 0
        self.atoms = []
        self.origin = None
        self.xgrid = None
        self.ygrid = None
        self.zgrid = None
        self.Nx = 0
        self.Ny = 0
        self.Nz = 0
        self.dens = None
        self.mideig = None
        self.hess = None

    def read(self, filename):
        """
        Reads a cube file into the cube object.
        :param filename: (str) path to cube file
        :return: 
        """
        if not os.path.isfile(filename):
            print(filename,"not file.")
            sys.exit(0)
        try:
            with open(filename, "r") as f:
                f.readline() #comment
                f.readline() #density title
                head = f.readline().split()
                self.Natom = int(head[0])
                self.origin = np.array(head[1:4], dtype=np.float64)
                self.xgrid = f.readline().split()
                self.ygrid = f.readline().split()
                self.zgrid = f.readline().split()
                self.Nx = int(self.xgrid[0])
                self.Ny = int(self.ygrid[0])
                self.Nz = int(self.zgrid[0])
                self.dens = np.zeros(shape=(self.Nx,self.Ny,self.Nz),dtype=np.float64)
                self.grad = np.zeros(shape=(self.Nx,self.Ny,self.Nz,3),dtype=np.float64)
                for i in range(self.Natom):
                    self.atoms.append(f.readline().split())
                ix = 0
                iy = 0
                iz = 0
                while True:
                    l = f.readline().split()
                    if len(l) == 1 and l[0] == "":
                        break
                    for v in l:
                        self.dens[ix][iy][iz] = v
                        iz += 1
                    if iz == self.Nz:
                        iz = 0
                        iy += 1
                    if iy == self.Ny:
                        iz = 0
                        iy = 0
                        ix += 1
                    if ix == self.Nx:
                        break
        except IOError:
            print(filename,"cannot be read.")
            sys.exit(0)
        #print(self.dens[0][0][0],self.dens[1][0][0],self.dens[0][1][0],self.dens[0][0][1],self.dens[-1][-1][-1])
        return

    def readXYZ(self, filename, offset=3):
        """
        Calculating the cube size: (xyz dim cube, g is the grid node distance, o is the offset in each dirrection, the
        default would be 80*80*80, a,b,c is the dimensions of the molecule)
        x*y*z = 512000
        x*g = a + o
        y*g = b + o
        z*g = c + o
        x*y*z*g**3 = (a+o)(b+o)(c+o)
        g = ((a+o)(b+o)(c+o)/512000)**1/3
        x = (a+o)/g
        :param filename: 
        :return: 
        """
        elements = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11,
                    'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21,
                    'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
                    'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41,
                    'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51,
                    'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
                    'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
                    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81,
                    'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
                    'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
                    'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
                    'Rg': 111, 'Cn': 112}
        if not os.path.isfile(filename):
            print(filename,"not file.")
            sys.exit(0)
        try:
            with open(filename, "r") as f:
                self.Natom = int(f.readline().split()[0])
                f.readline() #comment
                for i in range(self.Natom):
                    line = f.readline().split()
                    self.atoms.append([elements[line[0]], str(elements[line[0]])+'.00000', 1.8897259886*np.float64(line[1]),
                                       1.8897259886 * np.float64(line[2]), 1.8897259886*np.float64(line[3])])
                min = np.min(np.array(self.atoms, dtype=np.float64), axis=0)[2:]
                max = np.max(np.array(self.atoms, dtype=np.float64), axis=0)[2:]
                dim = max - min
                grid = np.cbrt(np.prod(dim + 2*offset) / 512000)
                self.Nx = int((dim[0] + 2*offset) / grid)
                self.Ny = int((dim[1] + 2*offset) / grid)
                self.Nz = int((dim[2] + 2*offset) / grid)
                self.grad = np.zeros(shape=(self.Nx,self.Ny,self.Nz,3), dtype=np.float64)
                self.dens = np.zeros(shape=(self.Nx,self.Ny,self.Nz), dtype=np.float64)
                self.xgrid = [self.Nx, grid, 0.0, 0.0]
                self.ygrid = [self.Ny, 0.0, grid, 0.0]
                self.zgrid = [self.Nz, 0.0, 0.0, grid]
                self.origin = min - offset
        except IOError:
            print(filename,"cannot be read.")
            sys.exit(0)
        return

    def write(self, filename, t="grad"):
        """
        Writes chosen volumetric data to a Gaussian-type cube file.
        :param filename: (str) path to cube file
        :param t: (str) type of data to write
        :return: 
        """
        try:
            with open(filename, "w") as f:
                if t == "dens" or t == "mideig":
                    values = 1
                elif t == "grad":
                    values = 4
                else:
                    print("Unknown type of cube.")
                    return
                f.write("Cube written by IGM.py\n")
                f.write("Electron density from Total SCF Density\n")
                f.write(str(self.Natom)+"  ")
                for i in self.origin:
                    f.write("%.6f  " % float(i))
                f.write(str(values)+"\n")
                f.write("%d  %f  %f  %f\n" % (self.Nx,float(self.xgrid[1]),float(self.xgrid[2]),float(self.xgrid[3])))
                f.write("%d  %f  %f  %f\n" % (self.Ny,float(self.ygrid[1]),float(self.ygrid[2]),float(self.ygrid[3])))
                f.write("%d  %f  %f  %f\n" % (self.Nz,float(self.zgrid[1]),float(self.zgrid[2]),float(self.zgrid[3])))
                for a in self.atoms:
                    for s in a:
                        f.write(str(s)+"  ")
                    f.write("\n")
                count = 0
                for ix in range(self.Nx):
                    for iy in range(self.Ny):
                        for iz in range(self.Nz):
                            if t == "dens" or t == "grad":
                                f.write("%.5E" % self.dens[ix][iy][iz])
                            elif t == "mideig":
                                f.write("%.5E" % self.mideig[ix][iy][iz])
                            count += 1
                            if count == 6:
                                f.write("\n")
                                count = 0
                            else:
                                f.write("  ")
                            if t == "grad":
                                for g in self.grad[ix][iy][iz]:
                                    f.write("%.5E" % g)
                                    count += 1
                                    if count == 6:
                                        f.write("\n")
                                        count = 0
                                    else:
                                        f.write("  ")
                        if count != 0:
                            f.write("\n")
                            count = 0
                    if count != 0:
                        f.write("\n")
                        count = 0
        except IOError:
            print(filename,"cannot be written.")
            sys.exit(0)
        return
    
    def gethessian(self):
        """Hessian using the calculated gradient.
        :return: 
        """
        self.hess = np.zeros(shape=(self.Nx, self.Ny, self.Nz, 3, 3), dtype=np.float64)
        self.hess[:,:,:,0,0] = np.gradient(self.grad[:,:,:,0], np.float64(self.xgrid[1]), axis=0) #
        self.hess[:,:,:,1,1] = np.gradient(self.grad[:,:,:,1], np.float64(self.ygrid[2]), axis=1) # 
        self.hess[:,:,:,2,2] = np.gradient(self.grad[:,:,:,2], np.float64(self.zgrid[3]), axis=2) # 
        self.hess[:,:,:,0,1] = self.hess[:,:,:,1,0] = np.gradient(self.grad[:,:,:,0], np.float64(self.ygrid[2]), axis=1) # 
        self.hess[:,:,:,0,2] = self.hess[:,:,:,2,0] = np.gradient(self.grad[:,:,:,0], np.float64(self.zgrid[3]), axis=2) # 
        self.hess[:,:,:,1,2] = self.hess[:,:,:,2,1] = np.gradient(self.grad[:,:,:,1], np.float64(self.zgrid[3]), axis=2) # 
        #print(self.hess[40,40,40,:,:])
        return

    def diag(self, params):
        m = self.hess[params[0],params[1],params[2],:,:]
        return np.linalg.eigvalsh(m)[1]

    def getmideig(self):
        """
        Calculate middle eigenvalue at each point.
        :return: 
        """
        self.mideig = np.zeros(shape=self.dens.shape, dtype=np.float64)
        if not hasattr(self,"hess"):
            self.gethessian()
        paramlist = list(itertools.product(range(self.Nx), range(self.Ny), range(self.Nz)))
        with closing(multiprocessing.Pool(processes=4)) as pool:
            ev = pool.map(self.diag, paramlist)
            pool.terminate()
        for i in range(len(paramlist)):
            self.mideig[paramlist[i]] = ev[i]
        return

    def getmideig_np(self):
        if not hasattr(self, "hess"):
            self.gethessian()
        self.mideig = np.zeros(shape=self.dens.shape, dtype=np.float64)
        for x in range(self.Nx):
            for y in range(self.Ny):
                for z in range(self.Nz):
                    self.mideig[x, y, z] = np.linalg.eigvalsh(self.hess[x, y, z])[1]
        return


def checkfragmentation(fullcube, fragcubes):
    """
    Checks if the fragmentation is correct for calculation.
    :param fullcube: Cube object defining full system and density.
    :param fragcubes: (list) of Cube objects of fragments.
    :return: 
    """
    atomcount = 0
    atomsinfrags = []
    for f in fragcubes:
        atomcount += f.Natom
        if f.dens.shape != fullcube.dens.shape:
            print("Some fragments are represented on a different grid!")
            sys.exit(0)
        for atom in f.atoms:
            for atom2 in fullcube.atoms:
                if np.linalg.norm(np.subtract(np.array(atom, dtype=np.float64), np.array(atom2, dtype=np.float64))) < 0.00001:
                    atomsinfrags.append(fullcube.atoms.index(atom2))
                    break
    if atomcount != fullcube.Natom:
        print("Total number of atoms in the subsystems is not equal to the full system!")
        print("Please provide cubes of all fragments in the same level of theory, or use promolecular densities by specifying flag -p/--promol.")
        sys.exit(0)
    if len(atomsinfrags) != fullcube.Natom:
        print("Some fragments have been translated or rotated!")
        sys.exit(0)
    return

def createatomfrag(atom, fc):
    a1 = np.array(
        [0.2815, 2.437, 11.84, 31.34, 67.82, 120.2, 190.9, 289.5, 406.3, 561.3, 760.8, 1016., 1319., 1658., 2042.,
         2501., 3024., 3625.], dtype=np.float64)
    a2 = np.array(
        [0.0, 0.0, 0.06332, 0.3694, 0.8527, 1.172, 2.247, 2.879, 3.049, 6.984, 22.42, 37.17, 57.95, 87.16, 115.7, 158.0,
         205.5, 260.0], dtype=np.float64)
    a3 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06358, 0.3331, 0.8878, 0.7888, 1.465, 2.170, 3.369, 5.211],
        dtype=np.float64)
    b1 = np.array(
        [1.8910741301059, 2.95945546019532, 5.23012552301255, 7.19424460431655, 9.44287063267233, 11.3122171945701,
         13.0378096479791, 14.9476831091181, 16.4473684210526, 18.2149362477231, 20.1612903225806, 22.271714922049,
         24.330900243309, 26.1780104712042, 27.9329608938547, 29.8507462686567, 31.7460317460318, 33.7837837837838],
        dtype=np.float64)
    b2 = np.array(
        [1.000000, 1.000000, 1.00080064051241, 1.43988480921526, 1.88679245283019, 1.82481751824818, 2.20653133274492,
         2.51635631605435, 2.50375563345018, 2.90107339715695, 3.98247710075667, 4.65116279069767, 5.33617929562433,
         6.0459492140266, 6.62690523525514, 7.30460189919649, 7.9428117553614, 8.56164383561644], dtype=np.float64)
    b3 = np.array([1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                   0.9769441187964, 1.28982329420869, 1.67728950016773, 1.42959256611866, 1.70910955392241,
                   1.94212468440474, 2.01045436268597, 2.26654578422484], dtype=np.float64)
    frags = []
    print("Creating cube for atom",atom)
    atomfrag = Cube()
    atomfrag.Natom = 1
    atomfrag.atoms = [atom]
    atomfrag.origin = fc.origin
    atomfrag.xgrid = fc.xgrid
    atomfrag.ygrid = fc.ygrid
    atomfrag.zgrid = fc.zgrid
    atomfrag.Nx = fc.Nx
    atomfrag.Ny = fc.Ny
    atomfrag.Nz = fc.Nz
    atomfrag.dens = np.zeros(shape=fc.dens.shape, dtype=np.float64)
    atomfrag.grad = np.zeros(shape=(atomfrag.Nx, atomfrag.Ny, atomfrag.Nz, 3), dtype=np.float64)
    for x in range(atomfrag.Nx):
        for y in range(atomfrag.Ny):
            for z in range(atomfrag.Nz):
                r = np.linalg.norm(np.array([np.float64(atom[2]) - (atomfrag.origin[0] + np.float64(atomfrag.xgrid[1]) * x),
                                             np.float64(atom[3]) - (atomfrag.origin[1] + np.float64(atomfrag.ygrid[2]) * y),
                                             np.float64(atom[4]) - (atomfrag.origin[2] + np.float64(atomfrag.zgrid[3]) * z)], dtype=np.float64))
                atomindex = int(atom[0]) - 1
                atomfrag.dens[x, y, z] = a1[atomindex] * np.exp(-b1[atomindex] * r) + a2[atomindex] * np.exp(-b2[atomindex] * r) + a3[atomindex] * np.exp(-b3[atomindex] * r)
    frags.append(atomfrag)
    print("Done")
    return atomfrag

def vmd_write(Natom, dd=False):
    """
    Writes VMD visualization file.
    :return: 
    """
    try:
        template1 = """#!/usr/local/bin/vmd
# VMD script written by save_state $Revision: 1.41 $
# VMD version: 1.8.6
#set viewplist
#set fixedlist
# Display settings
display projection   Orthographic
display nearclip set 0.000000
# load new molecule
mol new {c[densfile]} type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
mol addfile {c[colourfile]} type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
#
# representation of the atoms
mol delrep 0 top
mol representation Lines 1.00000
mol color Name
"""
#mol selection \{all\}
        template2 = """
mol material Opaque
mol addrep top
mol representation CPK 0.500000 0.300000  118.000000 131.000000
mol color Name
"""
#mol selection \{index     0 to    Natom \}
        template3 = """
mol material Opaque
mol addrep top
#
# add representation of the surface
mol representation Isosurface 0.30000 0 0 0 1 1
mol color Volume 1
"""
#mol selection \{all\}
        template4 = """
mol material Opaque
mol addrep top
mol selupdate 2 top 0
mol colupdate 2 top 0
mol scaleminmax top 2 {c[scalemin]} {c[scalemax]}
mol smoothrep top 2 0
"""
#mol drawframes top 2 \{now\}
#color scale method BGR
#set colorcmds \{\{color Name \{C\} gray\}\}

        context = {
        "colourfile":"mideig.cub",
        "densfile":"igm.cub",
        "scalemin":-2.0,
        "scalemax":2.0
        }
        if dd:
            context["colourfile"] = "diff.cub"

        with open("IGMpython.vmd", "w") as vmdfile:
            vmdfile.write(template1.format(c=context))
            vmdfile.write("mol selection {all}")
            vmdfile.write(template2.format(c=context))
            vmdfile.write("mol selection {index     0 to    "+str(Natom)+" }\n")
            vmdfile.write(template3.format(c=context))
            vmdfile.write("mol selection {all}")
            vmdfile.write(template4.format(c=context))
            vmdfile.write("mol drawframes top 2 {now}\n")
            vmdfile.write("color scale method BGR\n")
            vmdfile.write("set colorcmds {{color Name {C} gray}}\n")
            vmdfile.write("axes location off\n")
            vmdfile.write("color Display Background white\n")

    except IOError:
         print("IGMpython.vmd cannot be written.")
         sys.exit(0)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-f', '--frags', nargs="+", default=[])
    parser.add_argument('-p', '--promol', action='store_true', default=False, help='Use promolecular densities.')
    parser.add_argument('-d', '--densdiff', action='store_true', default=False, help='Use density difference for'
                                                                                     'colouring.')
    args = parser.parse_args()

    fullcube = Cube()
    fragcubes = []
    if not args.promol:
        t1 = time.time()
        fullcube.read(args.file)
        for f in args.frags:
            fragcubes.append(Cube())
            fragcubes[-1].read(f)
        if args.densdiff:
            Diffcube = copy.deepcopy(fullcube)
        t2 = time.time()
        print("Reading cubes:           %s seconds." % (t2 - t1))
        checkfragmentation(fullcube, fragcubes)
        t3 = time.time()
        print("Checking cubes:          %s seconds." % (t3 - t2))
    else:
        t1 = time.time()
        fullcube.readXYZ(args.file)
        crtatom = partial(createatomfrag, fc=fullcube)
        with closing(multiprocessing.Pool(processes=4)) as pool:
            fragcubes = pool.map(crtatom, fullcube.atoms)
            pool.terminate()

        for f in fragcubes:
            fullcube.dens = np.add(fullcube.dens, f.dens)
        t2 = time.time()
        print("Creating cubes:          %s seconds." % (t2 - t1))
        t3 = time.time()

    IGMgrad = np.zeros(shape=(fullcube.Nx,fullcube.Ny,fullcube.Nz,3), dtype=np.float64)
    #fullcube.dens = np.zeros(shape=(fullcube.Nx,fullcube.Ny,fullcube.Nz), dtype=np.float64)
    for f in fragcubes:
        if not args.promol and args.densdiff:
            Diffcube.dens = np.subtract(Diffcube.dens, f.dens)
        f.grad[:,:,:,0] = np.gradient(f.dens,np.float64(f.xgrid[1]), axis=0)
        f.grad[:,:,:,1] = np.gradient(f.dens,np.float64(f.ygrid[2]), axis=1)
        f.grad[:,:,:,2] = np.gradient(f.dens,np.float64(f.zgrid[3]), axis=2)
        IGMgrad = np.add(IGMgrad,np.absolute(f.grad))
    #with closing(Pool(processes=2)) as pool:
    #    IGMgrad[:,:,:,0] = np.gradient(fragsum,np.float64(fullcube.xgrid[1]),axis=0)
    #    pool.terminate()
    fullcube.grad[:,:,:,0] = np.gradient(fullcube.dens,np.float64(fullcube.xgrid[1]), axis=0)
    fullcube.grad[:,:,:,1] = np.gradient(fullcube.dens,np.float64(fullcube.ygrid[2]), axis=1)
    fullcube.grad[:,:,:,2] = np.gradient(fullcube.dens,np.float64(fullcube.zgrid[3]), axis=2)
    t4 = time.time()
    print("Calculating gradients:   %s seconds." % (t4 - t3))
    IGMcube = copy.deepcopy(fullcube)
    IGMcube.dens = np.subtract(np.linalg.norm(IGMgrad,axis=3),np.linalg.norm(fullcube.grad, axis=3))
    IGMcube.grad = np.subtract(np.absolute(IGMgrad),np.absolute(fullcube.grad))

    t5 = time.time()
    print("Setting up IGM:          %s seconds." % (t5 - t4))
    if not args.densdiff:
        fullcube.gethessian()
        t6 = time.time()
        print("Calculating Hessian:     %s seconds." % (t6 - t5))
        fullcube.getmideig()
        t7 = time.time()
        print("Calculating eigenvalues: %s seconds." % (t7 - t6))
        
    IGMcube.write("igm.cub",t="dens")
    if args.densdiff:
        Diffcube.write("diff.cub", t="dens")
        t8 = time.time()
        print("Writing cubes:           %s seconds." % (t8 - t5))
    else:
        fullcube.write("mideig.cub",t="mideig")
        t8 = time.time()
        print("Writing cubes:           %s seconds." % (t8 - t7))
    vmd_write(fullcube.Natom, args.densdiff)
    t9 = time.time()
    print("Writing vmd state:       %s seconds." % (t9 - t8))
    return

if __name__ == "__main__":
    main()