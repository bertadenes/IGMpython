#!/usr/bin/env python3

import argparse
import copy
import os
import sys
import time

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
                self.origin = np.array(head[1:4])
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
                        f.write(s+"  ")
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
    
    def getmideig(self):
        """
        Calculate middle eigenvalue at each point.
        :return: 
        """
        self.mideig = np.zeros(shape=self.dens.shape, dtype=np.float64)
        if not hasattr(self,"hess"):
            self.gethessian()
        for x in range(self.Nx):
            for y in range(self.Ny):
                for z in range(self.Nz):
                    self.mideig[x,y,z] = np.linalg.eigvalsh(self.hess[x,y,z])[1]
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
            if atom in fullcube.atoms:
                atomsinfrags.append(atom)
            # else:
                # print("Some fragments have been translated or rotated!")
                # sys.exit(0)
    if atomcount != fullcube.Natom:
        print("Total number of atoms in the subsystems is not equal to the full system!")
        print("Non-listed atoms will be treated as separate fragments.")
        for atom in fullcube.atoms:
            if atom not in atomsinfrags:
                fragcubes.append(createatomfrag(fullcube, atom))
    return

def createatomfrag(fullcube, atom):
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
    atomfrag = Cube()
    atomfrag.Natom = 1
    atomfrag.atoms = [atom]
    atomfrag.origin = fullcube.origin
    atomfrag.xgrid = fullcube.xgrid
    atomfrag.ygrid = fullcube.ygrid
    atomfrag.zgrid = fullcube.zgrid
    atomfrag.Nx = fullcube.Nx
    atomfrag.Ny = fullcube.Ny
    atomfrag.Nz = fullcube.Nz
    atomfrag.dens = np.zeros(shape=fullcube.dens.shape, dtype=np.float64)
    for x in range(atomfrag.Nx):
        for y in range(atomfrag.Ny):
            for z in range(atomfrag.Nz):
                r = np.linalg.norm([atom[2] - (atomfrag.origin[0] + atomfrag.xgrid[1] * x), atom[3] - (atomfrag.origin[1] + atomfrag.ygrid[2] * y), atom[4] - (atomfrag.origin[2] + atomfrag.zgrid[3] * z)])
                atomfrag.dens[x,y,z] = a1[atom[0]-1] * np.exp(-b1[atom[0]] * r) + a2[atom[0]-1] * np.exp(-b2[atom[0]] * r) + a3[atom[0]-1] * np.exp(-b3[atom[0]] * r)
    return atomfrag

def vmd_write(Natom):
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
mol addfile {c[mideigfile]} type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
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
mol representation CPK 1.000000 0.300000  118.000000 131.000000
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
        "mideigfile":"mideig.cub",
        "densfile":"igm.cub",
        "scalemin":-2.0,
        "scalemax":2.0
        }

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
    parser.add_argument('frags',nargs="+")
    args = parser.parse_args()

    t1 = time.time()
    fullcube = Cube()
    fullcube.read(args.file)
    fragcubes = []
    for f in args.frags:
        fragcubes.append(Cube())
        fragcubes[-1].read(f)
    t2 = time.time()
    print("Reading cubes:           %s seconds." % (t2 - t1))
    checkfragmentation(fullcube, fragcubes)
    t3 = time.time()
    print("Checking cubes:          %s seconds." % (t3 - t2))
    IGMgrad = np.zeros(shape=(fullcube.Nx,fullcube.Ny,fullcube.Nz,3),dtype=np.float64)
    for f in fragcubes:
        f.grad[:,:,:,0] = np.gradient(f.dens,np.float64(f.xgrid[1]),axis=0)
        f.grad[:,:,:,1] = np.gradient(f.dens,np.float64(f.ygrid[2]),axis=1)
        f.grad[:,:,:,2] = np.gradient(f.dens,np.float64(f.zgrid[3]),axis=2)
        IGMgrad = np.add(IGMgrad,np.absolute(f.grad))
    #with closing(Pool(processes=2)) as pool:
    #    IGMgrad[:,:,:,0] = np.gradient(fragsum,np.float64(fullcube.xgrid[1]),axis=0)
    #    pool.terminate()
    fullcube.grad[:,:,:,0] = np.gradient(fullcube.dens,np.float64(fullcube.xgrid[1]),axis=0)
    fullcube.grad[:,:,:,1] = np.gradient(fullcube.dens,np.float64(fullcube.ygrid[2]),axis=1)
    fullcube.grad[:,:,:,2] = np.gradient(fullcube.dens,np.float64(fullcube.zgrid[3]),axis=2)
    t4 = time.time()
    print("Calculating gradients:   %s seconds." % (t4 - t3))
    IGMcube = copy.deepcopy(fullcube)
    IGMcube.grad = np.subtract(np.absolute(IGMgrad),np.absolute(fullcube.grad))
    IGMcube.dens = np.subtract(np.linalg.norm(IGMgrad,axis=3),np.linalg.norm(fullcube.grad,axis=3))

    t5 = time.time()
    print("Setting up IGM:          %s seconds." % (t5 - t4))
    fullcube.gethessian()
    t6 = time.time()
    print("Calculating Hessian:     %s seconds." % (t6 - t5))
    fullcube.getmideig()
    t7 = time.time()
    print("Calculating eigenvalues: %s seconds." % (t7 - t6))
    fullcube.write("mideig.cub",t="mideig")
    IGMcube.write("igm.cub",t="dens")
    t8 = time.time()
    print("Writing cubes:           %s seconds." % (t8 - t7))
    vmd_write(fullcube.Natom)
    t9 = time.time()
    print("Writing vmd state:       %s seconds." % (t9 - t8))
    return

if __name__ == "__main__":
    main()