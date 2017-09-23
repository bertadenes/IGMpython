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
    for f in fragcubes:
        atomcount += f.Natom
        if f.dens.shape != fullcube.dens.shape:
            print("Some fragments are represented on a different grid!")
            sys.exit(0)
        for atom in f.atoms:
            if atom not in fullcube.atoms:
                print("Some fragments have been translated or rotated!")
                sys.exit(0)
    if atomcount != fullcube.Natom:
        print("Total number of atoms in the subsystems is not equal to the full system!")
        sys.exit(0)
    return


def vmd_write():
    """
    Writes VMD visualization file.
    :return: 
    """
    try:
        template1 = """#!/usr/local/bin/vmd
# VMD script written by save_state $Revision: 1.41 $
# VMD version: 1.8.6
set viewplist
set fixedlist
# Display settings
display projection   Orthographic
display nearclip set 0.000000
# load new molecule
mol new {c[densfile]}        type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
mol addfile {c[mideigfile]}        type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
#
# representation of the atoms
mol delrep 0 top
mol representation Lines 1.00000
mol color Name"""
#mol selection \{all\}
        template2 = """
mol material Opaque
mol addrep top
mol representation CPK 1.000000 0.300000  118.000000 131.000000
mol color Name"""
#mol selection \{index     0 to    92 \}
        template3 = """
mol material Opaque
mol addrep top
#
# add representation of the surface
mol representation Isosurface 0.30000 1 0 0 1 1
mol color Volume 0"""
#mol selection \{all\}
        template4 = """
mol material Opaque
mol addrep top
mol selupdate 2 top 0
mol colupdate 2 top 0
mol scaleminmax top 2 {c[scalemin]} {c[scalemax]}
mol smoothrep top 2 0"""
#mol drawframes top 2 \{now\}
#color scale method BGR
#set colorcmds \{\{color Name \{C\} gray\}\}

        context = {
        "mideigfile":"mideig.cub",
        "densfile":"igm.cub",
        "scalemin":0.0,
        "scalemax":1.0
        }

        with open("IGMpython.vmd", "w") as vmdfile:
            vmdfile.write(template1.format(c=context))
            vmdfile.write("mol selection \{all\}")
            vmdfile.write(template2.format(c=context))
            vmdfile.write("mol selection \{index     0 to    92 \}")
            vmdfile.write(template3.format(c=context))
            vmdfile.write("mol selection \{all\}")
            vmdfile.write(template4.format(c=context))
            vmdfile.write("mol drawframes top 2 \{now\}\n")
            vmdfile.write("color scale method BGR\n")
            vmdfile.write("set colorcmds \{\{color Name \{C\} gray\}\}\n")

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
    checkfragmentation(fullcube, fragcubes)
    t3 = time.time()
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
    IGMcube = copy.deepcopy(fullcube)
    IGMcube.grad = np.subtract(np.absolute(IGMgrad),np.absolute(fullcube.grad))
    IGMcube.dens = np.subtract(np.linalg.norm(IGMgrad,axis=3),np.linalg.norm(fullcube.grad,axis=3))
    
    t5 = time.time()
    fullcube.gethessian()
    t6 = time.time()
    fullcube.getmideig()
    t7 = time.time()

    fullcube.write("mideig.cub",t="mideig")
    IGMcube.write("igm.cub",t="dens")
    t8 = time.time()
    vmd_write()
    t9 = time.time()
    print("Reading cubes:           %s seconds." % (t2 - t1))
    print("Checking cubes:          %s seconds." % (t3 - t2))
    print("Calculating gradients:   %s seconds." % (t4 - t3))
    print("Setting up IGM:          %s seconds." % (t5 - t4))
    print("Calculating Hessian:     %s seconds." % (t6 - t5))
    print("Calculating eigenvalues: %s seconds." % (t7 - t6))
    print("Writing cubes:           %s seconds." % (t8 - t7))
    print("Writing vmd state:       %s seconds." % (t9 - t8))
    return

if __name__ == "__main__":
    main()