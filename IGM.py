#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
from multiprocessing import Pool
from contextlib import closing
import time
import copy

class cube(object):
    """docstring for cube"""
    def __init__(self):
        super(cube, self).__init__()
        self.Natom = 0
        self.atoms = []
        self.origin = None
        self.xgrid = None
        self.ygrid = None
        self.zgrid = None
        Nx = 0
        Ny = 0
        Nz = 0
        dens = None

    def read(self, filename):
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
        try:
            with open(filename, "w") as f:
                if t == "dens" or t == "mideig":
                    values = 1
                elif t == "grad":
                    values = 4
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
        """Hessian using the calculated gradient."""
        self.hess = np.zeros(shape=(self.Nx,self.Ny,self.Nz,3,3),dtype=np.float64)
        self.hess[:,:,:,0,0] = np.gradient(self.grad[:,:,:,0], np.float64(self.xgrid[1]), axis=0) # 
        self.hess[:,:,:,1,1] = np.gradient(self.grad[:,:,:,1], np.float64(self.ygrid[2]), axis=1) # 
        self.hess[:,:,:,2,2] = np.gradient(self.grad[:,:,:,2], np.float64(self.zgrid[3]), axis=2) # 
        self.hess[:,:,:,0,1] = self.hess[:,:,:,1,0] = np.gradient(self.grad[:,:,:,0], np.float64(self.ygrid[2]), axis=1) # 
        self.hess[:,:,:,0,2] = self.hess[:,:,:,2,0] = np.gradient(self.grad[:,:,:,0], np.float64(self.zgrid[3]), axis=2) # 
        self.hess[:,:,:,1,2] = self.hess[:,:,:,2,1] = np.gradient(self.grad[:,:,:,1], np.float64(self.zgrid[3]), axis=2) # 
        #print(self.hess[40,40,40,:,:])
        return
    
    def getmideig(self):
        if not hasattr(self,"hess"):
            self.gethessian()
        self.mideig = np.zeros(shape=self.dens.shape,dtype=np.float64)
        for x in range(self.Nx):
            for y in range(self.Ny):
                for z in range(self.Nz):
                    self.mideig[x,y,z] = np.linalg.eigvalsh(self.hess[x,y,z])[1]
        return
    
def checkfragmentation(fullcube, fragcubes):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('frags',nargs="+")
    args = parser.parse_args()
    
    t1 = time.time()
    fullcube = cube()
    fullcube.read(args.file)
    fragcubes = []
    for f in args.frags:
        fragcubes.append(cube())
        fragcubes[-1].read(f)
    t2 = time.time()
    checkfragmentation(fullcube,fragcubes)
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
    #for x in fullcube.dens:
    #    print(x[0,0])
    #for x in fullcube.grad:
    #    print(x[0][0][0])
    #for f in fragcubes:
    #    for x in f.grad:
    #        print(x[0][0][0])
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
    print("Reading cubes:           %s seconds." % (t2 - t1))
    print("Checking cubes:          %s seconds." % (t3 - t2))
    print("Calculating gradients:   %s seconds." % (t4 - t3))
    print("Setting up IGM:          %s seconds." % (t5 - t4))
    print("Calculating Hessian:     %s seconds." % (t6 - t5))
    print("Calculating eigenvalues: %s seconds." % (t7 - t6))
    print("Writing cubes:           %s seconds." % (t8 - t7))
    return

if __name__ == "__main__":
    main()