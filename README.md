# IGMpython
*D. Berta, D. Ferenc, T. Foldes, A. Hamza*

This program is an improved implementation of the work of [Hénon et al.](http://dx.doi.org/10.1039/c7cp02110k).
IGMpython uses **molecular density** from ab initio calculations, given as a cube file.

## Requirements
The program utilizes python 3 interpreter and depends on numpy.
Gaussian cube files are taken as input. Available and future
features are detailed below.

## Installation

Download this repository and note where you decide to store it. We suggest to create a
symbolic link to the script: 
```commandline
ln -s /wherever/this/program/is/IGM.py $HOME/bin/IGM.py
```
Make sure to use the absolute path, otherwise the link might be broken.

After reloading the environment, the program should be ready to use.
```commandline
. $HOME/.bashrc && . $HOME/.bash_profile
```

## Usage

IGMpython takes on positional and several optional arguments:
```commandline
IGM.py full.cube [fragment.cubes ...]
```
At the moment, the program can only be used with given fragment cubes. The fragment
densities needed to be represented on the same grid. This can be easily managed with
[cubegen](http://gaussian.com/cubegen/) utility.

The implementation of atomic densities us default fragments (similarly to
[IGMplot](http://kisthelp.univ-reims.fr/igmplot/)) is under way.
## Output

Two cubes are generated as output: igm.cub is the defined gradient, mideig.cub is the
second eigenvalue of the density Hessian at each point on the same grid for colouring
purposes.

The visualization can be done by [VMD](http://www.ks.uiuc.edu/Research/vmd/), plotting
igm.cub as isosurface and colouring by the volume of mideig.cub.

A VMD visualization state file is automatically generated to aid the process. (not yet tested)

### Example
The example inputs and outputs are provided for the case of benzene dimer.

## Further remarks

Feel free to address problems to [Denes Berta](mailto:berta.denes@ttk.mta.hu).
