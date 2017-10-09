# IGMpython
*D. Berta, D. Ferenc, T. Földes, A. Hamza*

*Thanks to Imre Bakó and Imre Pápai for theoretical support.*

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

IGMpython takes one positional and several optional arguments:
```commandline
IGM.py full.cube -f [fragment.cubes ...]
```
Using the QM density all fragments, even if atomic fragments are considered, needs to
be calculated at the same level of theory to get meaningful results. The fragment
densities needed to be represented on the same grid. This can be easily managed with
[cubegen](http://gaussian.com/cubegen/) utility.

We plan to provide a tool to generate the necessary inputs for case where fragmentation
is not straightforward.

By default, a second cube consisting of the second eigenvalues of the Hessian of the
density is created for colouring, as it was proposed in
[NCIplot](http://www.lct.jussieu.fr/pagesperso/contrera/nciplot.html).
Alternatively, we provide another cube of the electron density difference by request:
```commandline
IGM.py full.cube -f [fragment.cubes ...] -dd
```
The cubes and VMD state is generated accordingly.

The implementation of atomic densities as default fragments (similarly to
[IGMplot](http://kisthelp.univ-reims.fr/igmplot/)) is also available, although IGMpython
does not exploit the analytical differentiability of the simple exponential functions. The
atomic densities were take from [IGMplot](http://kisthelp.univ-reims.fr/igmplot/)).
 
In order to use this approach, IGMpython takes an xyz file as input, and the -p flag
must be specified.
```commandline
IGM.py molecule.xyz -p
```

## Output

Two cubes are generated as output: igm.cub is the defined gradient, mideig.cub is the
second eigenvalue of the density Hessian at each point on the same grid for colouring
purposes. Alternatively, diff.cub is the difference between the total electron density
and the sub of the fragment densities.

The visualization can be done by [VMD](http://www.ks.uiuc.edu/Research/vmd/), plotting
igm.cub as isosurface and colouring by the volume of mideig.cub/diff.cub.

A VMD visualization state file is automatically generated to aid the process. Please note
that VMD is looking for the cubes in the working directory.

### Example
The example inputs and outputs are provided for the case of benzene dimer. Further examples
will be available.

## Further remarks

Feel free to address problems to [Denes Berta](mailto:berta.denes@ttk.mta.hu).
