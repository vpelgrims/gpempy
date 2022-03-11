    gpempy -- GeneralDoc
    Copyright (C) 2022 V.Pelgrims

---

#				 G P E M P Y

#		    Galactic Polarized EMission in PYthon

***

### History:

gpempy has been developed originaly in Python 2.7 in the framework of the
[RADIOFOREGROUNDS](https://radioforegrounds.eu) project (radioforegrounds.eu) to reconstruct the large-scale
magnetic field of the Milky Way through the observation and analysis of the
diffuse polarized emission measured at submillimeter wavelengths by
cosmic microwave brackground experiments.
The codes has been upgraded in 2022 so that it can be run under Python 3.x.

### Context:

Our Galaxy is filled by an interstellar magnetic field that has an average strength of a few microgauss. This magnetic field lies in the entire disk of our Galaxy and extends into the halo up to high Galactic latitudes. The Galactic magnetic field appears to be inhomogeneous and it shows rapid spatial variations in magnitude and orientation. Some matter constituents of the Galaxy, such as dust grains or relativistic electrons, are sensitive to the ambient magnetic field. Matter and magnetic field couple in a complex way that leads to the emission of polarized light. In the domain of radio to submillimeter wavelengths, the polarized diffuse emissions are the synchrotron emission and the thermal dust emission, below and above about 80 GHz respectively. These diffuse emissions come from relativistic electrons that spiral along the lines of the magnetic field and from aspherical dust grains that line up their shortest axes with the magnetic field lines.

In principle, sky maps of the synchrotron and of the thermal dust polarized emissions may serve to study and constrain Galactic magnetic field models.
This suite of codes, gpempy, has been designed to make it possible to simulate full-sky maps of the Stokes parameters of the linear polarization of the synchrotron and thermal dust emission out of parametric models describing at large scales the matter density distribution and the three-dimensional (3D) vector field of the Galactic Magnetic Field.

### Goal and features:

The goal was to make the code relatively efficient and user-friendly.
As a result, there are several ways to call and execute the same routines/functions to compute things like 3D vector fields, density distributions, etc. This gives the user the choice to explore the models either in an abstract or specific way. Efforts have also been made to allow the user to easily switch from one specific functional form of the underlying parametric models to another using simple keywords, a convenient feature for model exploration.
This suite of codes _is not meant_ to be used in Bayesian analysis of the data such a through Markov Chain Monte Carlo treatment or other sampling methods.

### High-Level Package structure

The production of full-sky maps of polarized observable requires:
* the definition of the Galactic space; a grid
* to populate the Galaxy with some matter density distribution (e.g., dust and or relativistic electrons)
* to populate the Galaxy with a magnetic-field vector field
* to combine the above physical quantities in the appropriated way to model the emission
* to proceed to the line-of-sight integration to produce the sky maps of the observables.

In the package gpempy the workload is divided into four different modules, each dedicated to handling a specific part of the simulation process.

* [GalaxyBasics](GalaxyBasics.py) : that module is devoted
** to specify the Galactic space to be taken into account
** to define a convenient sampling of that space for the line-of-sight integration (based on HEALPix tessellation)
** to contain all the functions required for coordinate transformations and, in particular, to go from Galactocentric reference frame to heliocentric reference frame (the observer space) with scalar and vectorial quantities.

* [GalacticProfile](GalacticProfile.py) : that module is devoted
** to populate the Galaxy, defined from the above module, with matter density distribution.
It allows various different shapes (e.g., exponential disk, spherical halo, clumps, ...) with easily modulable functional forms of the coordinate functions.
** to visualize easily the resulting density distribution (cross-cut of the Galaxy and sky-projection)

* [BFIELD](BFIELD.py) : that module is devoted
** to populate the Galaxy with a given parametric model of the Galactic magnetic field.
It shall allows the user to choose among the numerous parametric models that can be found in the literature and potentially to modify some functional forms as well.
** to visualize easily the resulting magnetic vector field (cross-cut of the Galaxy and sky-projection)

* [GalacticForegrounds](GalacticForegrounds.py) : that module is devoted
** to compute the infinitesimal contributions of the diffuse emission according to specified models (at least for the dust)
** to proceed to the integration along the line-of-sight of the latter infinitesimal contributions to produce the diffuse emission sky map, strictly speaking, of the polarization observables.



### Authors:

V. Pelgrims wrote more than 99.99% of this code suite while working under the supervision of J.F. Macias-Perez as a postdoc of the [Radioforegrounds](https://radioforegrounds.eu) project.

### Related publication:

If this suite of codes is useful for you and your research please cite the following paper:
"Galactic magnetic field reconstruction using the polarized diffuse Galactic emission: formalism and application to Planck data"
Pelgrims, Macías-Pérez, Ruppin,
Astronomy & Astrophysics, Volume 652, id.A130, 31 pp.
10.1051/0004-6361/201833962


