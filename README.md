***
				 G P E M P Y

		    Galactic Polarized EMission in PYthon

***

``gpempy`` has been developed in the framework of the [RADIOFOREGROUNDS](https://radioforegrounds.eu) project to reconstruct the large-scale magnetic field of the Milky Way through the observation and analysis of the diffuse polarized emission measured at submillimeter wavelengths by cosmic microwave brackground experiments.

Specifically, "gpempy" is a simulation software to create full-sky maps of thermal dust and synchrotron polarized emission from three-dimensional parametric models describing the Galactic Magnetic field and the density distribution of matter (relativistic electron and dust grain) at large scales.

``gpempy`` is divided in four modules:
- BFIELD to estimate and visualize GMF models
- GalacticProfile to estimate and visualize models of matter density distribution
- GalacticForegrounds to estimate differential emission and produce full-sky maps of the integrated emission
- GalaxyBasics to handle space grid, coordinate transforms, etc.

``gpempy`` is modular and can be used to visualize and access models in a user-friendly way.
More general information on ``gpempy`` can be read [here](GeneralDoc.md) and we present an [overview](https://raw.githack.com/vpelgrims/gpempy/main/gpempy_overview.html), including practical examples, in the notebook.

If ``gpempy`` is useful for you and your research please cite [this](https://doi.org/10.1051/0004-6361/201833962) paper:
"Galactic magnetic field reconstruction using the polarized diffuse Galactic emission: formalism and application to Planck data"
Pelgrims, Macías-Pérez, Ruppin, Astronomy & Astrophysics, Volume 652, id.A130, 31 pp.

---
gpempy -- ReadMe  
Copyright (C) 2022  V.Pelgrims
