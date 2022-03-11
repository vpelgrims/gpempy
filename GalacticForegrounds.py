# -*- coding: utf-8 -*-
"""
    gpempy -- GalacticForegrounds
    Copyright (C) 2022  V.Pelgrims
    
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++                                                                       ++
++                     GALACTICFOREGROUNDS MODULE                        ++
++                                                                       ++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

|    Implementation of functions that generate all-sky maps of polarized
|    Galactic Foreground observables, that is the Stokes parameters I, Q,
|    and U, of the thermal dust and synchrotron emission
|
|    The line-of-sight integration is performed according to the midpoint
|    rule. The grid to sample the space is assumed to come from the
|    GalaxyBasics.GalacticTemplate() or similar architecture.
|
|    Starting Date of Module Creation: Oct 27 2016
|    @author: V.Pelgrims

"""

import numpy as np
import healpy as hp
import GalaxyBasics as gb
import GalacticProfile as gp
import BFIELD as bf

import time

###########################  THERMAL DUST EMISION  ##########################

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#               LEE AND DRAINE INSPIRED MODELIZATION                        #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def dIQU_dust(dust_density,Alpha_Bf,Gamma_Bf,**kwargs):
    """
    FUNCTION

        === DIQU_DUST : generate elemental IQU Stokes from thermal dust ===

    dIQU_dust(dust_density,
                  Alpha_Bf,
                  Gamma_Bf,
                  **kwargs{e_dust,
                           p_dust,
                           f_ma})

    From a dust profile, the angles describing the geometry of the projected
    Galactic B field and some other parameter, this functions evaluates the
    infinetisimal value of the Stokes I Q and U parameters from the Galactic
    thermal dust as:
        " dI = dl*e_dust*dust_density + dP*(1+cos(alpha)**2)/(3*sin(alpha)**2)
        " dQ = dP * cos(2*gamma)
        " dU = dP * sin(2*gamma)

        WHERE
        " dP = dl * e_dust * p^dust * dust density * f_g * f_ma
        with : 
              f_g = sin(alpha)**2 == geometrical factor
              f_ma = 3/2 (<cos(beta)**2> - 1/3) == misalignment factor
                   = | 0 if isotropic orientation
                     | 1 if perfect alignment with Bfield lines

              alpha the inclination angle of the Bfield line with respect to
                    the line of sigth
              gamma the polarization position angle perpendicular to the
                    Bfield component in the plane of the sky
              beta  the angle between grains' minor axes and B field line

    [see, e.g. Lee and Draine 1985 for this implementation]

    NOTE that the parameters e_dust and p_dust are correlated in this
         implementation ...
         p_dust is inversely proportional to e_dust and contains the
         information of the anisotropic extinction cross-section 


    INPUT:
    ------
     - dust_density : (npts,)-array with values of the dust density
         
     - Alpha_Bf : (npts,)-array with values of the inclination angle between
                  the line of sight and the B vector field

     - Gamma_Bf : (npts,)-array with values of the local polarization
                  position angle -- at 90 degree away from the projected B
                  vector field on the polarization plane

     **kwargs :
       - e_dust : dust emissivity.
                  Default is 1.

       - p_dust : free parameter of the model. Specifies a maximal degree
                  of linear polarization that can be obtained from thermal
                  dust emission.
                  Default is 0.2 but depend on dust model and is related to
                                 emissivity given the parameterization
                    
       - f_ma : missalignment factor is a scalar taking value between 0 and 1
                0 (for isotropic orientation of dust grains) and
                1 (for perfect alignment of shortest principal axes of dust
                   grains with the Bfield lines)
                Default is 0.8   -- arbitrary choice.

                         
    OUTPUT:
    -------
     - dI : Stokes I for the corresp. infinitesimal line element
     - dQ : Stokes I for the corresp. infinitesimal line element
     - dU : Stokes I for the corresp. infinitesimal line element
     
         They are (npts,)-arrays

    Created on Nov 21 2016
    @author: V.Pelgrims
    """

    #initialization to default setting
    default_setting = {'e_dust' : 1.0,
                       'p_dust' : 0.2,
                       'f_ma' : .8}

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        default_setting[key] = value
        if key not in ('e_dust','p_dust','f_ma'):
            raise ValueError('''
            Bad entries for keywored arguments. Key must be either:
            'e_dust', 'p_dust' or 'f_ma' ''')


    ### HERE WE GO
    #
    #dP
    dP = ( dust_density
           * default_setting['f_ma']
           * default_setting['e_dust']
           * default_setting['p_dust'] )
    #Stokes
    f_g = np.sin(Alpha_Bf)**2
    #Q
    dQ = dP * f_g * np.cos(2*Gamma_Bf)
    #U
    dU = dP * f_g * np.sin(2*Gamma_Bf)
    
    #intensity
    dI = ( dust_density
           * default_setting['e_dust']
           + dP * (2./3. - f_g) )
    
    
    #it is done
    return dI,dQ,dU


#
#
#


def IQU_dust(dust_density,Alpha_Bf,Gamma_Bf,*args,**kwargs):
    """
    FUNCTION

    === IQU_DUST: generate integrated IQU Stokes from thermal dust ===

    IQU_dust(dust_density,
                 Alpha_Bf,
                 Gamma_Bf,
                 *args{NPIX,
                       r_sampling_length,
                       step_r_sampling},
                 **kwargs{e_dust,
                          p_dust,
                          f_ma})

    From a dust profile, the angles describing the geometry of the projected
    Galactic B field, the NPIX parameter of the HEALPix map, the number of
    step and the step value for radial integration and some other parameters,
    this functions evaluates the integrated values of the Stokes I Q and U
    parameters from the Galactic thermal dust as:
        " I = sum along los of dI
        " Q = sum along los of dQ
        " U = sum along los of dU
    [see help(new_dIQU_dust) for expression of dI, dQ and dU]

    The integration is peformed as being simply the sum of the element along
    the same line of sight (linear discretization) times the length of the
    discretization interval.
    This is the most stupid integration one can imagine.

    INPUT:
    ------
     - dust_density : (npts,)-array with values of the dust density
         
     - Alpha_Bf : (npts,)-array with values of the inclination angle between
                  the line of sight and the B vector field

     - Gamma_Bf : (npts,)-array with values of the local polarization
                  position angle -- at 90 degree away from the projected B
                  vector field on the polarization plane

     *args:
       - NPIX : NPIX parameter of HEALPix map
                Default = 49152 = hp.nside2npix(64)

       - r_sampling_length : number of step along the line of sight
                             Default = 100

       - step_r_sampling : value of the interval of radial discretization
                           Default = 0.2 kpc

     **kwargs:
        - Same as for the function dIQU_dust()
                         

    OUTPUT:
    -------
     - I : integrated Stokes I
     - Q : integrated Stokes Q
     - U : integrated Stokes U
     
         They are (NPIX,)-arrays


    Created on Nov 22 2016
    @author: V.Pelgrims
    """

    Npix = 49152
    stepR = 0.2
    rSize = 100
    if len(args) >= 1:
        Npix = args[0]
        if len(args) ==3:
            rSize = args[1]
            stepR = args[2]

    ### HERE WE GO
    dI,dQ,dU = new_dIQU_dust(dust_density,
                             Alpha_Bf,
                             Gamma_Bf,
                             **kwargs)
    
    I = np.sum(np.reshape(dI,[rSize,Npix]),axis=0)*stepR
    Q = np.sum(np.reshape(dQ,[rSize,Npix]),axis=0)*stepR
    U = np.sum(np.reshape(dU,[rSize,Npix]),axis=0)*stepR

    return I,Q,U


#
#
#


def IQU_dust_DL(dust_density,Alpha_Bf,Gamma_Bf,*args,**kwargs):
    """
    FUNCTION

        === IQU_DUST_DL : integrated IQU Stokes from thermal dust
                          following Lee & Draine emission modelization ===

    IQU_dust_DL(dust_density,
                Alpha_Bf,
                Gamma_Bf,
                *args{NPIX,
                      r_sampling_length,
                      step_r_sampling},
                **kwargs{e_dust,
                         p_dust,
                         f_ma})

    From a dust profile, the angles describing the geometry of the projected
    Galactic B field, the NPIX parameter of the HEALPix map, the number of
    step and the step value for radial integration and parameters for the
    modelisation of the emission,
    this functions evaluates the integrated values of the Stokes I Q and U
    parameters from the Galactic thermal dust as:
        " I = sum along los of dI
        " Q = sum along los of dQ
        " U = sum along los of dU

    where 

        " dI = dl e_dust * dust_density + dP*(1+cos(alpha)**2)/(3*sin(alpha)**2)
        " dQ = dP * cos(2*gamma)
        " dU = dP * sin(2*gamma)
    where
        " dP = dl * e_dust * p^dust * dust density * f_g * f_ma
      with : 
            f_g = sin(alpha)**2 == geometrical factor
            f_ma = 3/2 (<cos(beta)**2> - 1/3) == misalignment factor
                 = | 0 if isotropic orientation
                   | 1 if perfect alignment with Bfield lines

            alpha the inclination angle of the Bfield line with respect to
                  the line of sigth
            gamma the polarization position angle perpendicular to the
                  Bfield component in the plane of the sky
            beta  the angle between grains' minor axes and B field line

    [see, e.g. Lee and Draine 1985 for this implementation]


    NOTE that the parameters e_dust and p_dust are correlated in this
         implementation ...
         p_dust is inversely proportional to e_dust and contains the
         information of the anisotropic extinction cross-section 


    The integration is peformed as being simply the sum of the element along
    the same line of sight (linear discretization) times the length of the
    discretization interval.
    This is the most stupid integration one can imagine.


    INPUT:
    ------
     - dust_density : (npts,)-array with values of the dust density
         
     - Alpha_Bf : (npts,)-array with values of the inclination angle between
                  the line of sight and the B vector field

     - Gamma_Bf : (npts,)-array with values of the local polarization
                  position angle -- at 90 degree away from the projected B
                  vector field on the polarization plane

     *args:
       - NPIX : NPIX parameter of HEALPix map
                Default = 49152 = hp.nside2npix(64)

       - r_sampling_length : number of step along the line of sight
                             Default = 100

       - step_r_sampling : value of the interval of radial discretization
                           Default = 0.2 kpc


     **kwargs :
       - e_dust : dust emissivity.
                  Default is 1.

       - p_dust : free parameter of the model. Specifies a maximal degree
                  of linear polarization that can be obtained from thermal
                  dust emission.
                  Default is 0.2 but depend on dust model and is related to
                                 emissivity given the parameterization
                    
       - f_ma : missalignment factor is a scalar taking value between 0 and 1
                0 (for isotropic orientation of dust grains) and
                1 (for perfect alignment of smallest principal axes of dust
                   grains with the Bfield lines)
                Default is 0.8  -- arbitrary choice.
                         

    OUTPUT:
    -------
     - I : integrated Stokes I
     - Q : integrated Stokes Q
     - U : integrated Stokes U
     
         They are (NPIX,)-arrays


    Created on Nov 23 2016
    @author: V.Pelgrims
    """

    #grid parameters
    Npix = 49152
    stepR = 0.2
    rSize = 100
    if len(args) >= 1:
        Npix = args[0]
        if len(args) ==3:
            rSize = args[1]
            stepR = args[2]

    #parameters for the emission modelization
    #initialization to default setting
    default_setting = {'e_dust' : 1.0,
                       'p_dust' : 0.2,
                       'f_ma' : .8}

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        default_setting[key] = value
        if key not in ('e_dust','p_dust','f_ma'):
            raise ValueError('''
            Bad entries for keywored arguments. Key must be either:
            'e_dust', 'p_dust' or 'f_ma' ''')
        #perhaps, this conditional test should be removed for speed in MCMC

    # HERE WE GO
    #first define explicitly the infinitesimal dI, dQ, dU
    #
    scalar_p = (default_setting['f_ma']
                * default_setting['e_dust']
                * default_setting['p_dust'])

    #geometrical factor
    f_g = np.sin(Alpha_Bf)**2

    #infinitesimal Stokes parameters
    dQ_ = dust_density * f_g * np.cos(2*Gamma_Bf)
    dU_ = dust_density * f_g * np.sin(2*Gamma_Bf)
    dI = dust_density * ( 1.0
                          + default_setting['f_ma']
                          * default_setting['p_dust']
                          * (2.0/3.0 - f_g) )

    #Integration with multiplicatif factors
    I = ( default_setting['e_dust']
          * np.sum(np.reshape(dI,[rSize,Npix]),axis=0) * stepR )
    Q = scalar_p * np.sum(np.reshape(dQ_,[rSize,Npix]),axis=0)*stepR
    U = scalar_p * np.sum(np.reshape(dU_,[rSize,Npix]),axis=0)*stepR

    return I,Q,U


#
#
#


def IQU_dust_fromModels(dust,bfield,*args,**kwargs):
    """
    FUNCTION

        === IQU_DUST_FROMMODEL: IQU Stokes from setting model files ===


    IQU_dust_fromModels(dust,
                        bfield,
                        *args{XYZ_gal,
                              dotproducts},
                        **kwargs{e_dust,
                                 p_dust,
                                 f_ma,
                                 NPIX,
                                 radial_sampling_length,
                                 radial_step})

    
    INPUT:
    ------
     - dust : dictionary that specifies the dust profile to be built
              using gp.Duster()

     - bfield : dictionary that specifies the bfield model to be built
                using bf.BFielder()

     *args:
       - XYZ_gal : Galactic coordinate where to evaluate the dust density
                   and the magnetic vector field.
                   Default is the default of gb.GalacticTemplate() and is
                   computed if not given as entry (time consuming)

       - dotproducts : dot products at all locations between the basis
                       vectors of the observer coordinate system and the
                       GalactoCentric one, e.g., given by
                       gb.GalacticTemplate().
                       Defaut is the default of gb.GalacticTemplate() and
                       is computed if not given as entry (time consuming)

     **kwargs:
       - Same as for the function dIQU_dust()

     AND :
       - NPIX : NPIX parameter for output map

       - radial_sampling_length : numbre of cells for l.o.s discretization

       - radial_step : [kpc] value of interval for l.o.s discretization


    Created on Oct 28 2016
    @author: V.Pelgrims

    """

    if len(args) == 0:
        #creat XYZ_gal and dotproducts
        _,XYZ_gal,dots = gb.GalacticTemplate()
    elif len(args) == 2:
        #XYZ_gal and dotproducts are given
        XYZ_gal = args[0]
        dots = args[1]
    else:
        raise ValueError('''
    Wrong number of given arguments.
    If given, they should be cartesian coordinates in GalactoCentric RF
    followed by dotproducts between basis vectors''')

    arguments = []
    if 'NPIX' in kwargs.keys():
        arguments.append(kwargs.pop('NPIX'))
    if 'radial_step' in kwargs.keys():
        arguments.append(kwargs.pop('radial_sampling_length'))
        arguments.append(kwargs.pop('radial_step'))
    if len(arguments) > 1:
        if arguments[0]*arguments[1] != len(XYZ_gal[0]):
            raise ValueError('''
    The size of the Galactic template is not compatible with the
    values of NPIX and radial_sampling_length.''')
    
    if type(dust) == dict:
        #everything needs to be computed for the dust density
        dust_density = gp.Duster(XYZ_gal,dust)
    else:
        #the dust density has already been evaluated before
        print('''
  - Interpret the dust density as being in Galactocentric reference frame''')
        dust_density = dust

    if type(bfield) == dict:
        #everything needs to be computed for the bfield model
        bfield_model = bf.BFielder(XYZ_gal,bfield)
    else:
        #the B field has already been computed
        print('''
  - Interpret the B vector field as being in Galactocentric reference frame''')
        bfield_model = bfield

    #bield related angles
    Alpha,Gamma = bf.__get_AlphaGamma_fromGal(bfield_model,dots)

    I,Q,U = IQU_dust(dust_density,
                     Alpha,
                     Gamma,
                     *arguments,
                     **kwargs)

    return I,Q,U


#
#
#

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                  Following Fauvet et al 2010 implementation               #

#   ! Un-physical implementation of grain alignment efficiency.             #
#   Kept for historical reasons                                             #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def fauvet_dIQU_dust(dust_density,Alpha_Bf,Gamma_Bf,**kwargs):
    """
    FUNCTION

    === FAUVET_DIQU_DUST : generate elemental IQU Stokes from thermal dust ===

    fauvet_dIQU_dust(dust_density,
              Alpha_Bf,
              Gamma_Bf,
              **kwargs{e_dust,
                       p_dust,
                       f_geom_model,
                       lambda_f_ma})

    From a dust profile, the angles describing the geometry of the projected
    Galactic B field and some other parameter, this functions evaluates the
    infinetisimal value of the Stokes I Q and U parameters from the Galactic
    thermal dust as:
        " dI = e_dust * dust_density * dl
        " dQ = dI * p_dust * cos(2*gamma) * f_g * f_ma
        " dU = dI * p_dust * sin(2*gamma) * f_g * f_ma

    [see Fauvet et al 2010 for this implementation]


    INPUT:
    ------
     - dust_density : (npts,)-array with values of the dust density
         
     - Alpha_Bf : (npts,)-array with values of the inclination angle between
                  the line of sight and the B vector field

     - Gamma_Bf : (npts,)-array with values of the local polarization
                  position angle -- at 90 degree away from the projected B
                  vector field on the polarization plane

     **kwargs :
       - e_dust : dust emissivity.
                  Default is 1.

       - p_dust : free parameter of the model. Specifies a maximal degree
                  of linear polarization that can be obtained from thermal
                  dust emission.
                  Default is 1.

       - f_geom_model : specifies the geometric factor to be used
                    'Fauvet' sets f_geom = (sin(Alpha_Bf))**2 (Fauvet 2010)
                     None sets f_geom = 1.
                          Default is 'Fauvet'.
                    
       - lambda_f_ma : specifies the index for the model of missalignment
                       of dust grain with respect to the Bfield line
                           " f_ma = (sin(Alpha_Bf))**lambda_f_ma
                       typically 1/2,1,3/2.
                       Default is 1...
                   !!! WATCH OUT THAT THIS IS NOT CORRECT AS NON PHYSICAL!!!
                       SHOULD BE SET TO 0 IN THE FUTURE                         

                         
    OUTPUT:
    -------
     - dI : Stokes I for the corresp. infinitesimal line element
     - dQ : Stokes I for the corresp. infinitesimal line element
     - dU : Stokes I for the corresp. infinitesimal line element
     
         They are (npts,)-arrays

    Created on Oct 27 2016
    Based on get_dIQU_dust.py written on Jul 6 2016

    @author: V.Pelgrims
    """

    #initialization to default setting
    default_setting = {'e_dust' : 1.0,
                       'p_dust' : 1.0,
                       'f_geom_model' : 'Fauvet',
                       'lambda_f_ma' : 1.0}

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        default_setting[key] = value
        if key not in ('e_dust','p_dust','f_geom_model','lambda_f_ma'):
            raise ValueError('''
            Bad entries for keywored arguments. Key must be either:
            'e_dust', 'p_dust', 'f_geom_model' or 'lambda_f_ma' ''')

    #geometric factor
    if default_setting['f_geom_model'] == 'Fauvet':
        f_geom = (np.sin(Alpha_Bf))**2
    elif default_setting['f_geom_model'] == None:
        f_geom = 1. + 0. * Alpha_Bf
    else:
        raise ValueError('''
        Bad entry for keyword argument "f_geom_model".',
        'The value should be either "Fauvet" or None''')

    #missalignment factor
    f_ma = (np.sin(Alpha_Bf))**default_setting['lambda_f_ma']
    #NOTE: this function makes certainly not the wanted job...
    #SHOULD BE SET TO 0 AS IT CORRESPONDS TO NOTHING PHYSICAL

    ### HERE WE GO
    #intensity
    dI = default_setting['e_dust'] * dust_density
    
    commonQU = (dI * default_setting['p_dust']
                * f_geom
                * f_ma)
    #Q Stokes
    dQ = commonQU * np.cos(2*Gamma_Bf)
    #U Stokes
    dU = commonQU * np.sin(2*Gamma_Bf)
    
    #it is done
    return dI,dQ,dU


#
#
#


def fauvet_IQU_dust(dust_density,Alpha_Bf,Gamma_Bf,*args,**kwargs):
    """
    FUNCTION

    === FAUVET_IQU_DUST: generate integrated IQU Stokes from thermal dust ===

    IQU_dust(dust_density,
             Alpha_Bf,
             Gamma_Bf,
             *args{NPIX,
                   r_sampling_length,
                   step_r_sampling},
             **kwargs{e_dust,
                      p_dust,
                      f_geom_model,
                      lambda_f_ma})

    From a dust profile, the angles describing the geometry of the projected
    Galactic B field, the NPIX parameter of the HEALPix map, the number of
    step and the step value for radial integration and some other parameters,
    this functions evaluates the integrated values of the Stokes I Q and U
    parameters from the Galactic thermal dust as:
        " I = sum along los of (dI = e_dust * dust_density * dl)
        " Q = sum along los of (dQ = dI * p_dust * cos(2*gamma) * f_g * f_ma)
        " U = sum along los of (dU = dI * p_dust * sin(2*gamma) * f_g * f_ma)
    [see Fauvet et al 2010]

    The line-of-sight integration of the observables is peformed following the
    simple midpoint rules; it is simply the sum of the element along the same
    line of sight (linear discretization) times the length of the
    discretization interval.

    INPUT:
    ------
     - dust_density : (npts,)-array with values of the dust density
         
     - Alpha_Bf : (npts,)-array with values of the inclination angle between
                  the line of sight and the B vector field

     - Gamma_Bf : (npts,)-array with values of the local polarization
                  position angle -- at 90 degree away from the projected B
                  vector field on the polarization plane

     *args:
       - NPIX : NPIX parameter of HEALPix map
                Default = 49152 = hp.nside2npix(64)

       - r_sampling_length : number of step along the line of sight
                             Default = 100

       - step_r_sampling : value of the interval of radial discretization
                           Default = 0.2 kpc

     **kwargs:
        - Same as for the function dIQU_dust()
                         

    OUTPUT:
    -------
     - I : integrated Stokes I
     - Q : integrated Stokes Q
     - U : integrated Stokes U
     
         They are (NPIX,)-arrays


    Created on Oct 27 2016
    Based on get_IQU_dust.py written on Sep 6 2016

    @author: V.Pelgrims
    """

    Npix = 49152
    stepR = 0.2
    rSize = 100
    if len(args) >= 1:
        Npix = args[0]
        if len(args) == 3:
            rSize = args[1]
            stepR = args[2]
        elif ((len(args) == 2) or (len(args)>=4)):
            raise ValueError('''args has a wrong dimension''')
        #

    ### HERE WE GO
    dI,dQ,dU = fauvet_dIQU_dust(dust_density,
                         Alpha_Bf,
                         Gamma_Bf,
                         **kwargs)
    
    I = np.sum(np.reshape(dI,[rSize,Npix]),axis=0)*stepR
    Q = np.sum(np.reshape(dQ,[rSize,Npix]),axis=0)*stepR
    U = np.sum(np.reshape(dU,[rSize,Npix]),axis=0)*stepR

    return I,Q,U


#
#
#


###########################  SYNCHROTRON EMISION  ##########################

def dIQU_sync(re_profile,Transv_Bf,Gamma_Bf,**kwargs):
    """
    FUNCTION

        === dIQU_sync : generate elemental IQU Stokes from synchrotron ===

    dIQU_sync(rel_elect_profile,
              transverseBfield_sq,
              Gamma_Bf,
              **kwargs{e_sync,
                       s_sync})

    From a profile of relativistic electrons, relevant quantities of Galactic
    magnetic field and emission parameters, this function computes the
    elemental contribution to I,Q,U of polarized synchrotron emission as
    given by:
        " dI = e_sync * rel_elect_profile
                      * (transverseBfield**2)**((s+1)/4) * dl
        " dQ = dI * p_sync * cos(2*gamma)
        " dU = dI * p_sync * sin(2*gamma)

            where " p_sync = (s+1)/(s+7/3) (s=3 usually)
                  " transverseBfield**2 = B_theta**2 + B_phi**2
                      with B_theta and B_phi the spherical components of
                      the Bfield, so transverse to the line of sight.
    
    INPUT:
    ------
     - rel_elect_profile : (npts,)-array with relativistic electron density
         
     - transverseBfield_sg: (npts,)-array with the square of the norm of
                            the transverse component of the Bfield

     - Gamma_Bf : (npts,)-array with values of the local polarization
                  position angle -- at 90 degree away from the projected B
                  vector field on the polarization plane

     **kwargs :
         - e_sync : synchrotron emissivity.
                    Default is 1.

         - s_sync : synchrotron spectral index (antenna). Default is 3
                    Is related to p_sync, the maximal degree of linear
                    polarization due to synchrotron as
                        " p_sync = (s+1)/(s+7/3) (Default = 3/4.)
                         
    OUTPUT:
    -------
     - dI : Stokes I for the corresp. infinitesimal line element
     - dQ : Stokes I for the corresp. infinitesimal line element
     - dU : Stokes I for the corresp. infinitesimal line element
     
         They are (npts,)-arrays

    Created on Oct 28 2016
    Based on get_dIQU_sync.py written on Jul 7 2016

    @author: V.Pelgrims
    """


    #initialization of **kargs-related variables
    default_setting = {'e_sync':1.0,
                       's_sync':3.0}

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        default_setting[key] = value
        if key not in ('e_sync','s_sync'):
            raise ValueError('''
            Bad entries for keywored arguments. Key must be either:
            'e_sync', 's_sync' ''')


    s_sync = default_setting['s_sync']
    p_sync = (s_sync + 1.)/(s_sync + 7./3.)

    ### HERE WE GO
    #intensity
    dI = (default_setting['e_sync']
          * re_profile * Transv_Bf**((s_sync+1.)/4.))
    
    #Q Stokes
    dQ = dI * p_sync * np.cos(2*Gamma_Bf)
    #U Stokes
    dU = dI * p_sync * np.sin(2*Gamma_Bf)
    
    #it is done
    return dI,dQ,dU


#
#
#


def IQU_sync(re_density,Trans_Bf,Gamma_Bf,*args,**kwargs):
    """
    FUNCTION

    === IQU_SYNC: generate integrated IQU Stokes from synchrotron ===

    IQU_sync(rel_elect_density,
             Btransverse_sq,
             Gamma_Bf,
             *args{NPIX,
                   r_sampling_length,
                   step_r_sampling},
             **kwargs{e_sync,
                      s_sync})

    From a relativistic electron density profile, the squared transverse norm
    of the Galactic B field, the elemental polarization position angle, the
    NPIX parameter of the HEALPix map, the number of step and the step value
    for radial integration and some other parameters,
    this functions evaluates the integrated values of the Stokes I Q and U
    parameters from the Galactic synchrotron as:
        " I = sum along los of (dI = e_sync * rel_elect_profile
                      * (transverseBfield**2)**((s+1)/4) * dl)
        " Q = sum along los of (dQ = dI * p_sync * cos(2*gamma))
        " U = sum along los of (dU = dI * p_sync * sin(2*gamma))

    The integration is peformed according to the midpoint rule. It is simply
    the sum of the element along the same line of sight (linear discretization)
    times the length of the discretization interval.


    INPUT:
    ------
     - rel_elect_profile : (npts,)-array with relativistic electron density
         
     - transverseBfield_sq: (npts,)-array with the square of the norm of
                            the transverse component of the Bfield

     - Gamma_Bf : (npts,)-array with values of the local polarization
                  position angle -- at 90 degree away from the projected B
                  vector field on the polarization plane

     *args:
       - NPIX : NPIX parameter of HEALPix map
                Default = 49152 = hp.nside2npix(64)

       - r_sampling_length : number of step along the line of sight
                             Default = 100

       - step_r_sampling : value of the interval of radial discretization
                           Default = 0.2 kpc

     **kwargs:
        - Same as for the function dIQU_sync()
                         

    OUTPUT:
    -------
     - I : integrated Stokes I
     - Q : integrated Stokes Q
     - U : integrated Stokes U
     
         They are (NPIX,)-arrays


    Created on Oct 27 2016
    Based on get_IQU_dust.py written on Sep 6 2016

    @author: V.Pelgrims
    """

    Npix = 49152
    stepR = 0.2
    rSize = 100
    if len(args) >= 1:
        Npix = args[0]
        if len(args) ==3:
            rSize = args[1]
            stepR = args[2]

    ### HERE WE GO
    dI,dQ,dU = dIQU_sync(re_density,
                         Trans_Bf,
                         Gamma_Bf,
                         **kwargs)
    
    I = np.sum(np.reshape(dI,[rSize,Npix]),axis=0)*stepR
    Q = np.sum(np.reshape(dQ,[rSize,Npix]),axis=0)*stepR
    U = np.sum(np.reshape(dU,[rSize,Npix]),axis=0)*stepR

    return I,Q,U


#
#
#


def IQU_sync_fromModels(CRL,bfield,*args,**kwargs):
    """
    FUNCTION

        === IQU_SYNC_FROMMODEL: IQU Stokes from setting model files ===


    IQU_sync_fromModels(CRL,
                        bfield,
                        *args{XYZ_gal,
                              dotproducts},
                        **kwargs{e_sync,
                                 s_sync,
                                 NPIX,
                                 radial_sampling_length,
                                 radial_step})


    INPUT:
    ------
     - CRL : dictonary that specified the cosmic ray leptons profile to
             be built, here using bp.Duster()

     - bfield : dictionary that specifield the bfield model to be built
                using bf.BFielder()

     *args:
       - XYZ_gal : Galactic coordinate where to evaluate the CRL density
                   and the magnetic vector field.
                   Default is the default of gb.GalacticTemplate() and is
                   computed if not given as entry (time consuming)

       - dotproducts : dot products at all locations between the basis
                       vectors of the observer coordinate system and the
                       GalactoCentric one, e.g., given by
                       gb.GalacticTemplate().
                       Defaut is the default of gb.GalacticTemplate() and
                       is computed if not given as entry (time consuming)

     **kwargs:
       - Same as for the function dIQU_sync()

     AND :
       - NPIX : NPIX parameter for output map

       - radial_sampling_length : numbre of cells for l.o.s discretization

       - radial_step : [kpc] value of interval for l.o.s discretization


    Created on Nov 2 2016
    @author: V.Pelgrims

    """

    if len(args) == 0:
        #creat XYZ_gal and dotproducts
        _,XYZ_gal,dots = gb.GalacticTemplate()
    elif len(args) == 2:
        #XYZ_gal and dotproducts are given
        XYZ_gal = args[0]
        dots = args[1]
    else:
        raise ValueError('''
    Wrong number of given arguments.
    If given, they should be cartesian coordinates in GalactoCentric RF
    followed by dotproducts between basis vectors''')

    arguments = []
    if 'NPIX' in kwargs.keys():
        arguments.append(kwargs.pop('NPIX'))
    if 'radial_step' in kwargs.keys():
        arguments.append(kwargs.pop('radial_sampling_length'))
        arguments.append(kwargs.pop('radial_step'))
    if len(arguments) != 0:
        if arguments[0]*arguments[1] != len(XYZ_gal[0]):
            raise ValueError('''
    The size of the Galactic template is not compatible with the
    values of NPIX and radial_sampling_length.''')
    
    if type(CRL) == dict:
        #everything needs to be computed for the rel. electron density
        crl_density = gp.Duster(XYZ_gal,CRL)
    else:
        #the rel. electron density has already been evaluated before
        print('''
  - Interpret the CRL density as being in Galactocentric reference frame''')
        crl_density = CRL

    if type(bfield) == dict:
        #everything needs to be computed for the bfield model
        bfield_model = bf.BFielder(XYZ_gal,bfield)
    else:
        #the B field has already been computed
        print('''
  - Interpret the B vector field as being in Galactocentric reference frame''')
        bfield_model = bfield

    #bield related angles
    _,Gamma,Btrans2 = bf.__get_AlphaGammaBtrans2_fromGal(bfield_model,dots)

    I,Q,U = IQU_sync(crl_density,
                     Btrans2,
                     Gamma,
                     *arguments,
                     **kwargs)

    return I,Q,U


#
#
#


#######################  SYNCHROTRON AND THERMAL DUST  #######################

def IQU_SandD(dust,crl,Alpha_Bf,Gamma_Bf,Btrans2,*args,**kwargs):
    """
    FUNCTION

    === IQU_SANDD: integrated IQU Stokes from thermal dust and Synchrotron ===

    IQU_dust(dust_density,
             CRL_density,
             Alpha_Bf,
             Gamma_Bf,
             Btrans2,
             *args{NPIX,
                   r_sampling_length,
                   step_r_sampling},
             **kwargs{e_dust,
                      p_dust,
                      f_ma,
                      s_sync,
                      e_sync})

    Use IQU_dust() and IQU_sync()

    From a dust profile, a relativistic electron density profile, the angles
    describing the Galactic B fiel and the norm (squared) of its transvers
    part and other arguments, this functions evalutes the integrated values
    of the Stokes I Q and U parameters from the Galactic thermal dust AND
    from synchrotron emission.
    [see IQU_dust anf IQU_sync for parameterization and modelling]

    The integration is peformed according to the midpoint rule. It is simply
    the sum of the element along the same line of sight (linear discretization)
    times the length of the discretization interval.

    INPUT:
    ------
     - dust_density : (npts,)-array with values of the dust density

     - CRL_density : (npts,)-array with values of the rel. electron density
         
     - Alpha_Bf : (npts,)-array with values of the inclination angle between
                  the line of sight and the B vector field

     - Gamma_Bf : (npts,)-array with values of the local polarization
                  position angle -- at 90 degree away from the projected B
                  vector field on the polarization plane

     - Btrans2 : (npts,)-array with values of the squared norm of the
                 transversed part of the  magnetic field vectors

     *args:
       - NPIX : NPIX parameter of HEALPix map
                Default = 49152 = hp.nside2npix(64)

       - r_sampling_length : number of step along the line of sight
                             Default = 100

       - step_r_sampling : value of the interval of radial discretization
                           Default = 0.2 kpc

     **kwargs:
        - Same as for the function dIQU_dust()
        - Same as for the function dIQU_sync()
                         

    OUTPUT:
    -------
     - Id : integrated Stokes I from thermal dust
     - Qd : integrated Stokes Q from thermal dust
     - Ud : integrated Stokes U from thermal dust
     - Is : integrated Stokes I from synchrotron
     - Qs : integrated Stokes Q from synchrotron
     - Us : integrated Stokes U from synchrotron
     
         They are (NPIX,)-arrays


    Created on Nov 2 2016
    @author: V.Pelgrims
    """

    #separate keyword argument for dust and synchrotron
    kwargs_s = {}
    if 's_sync' in kwargs.items():
        kwargs_s['s_sync'] = kwargs.pop('s_sync')
    if 'e_sync' in kwargs.items():
        kwargs_s['e_sync'] = kwargs.pop('e_sync')
    #if kwargs still contains something, it is keyword arguments for
    #dust related functions


    #Here I use the other functions to do the job.
    #For the shake of speed it would be better to explicitely perform
    #the computation here.

    #dust
    I_d,Q_d,U_d = IQU_dust(dust,
                           Alpha_Bf,
                           Gamma_Bf,
                           *args,
                           **kwargs)
    #synchrotron
    I_s,Q_s,U_s = IQU_sync(crl,
                           Btrans2,
                           Gamma_Bf,
                           *args,
                           **kwargs_s)

    return I_d,Q_d,U_d,I_s,Q_s,U_s


#
#
#


def IQU_SandD_fromModels(dust,CRL,bfield,*args,**kwargs):
    """
    FUNCTION

        === IQU_SANDD_FROMMODEL: IQU Stokes from setting model files ===


    IQU_SandD_fromModels(dust,
                         CRL,
                         bfield,
                         *args{XYZ_gal,
                               dotproducts},
                         **kwargs{e_dust,
                                  p_dust,
                                  f_ma,
                                  e_sync,
                                  s_sync,
                                  NPIX,
                                  radial_sampling_length,
                                  radial_step})


    INPUT:
    ------
     - dust : dictonary that specified the dust profile to be built,
              using bp.Duster()

     - CRL : dictonary that specified the cosmic ray leptons profile to
             be built, here using bp.Duster()

     - bfield : dictionary that specifield the bfield model to be built
                using bf.BFielder()

     *args:
       - XYZ_gal : Galactic coordinate where to evaluate the CRL density
                   and the magnetic vector field.
                   Default is the default of gb.GalacticTemplate() and is
                   computed if not given as entry (time consuming)

       - dotproducts : dot products at all locations between the basis
                       vectors of the observer coordinate system and the
                       GalactoCentric one, e.g., given by
                       gb.GalacticTemplate().
                       Defaut is the default of gb.GalacticTemplate() and
                       is computed if not given as entry (time consuming)

     **kwargs:
       - Same as for the function dIQU_sync()
       - Same as for the functions dIQU_dust()

     AND :
       - NPIX : NPIX parameter for output map

       - radial_sampling_length : numbre of cells for l.o.s discretization

       - radial_step : [kpc] value of interval for l.o.s discretization


    OUTPUT:
    -------
     - Id : integrated Stokes I from thermal dust
     - Qd : integrated Stokes Q from thermal dust
     - Ud : integrated Stokes U from thermal dust
     - Is : integrated Stokes I from synchrotron
     - Qs : integrated Stokes Q from synchrotron
     - Us : integrated Stokes U from synchrotron

    Created on Nov 2 2016
    @author: V.Pelgrims

    """

    if len(args) == 0:
        #creat XYZ_gal and dotproducts
        _,XYZ_gal,dots = gb.GalacticTemplate()
    elif len(args) == 2:
        #XYZ_gal and dotproducts are given
        XYZ_gal = args[0]
        dots = args[1]
    else:
        raise ValueError('''
    Wrong number of given arguments.
    If given, they should be cartesian coordinates in GalactoCentric RF
    followed by dotproducts between basis vectors''')

    arguments = []
    if 'NPIX' in kwargs.keys():
        arguments.append(kwargs.pop('NPIX'))
    if 'radial_step' in kwargs.keys():
        arguments.append(kwargs.pop('radial_sampling_length'))
        arguments.append(kwargs.pop('radial_step'))
    if len(arguments) != 0:
        if arguments[0]*arguments[1] != len(XYZ_gal[0]):
            raise ValueError('''
    The size of the Galactic template is not compatible with the
    values of NPIX and radial_sampling_length.''')

    #deal with various possiblities for keyword arguments and send them
    #to the right functions
    kwargs_s = {}
    if 'e_sync' in kwargs.items():
        kwargs_s['e_sync'] = kwargs.pop('e_sync')
    if 's_sync' in kwargs.items():
        kwargs_s['s_sync'] = kwargs.pop('s_sync')
    #if kwargs still contains something, it is keyword arguments for
    #dust related functions
        

    if type(dust) == dict:
        #everything needs to be computed for the dust density
        dust_density = gp.Duster(XYZ_gal,dust)        
    else:
        #the rel. electron density has already been evaluated before
        print('''
  - Interpret the dust density as being in Galactocentric reference frame''')
        dust_density = dust


    if type(CRL) == dict:
        #everything needs to be computed for the rel. electron density
        crl_density = gp.Duster(XYZ_gal,CRL)
    else:
        #the rel. electron density has already been evaluated before
        print('''
  - Interpret the CRL density as being in Galactocentric reference frame''')
        crl_density = CRL


    if type(bfield) == dict:
        #everything needs to be computed for the bfield model
        bfield_model = bf.BFielder(XYZ_gal,bfield)
    else:
        #the B field has already been computed
        print('''
  - Interpret the B vector field as being in Galactocentric reference frame''')
        bfield_model = bfield

    #bield related angles
    Alpha,Gamma,Btrans2 = bf.__get_AlphaGammaBtrans2_fromGal(bfield_model,dots)

    #dust
    Id,Qd,Ud = IQU_dust(dust_density,
                        Alpha,
                        Gamma,
                        *arguments,
                        **kwargs)

    #synchrotron
    Is,Qs,Us = IQU_sync(crl_density,
                        Btrans2,
                        Gamma,
                        *arguments,
                        **kwargs_s)

    return Id,Qd,Ud,Is,Qs,Us


#
#
#




###########################################################################
#                                                                         #
#################################  E N D  #################################
#                                                                         #
###########################################################################
