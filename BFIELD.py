# -*- coding: utf-8 -*-
"""
    gpempy -- BFIELD
    Copyright (C) 2022  V.Pelgrims
    
 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 ++                                                                       ++
 ++                             BFIELD MODULE                             ++
 ++                                                                       ++
 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 |    Implementation of various vector field configurations that can
 |    be used to model large-scale Galactic magnetic fields
 |
 |    and some basic functions to deal with them and compute relevant
 |    quantities necessary for observable evaluation
 |
 |    and some basic functions for an easy visualization of the models
 |
 |    Starting Date of Module Creation: Oct 24 2016
 |    @author: V.Pelgrims

 """

import numpy as np
import GalaxyBasics as gb

pi = np.pi
deg2rad = np.pi / 180.0

def ASS(coord,**kwargs):
    """
    FUNCTION
        ===  ASS : AxiSymmetric Spiral Galactic magnetic field model ===

    ASS(coordinates,**kwargs{numerous parameters})

        " vec{B} = B_rho vec{u_rho} + B_phi vec{u_phi} + B_z vec{u_z}
            where
           " B_rho = B_amp(rho,z) * sin(pitch) * cos(Xi(z))
           " B_phi = B_amp(rho,z) * cos(pitch) * cos(Xi(z))
           " B_z   = B_amp(rho,z) * sin(Xi(z))
           with
               " Xi(z) = Xi_0 * tanh(z / z_0)
               " B_amp(rho,z) = cst,
                                funct of sph. radial coord
                                funct of cyl. radial coord
               " pitch = pitch angle


    INPUT:
    ------
     - coord :  an (3,n)-array with Galactic coordinates at which the
                Bfield has to be evaluated. Default format is cylindrical.

     **kwargs :
       - coord_format : string that specifies the coord format of the INPUT
                        coordinates. Can be cartesian, spherical or cylindrical
                        (the output are in cylindrical, what so ever)
                        Default is 'cartesian'
       - B_amp_type : string that specifies if the amplitude is to be
                      a constante ['cst'], a function of the cyl. radial
                      coordinate ['cyl'] or a function of the SPH. radial
                      coordinate ['sph']
                      Default = 'cyl'


      [model parameter]
       - B_0 : amplitude of the regular large-scale B field at the Sun [muG]
       - B_amp_param : additional parameter for the radial dependence
                       (if it is the case)
       - B_amp_type : string to specify the radial dependence
                      'cst', 'cyl','sph' for constant, cylindircal or spherical
    
       - pitch : pitch angle [rad] of the spiral arms [Default = 11.5 deg]
       - rho_0 : radial scale = dist. between the Sun and the Gal Centre [kpc]
       - Xi_0 :  tilt parameter in [rad]
       - z_0 :   a vertical scale in [kpc]

       > getBFieldDefault('ASS') for default setting values of parameters
    


    OUTPUT:
    ------
    B_rho, B_z, B_phi : the component of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at
                        all locations specified in 'coord'


    Created on Jul 4 2017
    @author: V.Pelgrims

    """

    #initialization of default ASS model parameters
    parameter = {'pitch':11.5*deg2rad,
                 'Xi_0':25.0*deg2rad,
                 'z_0':1.0,
                 'rho_0':8.0
                 }

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value
        if key not in ('B_0','B_amp_param','B_amp_type',
                       'pitch','Xi_0','z_0','rho_0',
                       'coord_format'):
            raise ValueError('''
            Bad entry for optional argument. Key must be one either
            'B_0','B_amp_param','B_amp_type','pitch','Xi_0','z_0',
            'rho_0','coord_format' ''')

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')


    B_amp = __B0_of_r(rho,z,**parameter)

    ### Tilt angle
    Xi_z = parameter['Xi_0'] * np.tanh(z / parameter['z_0'])

    
    #cylindrical component of the magnetic vector field
    B_rho = B_amp * np.sin(parameter['pitch']) * np.cos(Xi_z)
    B_phi = B_amp * np.cos(parameter['pitch']) * np.cos(Xi_z)
    B_z = B_amp * np.sin(Xi_z)

    return np.array([B_rho,B_z,B_phi])


#
#
#


def BSS(coord,**kwargs):
    """
    FUNCTION
            === BSS : BiSymmetric Spiral Galactic magnetic field model ===

    It is the same model as the MLS model, modulo conventions

        BSS(coordinates,**kwargs{numerous parameters})

        " vec{B} = B_rho vec{u_rho} + B_phi vec{u_phi} + B_z vec{u_z}

        where           
           " B_rho = B_amp(rho,z) * cos(phi - beta * ln(rho/rho_0))
                               * sin(pitch) * cos(Xi(z))

           " B_phi = B_amp(rho,z) * cos(phi - beta * ln(rho/rho_0))
                               * cos(pitch) * cos(Xi(z))

           " B_z   = B_amp(rho,z) * sin(Xi(z))

            with
               " beta = 1/tan(pitch)
               " Xi(z) = Xi_0 * tanh(z / z_0)
               " B_amp(rho,z) = cst,
                                funct of sph. radial coord
                                funct of cyl. radial coord
               " pitch = pitch angle


    INPUT:
    ------
      - coord :  an (3,n)-array with coordinates at which the Bfield has to be
             evaluated. Default 'coord' format is cartesian
     **kwargs :
       - coord_format : string that specifies the coord format of the INPUT
                        coord. Can be cartesian, spherical or cylindrical
                        (the output are in cylindrical, what so ever)
                        Default is 'cartesian'
       - B_amp_type : string that specifies if the amplitude is to be
                      a constante ['cst'], a function of the cyl. radial
                      coordinate ['cyl'] or a function of the SPH. radial
                      coordinate ['sph']
                      Default = 'cyl'


      [model parameter]
       - B_0 : amplitude of the regular large-scale B field at the Sun [muG]
       - B_amp_param : additional parameter for the radial dependence
                       (if it is the case)
       - B_amp_type : string to specify the radial dependence
                      'cst', 'cyl','sph' for constant, cylindircal or spherical

       - pitch : pitch angle [rad] of the spiral arms
       - rho_0 : radial scale = dist. between the Sun and the Gal Centre [kpc]
       - Xi_0 :  tilt angle parameter in [rad]
       - z_0 :   height scale in [kpc]

       > getBFieldDefault('BSS') for default setting values of parameters
    
    
    OUTPUT:
    ------
    B_rho, B_z, B_phi : the components of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at
                        all locations specified in 'coord'

    Created on Jul 5 2017
    Based on MLS() written on Oct 24 2016

    @author: V.Pelgrims
    """

    #initialization of the MLS model parameters [Fauvet et al 2010]
    parameter = {'pitch':11.5*deg2rad,
                 'rho_0':8.0,
                 'Xi_0':25*deg2rad,
                 'z_0':1.0
                 }
    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value
        if key not in ('B_0','B_amp_param','pitch','Xi_0','z_0',
                       'rho_0','coord_format','B_amp_type'):
            raise ValueError('''
            Bad entry for optional argument. Key must be one either
            'B_0','B_amp_param':2.1,'pitch',,'Xi_0','z_0',
            'rho_0','coord_format','B_amp_type' ''')

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')


    #set the radial dependence of the amplitude, if appropriate
    B_amp = __B0_of_r(rho,z,**parameter)

    #sets the angular dependence of the amplitude
    #i.e. produces log spiral pattern in amplitude
    phi_rho = np.cos(phi -
                     1.0/np.tan(parameter['pitch']) *
                     np.log(rho / parameter['rho_0']) )
    #height dependence
    Xi_z = parameter['Xi_0'] * np.tanh(z / parameter['z_0'] )

    #cylindrical components of the magnetic vector field
    B_rho = B_amp * phi_rho * np.sin(parameter['pitch']) * np.cos(Xi_z)
    B_phi = B_amp * phi_rho * np.cos(parameter['pitch']) * np.cos(Xi_z)
    B_z = B_amp * np.sin(Xi_z)

    return np.array([B_rho,B_z,B_phi])


#
#
#


def QSS(coord,**kwargs):
    """
    FUNCTION
            === QSS : BiSymmetric Spiral Galactic magnetic field model ===

    It is almost the same model as the BSS model, except that the phase is
    multiplied by 2 to double the number of field line reversals

        QSS(coordinates,**kwargs{numerous parameters})

        " vec{B} = B_rho vec{u_rho} + B_phi vec{u_phi} + B_z vec{u_z}

        where           
           " B_rho = B_amp(rho,z) * cos(2*(phi - beta * ln(rho/rho_0)))
                               * sin(pitch) * cos(Xi(z))

           " B_phi = B_amp(rho,z) * cos(2*(phi - beta * ln(rho/rho_0)))
                               * cos(pitch) * cos(Xi(z))

           " B_z   = B_amp(rho,z) * sin(Xi(z))

            with
               " beta = 1/tan(pitch)
               " Xi(z) = Xi_0 * tanh(z / z_0)
               " B_amp(rho,z) = cst,
                                funct of sph. radial coord
                                funct of cyl. radial coord
               " pitch = pitch angle


    INPUT:
    ------
      - coord :  an (3,n)-array with coordinates at which the Bfield has to be
             evaluated. Default 'coord' format is cartesian
     **kwargs :
       - coord_format : string that specifies the coord format of the INPUT
                        coord. Can be cartesian, spherical or cylindrical
                        (the output are in cylindrical, what so ever)
                        Default is 'cartesian'
       - B_amp_type : string that specifies if the amplitude is to be
                      a constante ['cst'], a function of the cyl. radial
                      coordinate ['cyl'] or a function of the SPH. radial
                      coordinate ['sph']
                      Default = 'cyl'


      [model parameter]
       - B_0 : amplitude of the regular large-scale B field at the Sun [muG]
       - B_amp_param : additional parameter for the radial dependence
                       (if it is the case)
       - B_amp_type : string to specify the radial dependence
                      'cst', 'cyl','sph' for constant, cylindircal or spherical

       - pitch : pitch angle [rad] of the spiral arms
       - rho_0 : radial scale = dist. between the Sun and the Gal Centre [kpc]
       - Xi_0 :  tilt angle parameter in [rad]
       - z_0 :   height scale in [kpc]

       > getBFieldDefault('BSS') for default setting values of parameters
    
    
    OUTPUT:
    ------
    B_rho, B_z, B_phi : the component of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at
                        all locations specified in 'coord'

    Created on Apr 4 2018
    Based on BSS() written on Jul 5 2017

    @author: V.Pelgrims
    """

    #initialization of the MLS model parameters [Fauvet et al 2010]
    parameter = {'pitch':11.5*deg2rad,
                 'rho_0':8.0,
                 'Xi_0':25*deg2rad,
                 'z_0':1.0
                 }
    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value
        if key not in ('B_0','B_amp_param','pitch','Xi_0','z_0',
                       'rho_0','coord_format','B_amp_type'):
            raise ValueError('''
            Bad entry for optional argument. Key must be one either
            'B_0','B_amp_param':2.1,'pitch',,'Xi_0','z_0',
            'rho_0','coord_format','B_amp_type' ''')

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')


    #set the radial dependence of the amplitude, if appropriate
    B_amp = __B0_of_r(rho,z,**parameter)

    #set the angular dependence of the amplitude
    #i.e. produces log spiral pattern in amplitude
    phi_rho = np.cos(2*(phi -
                     1.0/np.tan(parameter['pitch']) *
                     np.log(rho / parameter['rho_0'])) )
    #height dependence
    Xi_z = parameter['Xi_0'] * np.tanh(z / parameter['z_0'] )

    #cylindrical components of the magnetic vector field
    B_rho = B_amp * phi_rho * np.sin(parameter['pitch']) * np.cos(Xi_z)
    B_phi = B_amp * phi_rho * np.cos(parameter['pitch']) * np.cos(Xi_z)
    B_z = B_amp * np.sin(Xi_z)

    return np.array([B_rho,B_z,B_phi])


#
#
#


def LSA(coord,**kwargs):
    """
    FUNCTION
            === LSA : Logarithmic Spiral Arm Galactic magnetic field model ===

    This is the WMAP Galactic magnetic field model.
    It is NOT strictly logarithmic (as the pitch angle varies)
    There is NO arm (in magnetic field norm/amplitude) and NO field reversal.

    >>> it is a ASS with a pitch angle that varies as a function of the
        radial coordinates.

        LSA(coordinates,**kwargs{numerous parameters})

        " vec{B} = B_rho vec{u_rho} + B_phi vec{u_phi} + B_z vec{u_z}
        where           
           " B_rho = B_amp(rho,z) * sin(psi(rho)) * cos(Xi(z))
           " B_phi = B_amp(rho,z) * cos(psi(rho)) * cos(Xi(z))
           " B_z   = B_amp(rho,z) * sin(Xi(z))
            with
               " psi_rho = psi_0 + psi_1 * ln(rho / rho_0)
               " Xi(z) = Xi_0 * tanh(z / z_0)
               " B_amp(rho,z) = cst,
                                funct of sph. radial coord
                                funct of cyl. radial coord


    INPUT:
    ------
      - coord :  an (3,n)-array with coordinates at which the Bfield has to be
             evaluated. Default 'coord' format is cartesian
     **kwargs :
        - coord_format : string that specifies the coord format of the INPUT
                         coord. Can be cartesian, spherical or cylindrical
                         (the output are in cylindrical, what so ever)
                         Default is 'cartesian'
        - B_amp_type : string that specifies if the amplitude is to be
                       a constante ['cst'], a function of the cyl. radial
                       coordinate ['cyl'] or a function of the SPH. radial
                       coordinate ['sph']
                       Default = 'cst'
      [model parameter]
       - B_0 : amplitude of the regular large-scale B field at the Sun [muG]
       - B_amp_param : additional parameter for the radial dependence
                       (if it is the case)

       - psi_0 : cst param of the pitch angle [rad] of the spiral arms
       - psi_1 : amplitude of the radial dep of the pitch angle [rad]
       - rho_0 : radial scale = dist. between the Sun and the Gal Centre [kpc]
       - Xi_0 :  tilt angle parameter in [rad]
       - z_0 :   height scale in [kpc]

       > getBFieldDefault('LSA') for default setting values of parameters

    
    OUTPUT:
    ------
    B_rho, B_z, B_phi : the component of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at
                        all locations specified in 'coord'

    Created on Jul 5 2017
    Based on BSS

    @author: V.Pelgrims
    """

    #initialization of the LSA model parameters [Page et al 2010]
    parameter = {'psi_0':27.0*deg2rad,
                 'psi_1':0.9*deg2rad,
                 'rho_0':8.0,
                 'Xi_0':25*deg2rad,
                 'z_0':1.0,
                 'B_amp_type':'cst'
                 }
    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value
        if key not in ('B_0','B_amp_param','psi_0','psi_1','Xi_0','z_0',
                       'rho_0','coord_format','B_amp_type'):
            raise ValueError('''
            Bad entry for optional argument. Key must be one either
            'B_0','B_amp_param','psi_0','psi_1','Xi_0','z_0',
            'rho_0','coord_format','B_amp_type' ''')

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')


    #set the radial dependence of the amplitude, if appropriate
    B_amp = __B0_of_r(rho,z,**parameter)

    #set the radial dependence of the pitch angle
    psi_rho = parameter['psi_0'] + (parameter['psi_1']
                                    * np.log(rho / parameter['rho_0']))
    #height dependence
    Xi_z = parameter['Xi_0'] * np.tanh(z / parameter['z_0'])

    #cylindrical components of the magnetic vector field
    B_rho = B_amp * np.sin(psi_rho) * np.cos(Xi_z)
    B_phi = B_amp * np.cos(psi_rho) * np.cos(Xi_z)
    B_z = B_amp * np.sin(Xi_z)

    return np.array([B_rho,B_z,B_phi])


#
#
#


def CCR(coord,**kwargs):
    """
    FUNCTION
            === CCR : Concentric Circular Galactic magnetic field model ===

        CCR(coordinates,**kwargs{numerous parameters})

        " vec{B} = B_rho vec{u_rho} + B_phi vec{u_phi} + B_z vec{u_z}
        where           
           " B_rho = 0
           " B_phi = B_amp(rho,z) * sin(pi*(rho - rho_0 - Dr)/omega)
                                    /sin(pi*Dr/omega)
                                  * cos(Xi(z))
           " B_z   = B_amp(rho,z) * sin(Xi(z))
            with
               " omega = (radial) periode of reversal
               " Dr = distance to the first reversal
               " Xi(z) = Xi_0 * tanh(z / z_0)
               " B_amp(rho,z) = cst,
                                funct of sph. radial coord
                                funct of cyl. radial coord


    INPUT:
    ------
      - coord :  an (3,n)-array with coordinates at which the Bfield has to be
             evaluated. Default 'coord' format is cartesian
     **kwargs :
        - coord_format : string that specifies the coord format of the INPUT
                         coord. Can be cartesian, spherical or cylindrical
                         (the output are in cylindrical, what so ever)
                         Default is 'cartesian'
        - B_amp_type : string that specifies if the amplitude is to be
                       a constante ['cst'], a function of the cyl. radial
                       coordinate ['cyl'] or a function of the SPH. radial
                       coordinate ['sph']
                       Default = 'cst'
      [model parameter]
       - B_0 : amplitude of the regular large-scale B field at the Sun [muG]
       - B_amp_param : additional parameter for the radial dependence
                       (if it is the case)

       - omega : (radial) periode of reversa [kpc]l
       - Dr : distance to the first reversal [kpc]
       - rho_0 : radial scale = dist. between the Sun and the Gal Centre [kpc]
       - z_0 :   height scale in [kpc]

       > getBFieldDefault('CCR') for default setting values of parameters

    
    OUTPUT:
    ------
    B_rho, B_z, B_phi : the component of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at
                        all locations specified in 'coord'

    Created on Jul 5 2017
    Based on BSS

    @author: V.Pelgrims
    """

    #initialization of the CCR model parameters [Rand & Kulkarni 1989]
    parameter = {'omega':3.1,
                 'Dr':0.6,
                 'rho_0':8.0,
                 'B_0':1.6,
                 'z_0':1.0,
                 'Xi_0':25*deg2rad,
                 'B_amp_type':'cst'
                 }
    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value
        if key not in ('B_0','B_amp_param','B_amp_type',
                       'omega','Dr','rho_0','z_0',
                       'Xi_0','coord_format'):
            raise ValueError('''
            Bad entry for optional argument. Key must be one either
            'B_0','B_amp_param','B_amp_type','omega','Dr','Xi_0','z_0',
            'rho_0','coord_format','B_amp_type' ''')

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')


    #set the radial dependence of the amplitude, if appropriate
    B_amp = __B0_of_r(rho,z,**parameter)

    #height dependence
    Xi_z = parameter['Xi_0'] * np.tanh(z / parameter['z_0'])

    #cylindrical components of the magnetic vector field
    B_rho = 0*Xi_z
    B_phi = B_amp * (np.sin(np.pi*(rho - parameter['rho_0']
                                   + parameter['Dr'])/parameter['omega'])
                     /np.sin(np.pi*parameter['Dr']/parameter['omega'])
                     * np.cos(Xi_z))
    B_z = B_amp * np.sin(Xi_z)

    return np.array([B_rho,B_z,B_phi])


#
#
#


def BT(coord,**kwargs):
    """
    FUNCTION
            === BT : BiToroidal Galactic magnetic field model ===

        BT(coordinates,**kwargs{numerous parameters})

    Bitoroidal model adapted from Ruiz-Granados et al. 2010

        " vec{B} = B_rho vec{u_rho} + B_phi vec{u_phi} + B_z vec{u_z}
        where           
           " B_rho = 0
           " B_phi = B_amp(rho,z) * atan(z/sig1)
                                  * exp(-z**2 / (2sig2**2))
                                  * rho/rho_0 * exp(-(rho - rho_0)/rho_0)
           " B_z   = Bz_0
            with
               " B_amp(rho,z) = cst [default]
                                funct of sph. radial coord
                                funct of cyl. radial coord


    INPUT:
    ------
      - coord :  an (3,n)-array with coordinates at which the Bfield has to be
                 evaluated. Default 'coord' format is cartesian
     **kwargs :
        - coord_format : string that specifies the coord format of the INPUT
                         coord. Can be cartesian, spherical or cylindrical
                         (the output are in cylindrical, what so ever)
                         Default is 'cartesian'
        - B_amp_type : string that specifies if the amplitude is to be
                       a constante ['cst'], a function of the cyl. radial
                       coordinate ['cyl'] or a function of the SPH. radial
                       coordinate ['sph']
                       Default = 'cyl'
      [model parameter]
       - B_0 : amplitude of the regular large-scale B field at the Sun [muG]
       - B_amp_param : additional parameter for the radial dependence
                       (if it is the case)

       - rho_0 : radial scale = dist. between the Sun and the Gal Centre [kpc]
       - sig1 : first height scale in [kpc]
       - sig2 : first height scale in [kpc]
       - Bz_0 : constant value [muG]

       > getBFieldDefault('CCR') for default setting values of parameters

    
    OUTPUT:
    ------
    B_rho, B_z, B_phi : the component of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at
                        all locations specified in 'coord'

    Created on Jul 5 2017
    Based on BSS

    @author: V.Pelgrims
    """

    #initialization of the BT model parameters [Ruiz-Granados et al. 2010]
    parameter = {'rho_0':8.0,
                 'B_0':3.0,
                 'sig1':1.0,
                 'sig2':1.0,
                 'Bz_0':0.2
                 }
    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value
        if key not in ('B_0',
                       'rho_0','sig1','sig2',
                       'Bz_0','coord_format'):
            raise ValueError('''
            Bad entry for optional argument. Key must be one either
            'B_0','sig1','sig2','rho_0','coord_format' ''')

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')


    #radial dependence
    B_of_rho = parameter['B_0'] * (rho / parameter['rho_0']
                                   * np.exp(-(rho - parameter['rho_0'])
                                            /parameter['rho_0']))

    #height dependence
    B_of_z = (np.arctan(z/parameter['sig1'])
              * np.exp(- z**2 / (2*parameter['sig2']**2)))

    #cylindrical components of the magnetic vector field
    B_rho = 0*B_of_z
    B_phi = B_of_rho * B_of_z
    B_z = parameter['Bz_0'] + 0*B_of_z

    return np.array([B_rho,B_z,B_phi])


#
#
#


def __B0_of_r(rho,z,**kwargs):
    """
    Internal function
        Intended to modulate the magnetic field amplitude by a function
        of the radial (cyl or sph) coordinate

    If constant:     B_amp = B_0     for all rho,z
    If cylindrical:  B_amp = B_0 * 1/(1 + rho/B_amp_param)
    If spherical:    B_amp = B_0 * exp(-(r - rho_0)/B_amp_param)
    B_amp = B_0    if constant
            B_0 * 1/(1 + rho/rho_0)    if cylindrical
            B_0 * exp(-(r - rho_0)/B_amp_param) if spherical

    B_amp is automatically normalized such that B_amp(at sun) = B_sun meant
    to be given by the 'B_0' param in kwargs.

    INPUT:
    ------
      - rho : cylindircal radial coordinate
      - z : height coordinates

      **kwargs : containing the information to build the wanted function
         - B_0 : an overall amplitude
         - B_amp_param : an additional parameter for the radial function
         - B_amp_type : string to specify the fonctional form. Should be
                        'cst','cyl' or 'sph'
         - rho_0 : is supposed to contained the dist. btw the Sun and the GC


    OUTPUT:
    ------
      - B_amp : the field amplitude at each location specified by rho,z


    Creation date : Jul 5 2017
    @author: V.Pelgrims
    """

    param = {'B_0':2.1,
             'B_amp_type':'cyl',
             'B_amp_param':8.0,
             'rho_0':8.0
             }

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        param[key] = value
    
    ###
    if param['B_amp_type'] == 'cst':
        B_amp = param['B_0']
    elif param['B_amp_type'] == 'cyl':
        B_0 = param['B_0'] * (1. + param['rho_0']/param['B_amp_param'])
        B_amp = B_0 * 1./(1. + rho/param['B_amp_param'])
    elif param['B_amp_type'] == 'sph':
        B_amp = (param['B_0']
                 * np.exp(-((rho**2 + z**2)**.5 - param['rho_0'])
                          /param['B_amp_param']))
    else:
        raise ValueError('''
        Bad entry for optional argument 'B_amp_type'.
        Key must be one either: 'cst', 'sph' or 'cyl' ''')
    
    return B_amp


#
#
#


def ARM4(coord,**kwargs):
    """
    FUNCTION
        === ARM4 : 4 log Sprial Arms Galactic magnetic field model ===

        ARM4(coordinates,**kwargs{numerous parameters})

    'ARM' model adapted from literature
     The arms are identical with a 90 degree rotation

        " vec{B} = B_rho vec{u_rho} + B_phi vec{u_phi} + B_z vec{u_z}
        where

        "Brho = B(rho) * Bspiral(rho,phi) * B(z) sin(pitch)
        "Bphi = B(rho) * Bspiral(rho,phi) * B(z) cos(pitch)
        "Bz = 0

            with
            " B(rho) = B_0 * exp(-|rho-rho_cusp|/sig_rho)
            " B(z) = 1./(cosh(z/sig_z))**2
            " Bspiral = #if B_s_type='rho'
                         sum_i{exp(-|rho - rho_S_i(phi)|/rho_0) }
                        #esle if B_s_type = 'phi' [default]
                         sum_i{exp(-(phi - phi_S_i(rho))**2)/(2(phi_0**2))}
               with
            " rho_S_i(phi) = rho0_i * exp(phi*tan(pitch))
            " phi_S_i(rho) = ln(rho/rho0_i)/tan(pitch)
                where
                   phi0_i = phi00 + i*pi/2                      #imposed
                   rho0_i = rho0_i * exp(phi0_i * tan(pitch)))  #computed
                   i = 1,2,3,4


    INPUT:
    ------
      - coord :  an (3,n)-array with coordinates at which the Bfield has to be
                 evaluated. Default 'coord' format is cartesian
     **kwargs :
        - coord_format : string that specifies the coord format of the INPUT
                         coord. Can be cartesian, spherical or cylindrical
                         (the output are in cylindrical, what so ever)
                         Default is 'cartesian'
        - B_amp_type : string that specifies if the amplitude is to be
                       a constante ['cst'], a function of the cyl. radial
                       coordinate ['cyl'] or a function of the SPH. radial
                       coordinate ['sph']
                       Default = 'cyl'
      [model parameter]
       - B_0 : amplitude of the regular large-scale B field at the Sun [muG]
       - B_of_rho_type : function form of the overal radial dependence
       - B_of_rho_scale : [kpc] paramete of the function of the radial
                          dependence
       - rho_cusp : [kpc] gives the radius of maximum field amplitude
       - pitch : [rad] pitch angle of the spiral (same for all)
       - phi_00 : [rad] angular start of the 4th arm [default = 10degree]
       - rho_00 : [kpc] 1 [default] spiral arm radial parameter
       - amp_s_i : tuple of 4 amplitude for the field strength of the arm
                   they are meant to be relative amplitudes.
       - B_s_type : defines the amplitude dependence of the arms. Can be
                    set either with radial ('rho') or with angular ('phi')
                    dependence. (see Bspiral above)
       - B_s_param : parameter of the function form for the arm amplitude
       - B_of_z_type : functional form of the height dependence (cosh)
       - B_of_z_scale : height scale [default = 1.]
       - rho_min : [kpc] minimum radial scale from which to compute the
                    magnetic field


       > getBFieldDefault('ARM4') for default setting values of parameters

    
    OUTPUT:
    ------
    B_rho, B_z, B_phi : the component of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at
                        all locations specified in 'coord'


    Created on Jul 6 2017
    @author: V.Pelgrims
    """


    #initialization
    parameter = {'B_0':2.1,
                 'B_of_rho_type':'d_exp',
                 'B_of_rho_scale':8.0,
                 'rho_cusp':.7,
                 'pitch':11.5*deg2rad,
                 'phi_00':10.*deg2rad,
                 'rho_00':1.,
                 'amp_s_i':[1.,1.,1.,1.],
                 'B_s_type':'phi',
                 'B_s_param':15*deg2rad,
                 'B_of_z_type':'cosh',
                 'B_of_z_scale':1.0,
                 'rho_min':0.0}

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')

    B_of_rho = OneParamFunc(rho-parameter['rho_cusp'],
                            parameter['B_of_rho_scale'],
                            parameter['B_of_rho_type'])

    
    ### SPIRAL FEATURE
    phi0_i = np.mod(np.asarray([parameter['phi_00']+(i+1)*np.pi/2.
                                for i in range(4)]),2*pi)
    rho0_i = parameter['rho_00'] * np.exp(phi0_i*np.tan(parameter['pitch']))

    ### IMPORTANT to not miss part of contribution...
    #Should take into account the fact that +10deg and -10deg
    #are separated by 20deg and not 340deg.
    phi_rho = np.asarray([np.log(rho/rho0_i[i])/np.tan(parameter['pitch'])
                          for i in range(4)])
    diff_phi = np.arccos(np.cos(phi - phi_rho))

    if parameter['B_s_type'] == 'phi':
        #amplitude follows spiral through angular parameterization
        B_spiral = np.sum([parameter['amp_s_i'][i]
                           *OneParamFunc(diff_phi[i],
                                         parameter['B_s_param'],
                                         'Gauss') for i in range(4)],axis=0)
        #
    elif parameter['B_s_type'] == 'rho':
        #amplitude follows spiral through radial parameterization
        B_spiral = np.sum([parameter['amp_s_i'][i]
                           *OneParamFunc(rho-rho0_i[i]
                                         *np.exp((phi_rho[i]+diff_phi[i])
                                                 *np.tan(parameter['pitch'])),
                                         parameter['B_s_param'],
                                         'd_exp') for i in range(4)],axis=0)
    else:
        raise ValueError('''
        Bad entry for optional argument 'B_s_type'.
        Key must be one either: 'phi' or 'rho' ''')

    #height dependence
    B_of_z = OneParamFunc(z,
                          parameter['B_of_z_scale'],
                          parameter['B_of_z_type'])
    B_of_z[rho <= parameter['rho_min']] *= 0

    B_amp = parameter['B_0'] * B_of_rho * B_spiral * B_of_z
    #cylindrical components of the magnetic vector field
    B_rho = B_amp * np.sin(parameter['pitch'])
    B_phi = B_amp * np.cos(parameter['pitch'])
    B_z = 0*B_of_z

    return np.array([B_rho,B_z,B_phi])


#
#
#


def RING(coord,**kwargs):
    """
    FUNCTION
        === RING : ring Galactic magnetic field model ===

    RING(coord,list_of_parameter)

    INPUT:
     - coord.
     model param
    - amplitude
    - inner radius
    - outer radius

    Created on Jul 7 2017
    @author: V.Pelgrims
    """

    #initialization
    parameter = {'B_0':2.1,
                 'rho_in':2.,
                 'rho_ex':5.
                 }

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')

    Brho = 0*z
    Bphi = parameter['B_0'] + 0*z
    Bz = 0*z

    Bphi[rho <= parameter['rho_in']] *=0
    Bphi[rho >= parameter['rho_ex']] *=0

    return np.asarray([Brho,Bz,Bphi])

#
#
#


def COMPO(coord,**list_of_field):
    """
    FUNCTION
       === COMPO : composite Galactic magnetic field model ===

    COMPO(coord,**kwargs({'model_type':{model_param},
                          'model_type2':{model_param2},
                           ...}))

    Makes use of BFielder for briefty

    INPUT:
    ------
     - coord : the coordinates where t compute the things

     - B_sun : [muG] the field strength at sun location.

     - list_of_field_components : dictionary with (key,value)-couple.

           - 'key' is a string that must be either:
                ASS
                BSS
                LSA
                ARM4
                BT
                CCR
                RING

           - 'value' is a dictionary that contains the parameters of the
                     corresponding profile in 'key' if default values are
                     not to be used.

             >>>  See getDusterDefault('key') for default values <<<

      *Remarks*
      The relative amplitude of each component should be given as the
      B_0 keyed argument. A automatic renormalization is then performed to
      set the field strength at the Sun location to be B_sun.
      >>> B = B * (|B(8,0,pi)| / B_sun)


    OUTPUT:
    ------
    B_rho, B_z, B_phi : the component of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at
                        all locations specified in 'coord'

    Created on Jul 7 2017
    @auhtor: V.Pelgrims
    """

    #initialization
    if 'B_sun' not in list_of_field.keys():
        B_sun = 2.1
    else:
        B_sun = list_of_field['B_sun']
        list_of_field.pop('B_sun')

    n = len(coord[0])
    Bfield = np.zeros((3,n))
    #B_at_sun = 0
    B_at_sun = np.zeros((3,2))
    
    #if sun position is not at default position
    if 'r_sun' in list_of_field.keys():
        xyz_sun = np.asarray([-list_of_field['r_sun'],0,0])
        list_of_field.pop('r_sun')
    elif 'XYZ_sun' in list_of_field.keys():
        xyz_sun = list_of_field['XYZ_sun']
        list_of_field.pop('XYZ_sun')
    else:
        xyz_sun = np.asarray([-8.,0.,0.])
        #small trick to make the last line running
        xyz_sun = np.vstack((xyz_sun,xyz_sun)).T

    print(list_of_field)

    for key,value in list_of_field.items():
        print(''' - ''', key)
        Bfield += BFielder(coord,{key:value})
        B_at_sun += BFielder(xyz_sun,{key:value})
    #trick to make the last line work
    B_at_sun = B_at_sun[:,0]

    #normalization
    Bfield *= B_sun / np.sum(B_at_sun**2,axis=0)**.5

    return Bfield

#
#
#


def OneParamFunc(var,param,form):
    """
    FUNCTION

        ===  OneParamFunc: One parameter functions ===
    
    For a given set of variable values, this function evaluates the
    values of a one-parameter function, with the specified form, given
    the value of the parameter.

        " f_of_var =
              func(var,param)
                    = exp(-var**2 / (2*param**2))     Gaussian
                    = 1 / ( cosh(-var / param) )**2   Hyperbolic Secant
                    = exp(- |var| / param)            Decreasing Exponential
                    = |var| ** param                  Power Law
                    = max(cos(pi * var / 2*param),0)  Cosine


    INPUT:
    ------
     - var : (n,)-array with values of the variables for which the
             function as to be evaluated

     - param : the parameter of the function. Given the function to
               be evaluated, this paramter can be the slope if the
               power law, the standard deviation of the Gaussian etc.

     - form : string that determine the fonctionl form to be computed.
              It determines if the function is:
              - Gaussian : 'Gaussian', 'Gauss', 'G'
              - Hyperbolic Secant : 'HyperbolicSecant','HS', 'H', 'cosh'
              - Decreasing Exponantial : 'Decreasing_Exp', 'DE', 'exp',
                                         'd_exp'
              - Power Law : 'PoweLaw', 'PL', 'power'
              - Cosine : 'Cosine', 'cos', 'C'


    OUTPUT:
    -------
     - f(var) : (n,)-array with evaluations of the function at var values.


    Created on Sep 27, 2016
    Copied from GalacticProfile.py module
    @author: V.Pelgrims
    """

    if (form == 'Gaussian' or form == 'Gauss' or form == 'G'):
        f_of_var = np.exp(-(var*var) / (2 * param*param))
    elif (form == "Decreasing_Exp" or form == 'd_exp' or
          form == 'exp' or form == 'DE'):
        f_of_var = np.exp( - np.abs(var) / param )
    elif (form == 'HyperbolicSecant' or form == 'H' or
          form == 'HS' or form == 'cosh'):
        f_of_var = 1 / (np.cosh(- var / param) * np.cosh(- var / param))
    elif (form == 'PowerLaw' or form == 'PL' or form == 'power'):            
        f_of_var = np.abs(var) ** param
    elif (form == 'Cosine' or form == 'C' or form == 'cos'):
        f_of_var = np.maximum(np.cos(pi*var / (2*param)), 0.0)
    else:
        print( '''
    The one-parameter functional referenced by %s is not recognized.
    Try again!'''%(form))
        return

    return f_of_var


#
#
#


def WMAP(coord,**kwargs):
    """
    FUNCTION

            ===  WMAP : Galactic magnetic field of WMAP paper  ===
    
    WMAP(coordinates,**kwargs{model parameters etc.})
    
    Implementation of the Galactic magnetic field model presented in
    [Page et al 2007]. It is call the Logarithmic Spiral Arm (LSA) model.

    !!! NOTE THAT we add the possibility for a regular large-scale B field
    component which is not implemented originally in Page et al.
    
        " vec{B} = B_rho vec{u_rho} + B_z vec{u_z} + B_phi vec{u_phi}
            where
           
           " B_rho = B_reg(r) * sin(psi(rho)) * cos(Xi(z))
           " B_phi = B_reg(r) * cos(psi(rho)) * cos(Xi(z))
           " B_z   = B_reg(r) * sin(Xi(z))
           with
               " psi(rho) = psi_0 + psi_1 * ln(rho / rho_0)
               " Xi(z) = Xi_0 * tanh(z / z_0)
               " B_reg = B_0 * np.exp( - (r - R_Sun) / R_B )
                   [exponential distribution with spherical symm.]

                   r = spherical radial coord.
                   rho = cylindrical radial coord.
                   z = cylindrical height


    INPUT:
    ------
     - coord :  an (3,n)-array with Galactic coordinates at which the
                Bfield has to be evaluated. Default format is cylindrical.

     **kwargs :
       - coord_format : string that specifies the coord format of the INPUT
                        coordinates. Can be cartesian, spherical or cylindrical
                       (the output are in cylindrical, what so ever)
                       Default is 'cartesian'
       - withBregular : boolean that chooses to add the regular large-scale B
                        field with exponential distrib (True) or to stuck to
                        Page et al. model (False).
                        Default is True
                        If False, then Breg = B_0 at all locations!

      [model parameter]
       - B_0 : amplitude of the regular large-scale B field at the Sun [muG]
       - r_sun : distance between the Sun and the Gal Centre [kpc]
       - R_B : radial scale of the regular large-scale B field [kpc]
    
       - psi_0 : openning angle [rad] of the spiral arms [Default = 27 deg]
       - psi_1 : some parameter :p in [rad]
       - rho_0 : a radial scale, rho_0 = R_Sun except specified
       - Xi_0 :  tilt parameter in [rad]
       - z_0 :   a vertical scale in [kpc]

       > getBFieldDefault('WMAP') for default setting values of parameters

    
    OUTPUT:
    ------
    B_rho, B_z, B_phi : the component of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at all
                        locations specified in 'coord'


    Created on Oct 24 2016
    Based on BField_WMAP.py written on Jun 23 2016

    @author: V.Pelgrims
    """


    #initialization of default LSA model parameters [Page et al 2007]
    #taking care of their ERRATUM
    parameter = {'B_0':2.1,
                 'psi_0':27.0*deg2rad,
                 'psi_1':0.9*deg2rad,
                 'rho_0':8.0,
                 'Xi_0':25.0*deg2rad,
                 'z_0':1.0,
                 'r_sun':8.0,
                 'R_B':8.5}
    #the two last parameters are default initialization of regular
    #normalization in case it is called

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value
        if key not in ('B_0','psi_0','psi_1','rho_0','Xi_0','z_0',
                       'r_sun','R_B','coord_format','withBRegular'):
            raise ValueError('''
            Bad entry for optional argument. Key must be one either
            'B_0', 'psi_0', 'psi_1', 'rho_0', 'Xi_0', 'z_0', 'r_sun',
            'R_B', 'coord_format' or'withBRegular' ''')

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')

    #with or without decreasing B field magnitude according to Galactic
    #centre distance
    if 'withBRegular' in kwargs.keys() and kwargs['withBRegular']:
        #compute the spherical radial coordinate and evaluate the
        #large-scale amplitude of the regular field component
        Breg = B_Regular((rho*rho + z*z)**.5,
                        parameter['B_0'],
                        parameter['r_sun'],
                        parameter['R_B'])
    else:
        Breg = parameter['B_0']


    #Building the model arguments
    #LSA feature : polar coordinates depend one another
    phi_rho = (parameter['psi_0'] +
               parameter['psi_1'] *
               np.log(rho / parameter['rho_0']))
    #height dependence
    Xi_z = parameter['Xi_0'] * np.tanh(z / parameter['z_0'])

    #cylindrical component of the magnetic vector field
    B_rho = Breg * np.sin(phi_rho) * np.cos(Xi_z)
    B_phi = Breg * np.cos(phi_rho) * np.cos(Xi_z)
    B_z = Breg * np.sin(Xi_z)

    return np.array([B_rho,B_z,B_phi])


#
#
#


def MLS(coord,**kwargs):
    """
    FUNCTION
            === MLS : MODIFIED LOGARITHMIC SPIRAL BFIELD MODEL ===

    MLS(coordinates,**kwargs{model_parameters etc.})
    
    Implementation of the Galactic magnetic field as presented in
    Fauvet et al 2010.
    It is call the Modified Logarithmic Spiral (MLS) model and it is
    based on the LSA model of Page et al 2007, so is the code.
    
        " vec{B} = B_rho vec{u_rho} + B_z vec{u_z} + B_phi vec{u_phi}
            
        where
           
           " B_rho = B_reg(r) * cos(phi + beta)
                               * ln(rho/rho_0) * sin(p) * cos(Xi)

           " B_phi = - B_reg(r) * cos(phi + beta)
                               * ln(rho/rho_0) * cos(p) * cos(Xi)

           " B_z   = B_reg(r) * sin(Xi)

            with
                " beta = 1/tan(p)
                " Xi(z) = Xi_0 * (z / z_0)
                " B_reg = B_0 * np.exp( - (r - R_Sun) / R_B )
                    [exponential distribution with spherical symm.]
                
                    r = spherical radial coord.
                    rho = cylindrical radial coord.
                    z = cylindrical height
                    phi = cylindrical angular coord.


    INPUT:
    ------
     - coord :  an (3,n)-array with coordinates at which the Bfield has to be
             evaluated. Default 'coord' format is cartesian
     **kwargs :
       - coord_format : string that specifies the coord format of the INPUT
                        coordinates. Can be cartesian, spherical or cylindrical
                       (the output are in cylindrical, what so ever)
                       Default is 'cartesian'
       - withBregular : boolean that chooses to add the regular large-scale B
                        field with exponential distrib (True) or to stuck to
                        Page et al. model (False).
                        Default is True
                        If False, then Breg = B_0 at all locations!

      [model parameter]
       - B_0 : amplitude of the regular large-scale B field at the Sun [muG]
       - R_Sun : distance between the Sun and the Gal Centre [kpc]
       - R_B : radial scale of the regular large-scale B field [kpc]
    
       - pitch : pitch angle [deg] between spiral arms
       - rho_0 : a radial scale, rho_0 = R_Sun except specified
       - Xi_0 :  tilt parameter in [rad]
       - z_0 :   a vertical scale in [kpc]

       > getBFieldDefault('MLS') for default setting values of parameters
    
    
    OUTPUT:
    ------
    B_rho, B_z, B_phi : the component of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at all
                        locations specified in 'coord'

    Created on Oct 24 2016
    Based on BField_MLS.py written on Jun 23 2016

    @author: V.Pelgrims
    """

    #initialization of the MLS model parameters [Fauvet et al 2010]
    parameter = {'B_0':2.1,
                 'pitch':-30.0*deg2rad,
                 'rho_0':8.0,
                 'Xi_0':25*deg2rad,
                 'z_0':1.0,
                 'r_sun':8.0,
                 'R_B':8.5}
    #the two last parameters are default initialization of regular
    #normalization in case it is called

    #set MLS parameters to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value
        if key not in ('B_0','pitch','rho_0','Xi_0','z_0',
                       'r_sun','R_B','coord_format','withBRegular'):
            raise ValueError('''
            Bad entry for optional argument. Key must be one either
            'B_0', 'psi_0', 'psi_1', 'rho_0', 'Xi_0', 'z_0', 'r_sun',
            'R_B', 'coord_format' or'withBRegular' ''')

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')

    #with or without decreasing B field magnitude according to Galactic
    #centre distance
    if 'withBRegular' in kwargs.keys() and kwargs['withBRegular']:
        #compute the spherical radial coordinate and evaluate the
        #large-scale amplitude of the regular field component
        Breg = B_Regular((rho*rho + z*z)**.5,
                         parameter['B_0'],
                         parameter['r_sun'],
                         parameter['R_B'])
    else:
        Breg = parameter['B_0']


    #Build the model arguments
    #LSA feature : polar coordinates depend one another
    phi_rho = np.cos(phi +
                     1.0/np.tan(parameter['pitch']) *
                     np.log(rho / parameter['rho_0']) )
    #height dependence
    Xi_z = parameter['Xi_0'] * np.tanh(z / parameter['z_0'] )

    #cylindrical components of the magnetic vector field
    B_rho = Breg * phi_rho * np.sin(parameter['pitch']) * np.cos(Xi_z)
    B_phi = - Breg * phi_rho * np.cos(parameter['pitch']) * np.cos(Xi_z)
    B_z = Breg * np.sin(Xi_z)

    return np.array([B_rho,B_z,B_phi])


#
#
#


def B_Regular(radial_coord,amp_atSun,r_sun,radial_scale):
    """
    FUNCTION
            ===  B_REGULAR : REGULAR MAGNETIC FIELD ===

    Implementation of the large-scale regular magnetic field with
    amplitude having spherical symmetry about the Galactic Centre.
    The B field component model is described in Han et al 2006:

          "  A = A0 * np.exp(-(radial_coord - r_sun)/radial_scale)


    INPUT:
    ------
     - radial_coord : (npts,)-array of radial coordinates [kpc] (distance
                      to the Galactic Centre) at which the Bfield has to be
                      computed

     - amp_atSun : [uG] amplitude of the Bfield at Sun location

     - r_sun : [kpc] Sun distance to the Galactic Center

     - radial_scale : [kpc] radial scale of the model

    OUTPUT:
    ------
     - amplitude : (npts)-array with amplitude of the magnetic field at all
                   input location.
                   Unit is the same as the one in amp_atSun: [uG]

    Created on Oct 24 2016
    @author: V.Pelgrims
    """


    amplitude = amp_atSun * np.exp(- ( radial_coord - r_sun )
                                   / radial_scale )

    return amplitude


#
#
#


def JAFFE(coord,**kwargs):
    """
    FUNCTION
        === JAFFE : Jaffe et al. Galactic magnetic field model ===

        JAFFE(coordinates,**kwargs{numerous parameters})

    This is a composite model with inner 'ring' field + spiral pattern
    with non trivial definition of the field strenght in the arms.
    The arms are identical with a 90 degree rotation
    There is no out-of-plane component.


        " vec{B} = B_rho vec{u_rho} + B_phi vec{u_phi} + B_z vec{u_z}
        where

        "Brho = B(rho) * Bspiral(rho,phi) * sin(pitch) if rho>rho_ring
                0 if rho<= rho_ring
        "Bphi = B(rho) * Bspiral(rho,phi) * cos(pitch) if rho>rho_ring
                B(rho) B(z) if rho<=rho_ring
        "Bz = 0 everywhere

            with
            " B(rho) = B_0 * exp(-rho/rho_scale)
            " B(z) = B0_disk/(cosh(z/h_disk))**2 + B0_halo/(cosh(z/h_halo))**2
            " Bspiral = sum_i{Ai * c(rho) * exp(-|rho - rho_S_i(phi)|/d0(rho)}
                        
               with
            " d0(rho) = d0 / (c(rho) * B(rho))
            " c(rho) = C0                   if rho <= rho_cc
                       C0 (rho/rho_cc)**-3  if rho > rho_cc
            " rho_S_i(phi) = rho0_S * exp((phi - phi0_i)*tan(pitch))
                where
                   phi0_i = phi00 + i*pi/2                      #imposed
                   i = 1,2,3,4


    INPUT:
    ------
      - coord :  an (3,n)-array with coordinates at which the Bfield has to be
                 evaluated. Default 'coord' format is cartesian
     **kwargs :
        - coord_format : string that specifies the coord format of the INPUT
                         coord. Can be cartesian, spherical or cylindrical
                         (the output are in cylindrical, what so ever)
                         Default is 'cartesian'
        - B_amp_type : string that specifies if the amplitude is to be
                       a constante ['cst'], a function of the cyl. radial
                       coordinate ['cyl'] or a function of the SPH. radial
                       coordinate ['sph']
                       Default = 'cyl'
      [model parameter]
       - B_0 : amplitude of the regular large-scale B field at the Sun [muG]
       - rho_scale : [kpc] parameter of the function of the radial dependence
       - C0 : Constant. Fix amplitude value of the spiral arm pattern
       - rho_cc : [kpc] amplitude suppression radius
       - d0 : distance scale for the exp. decrease of arm amplitude
       - pitch : [rad] pitch angle of the spiral (same for all) (11.5deg)
       - phi_00 : [rad] angular start of the 4th arm [default = 10degree]
       - rho0_S : [kpc] 1 [default] spiral arm radial parameter
       - A_i : tuple of 4 amplitudes for the field strength of the arms
               they are meant to be relative amplitudes.

       - B0_disk : amplitude of the height dep. for the disk INSIDE the ring
       - B0_halo : amplitude of the height dep. for the halo INSIDE the ring
       - h_disk : height-scale for the height dep. for the disk INSIDE the ring
       - h_halo : height-scale for the height dep. for the halo INSIDE the ring
       - h_c : height-scale for the height dep. for the disk OUTSIDE the ring
       - rho_ring_in : [kpc] ring radius below which there is no field
       - rho_ring_ex : [kpc] ring radius above which the field is log spiral
                        and bdelow which the field is azimuthal


       > getBFieldDefault('JAFFE') for default setting values of parameters

    
    OUTPUT:
    ------
    B_rho, B_z, B_phi : the component of the vectorial B field in CYLINDRICAL
                        coordinate system centred on the Galactic Centre at
                        all locations specified in 'coord'


    Created on Jul 10 2017
    @author: V.Pelgrims
    """


    #initialization
    parameter = {'B_0':1.,
                 'rho_scale':20.0,
                 'C0':2.5,
                 'rho_cc':12.0,
                 'd0':0.3,
                 'pitch':11.5*deg2rad,
                 'phi_00':10.*deg2rad,
                 'rho0_S':7.1,
                 'A_i':[3.,.5,-4.,1.2],
                 'B0_disk':0.1,
                 'B0_halo':0.83,
                 'h_disk':0.1,
                 'h_halo':3,
                 'h_c':2.,
                 'rho_ring_in':2.0,
                 'rho_ring_ex':5.0,
                 'Rho_max':20}

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')

    B_of_rho = parameter['B_0'] * np.exp(-rho**2/parameter['rho_scale']**2)    
    B_of_z = (parameter['B0_disk']/(np.cosh(z/parameter['h_disk']))**2
              + parameter['B0_halo']/(np.cosh(z/parameter['h_halo']))**2)

    
    ### SPIRAL FEATURE
    phi0_i = np.mod(np.asarray([parameter['phi_00']+(i+1)*np.pi/2.
                                for i in range(4)]),2*pi)
    rho0_i = parameter['rho0_S'] * np.exp(phi0_i*np.tan(parameter['pitch']))

    phi_rho = np.asarray([np.log(rho/rho0_i[i])/np.tan(parameter['pitch'])
                          for i in range(4)])
    diff_phi_i = np.asarray([np.arccos(np.cos(phi - phi_rho[i]))
                             for i in range(4)])

    rhoi_of_phi = np.asarray([rho0_i[i]
                              *np.exp((phi_rho[i]+diff_phi_i[i])
                                      *np.tan(parameter['pitch']))
                              for i in range(4)])

    c_of_rho = parameter['C0'] + 0*rho
    c_of_rho[rho>parameter['rho_cc']] *= (rho[rho>parameter['rho_cc']]
                                          /parameter['rho_cc'])**(-3)

    d0_of_rho = parameter['d0']/(c_of_rho*B_of_rho)

    Bspiral = (c_of_rho / (np.cosh(z/parameter['h_c']))**2
               * np.sum([parameter['A_i'][i]
                         *np.exp(-(rho-rhoi_of_phi[i])**2/d0_of_rho**2)
                         for i in range(4)],
                         axis=0))

    ###
    ind_ring = (rho <= parameter['rho_ring_ex'])
    ind_out = (rho > parameter['Rho_max'])

    B_rho = B_of_rho * np.sin(parameter['pitch']) *Bspiral
    B_rho[ind_ring] *= 0
    B_rho[ind_out] *= 0

    B_phi = B_of_rho * np.cos(parameter['pitch']) * Bspiral
    B_phi[ind_ring] = B_of_rho[ind_ring] * B_of_z[ind_ring]
    #and set to zero things inside the inner radius
    B_phi[rho <= parameter['rho_ring_in']] *= 0
    B_phi[ind_out] *= 0

    B_z = 0*B_of_z

    return np.array([B_rho,B_z,B_phi])

    
#
#
#


#def newModel(coordinate,**kwargs{name_param:value_param}):
#    """
#       This is expected to be your favourite B field model
#    """
#
#    ...
#
#    return np.array([B_rho,B_z,B_phi])


#
#
#


def BFielder(coordinate,bfield_setup):
    """
    FUNCTION

            ===  BFIELDER : populates the space with magnetic field  ===

    This function handles the various possibilities of geometrical models
    for the large-scale Galactic magnetic fields in a friendly user way.

    The purpose of this function is a standardization of Bfield call
    such as
        Bfield = BFielder(XYZ,
                          {'name_bfield_model':{'name_param_model':
                                                value_param_model}})

    INPUT:
    ------
     - coordinate : (3,npts)-array with Galactic coordinates at which
                    the B field needs to be computed

     - bfield_setup : dictionary with (key,value) such that
                     - 'key' is a string that must be a Bfield model name:
                         "WMAP"   for [Page et al 2007] model
                         "MLS"    for [Fauvet et al 2010] model

                     - 'value' is a dictionary with parameter names and
                        parameter values for the BField to be built.

    OUTPUT:
    -------
     - bfield : tuple with cylindrical components of the magnetic
                vector field built with the input model at Galactic
                location specified by coordinates

                bfield = [B_rho, B_z, B_phi] is assumed it can be called
                as B_rho,B_z,B_phi = BFielder(XYZ,B_field_model)

    Created on Oct 25 2016
    @author: V.Pelgrims

    """

    if 'WMAP' == list(bfield_setup)[0]:
        #build WMAP model
        bfield = WMAP(coordinate,
                      **bfield_setup['WMAP'])
        if len(bfield_setup['WMAP']) == 0:
            print('''
        WMAP B field model has been computed with DEFAULT parameter values''')
    #
    elif 'MLS' == list(bfield_setup)[0]:
        #build MLS model
        bfield = MLS(coordinate,
                     **bfield_setup['MLS'])
        if len(bfield_setup['MLS']) == 0:
            print('''
        MLS B field model has been computed with DEFAULT parameter values''')
    #
    elif 'ASS' == list(bfield_setup)[0]:
        #build ASS model
        bfield = ASS(coordinate,**bfield_setup['ASS'])
        if len(bfield_setup['ASS']) == 0:
            print('''
        ASS B field model has been computed with DEFAULT parameter values''')
    #
    elif 'BSS' == list(bfield_setup)[0]:
        #build BSS model
        bfield = BSS(coordinate,**bfield_setup['BSS'])
        if len(bfield_setup['BSS']) == 0:
            print('''
        BSS B field model has been computed with DEFAULT parameter values''')
    #
    elif 'QSS' == list(bfield_setup)[0]:
        #build QSS model
        bfield = QSS(coordinate,**bfield_setup['QSS'])
        if len(bfield_setup['QSS']) == 0:
            print('''
        QSS B field model has been computed with DEFAULT parameter values''')
    #
    elif 'LSA' == list(bfield_setup)[0]:
        #build LSA model
        bfield = LSA(coordinate,**bfield_setup['LSA'])
        if len(bfield_setup['LSA']) == 0:
            print('''
        LSA B field model has been computed with DEFAULT parameter values''')
    #
    elif 'CCR' == list(bfield_setup)[0]:
        #build CCR model
        bfield = CCR(coordinate,**bfield_setup['CCR'])
        if len(bfield_setup['CCR']) == 0:
            print('''
        CCR B field model has been computed with DEFAULT parameter values''')
    #
    elif 'BT' == list(bfield_setup)[0]:
        #build BT model
        bfield = BT(coordinate,**bfield_setup['BT'])
        if len(bfield_setup['BT']) == 0:
            print('''
        BT B field model has been computed with DEFAULT parameter values''')
    #
    elif 'ARM4' == list(bfield_setup)[0]:
        #build ARM4 model
        bfield = ARM4(coordinate,**bfield_setup['ARM4'])
        if len(bfield_setup['ARM4']) == 0:
            print('''
        ARM4 B field model has been computed with DEFAULT parameter values''')
    #
    elif 'RING' == list(bfield_setup)[0]:
        #build RING model
        bfield = RING(coordinate,**bfield_setup['RING'])
        if len(bfield_setup['RING']) == 0:
            print('''
        RING B field model has been computed with DEFAULT parameter values''')
    #
    elif 'COMPO' == list(bfield_setup)[0]:
        #build ARM4 model
        bfield = COMPO(coordinate,**bfield_setup['COMPO'])
        if len(bfield_setup['COMPO']) == 0:
            raise ValueError('''
        COMPO B field model needs entries !!!''')
    #
    elif 'JAFFE' == list(bfield_setup)[0]:
        #build JAFFE model
        bfield = JAFFE(coordinate,**bfield_setup['JAFFE'])
        if len(bfield_setup['JAFFE']) == 0:
            print('''
        JAFFE B field model has been computed with DEFAULT parameter values''')
    #
#    elif 'your_favourite_model_name' in bfield_setup.keys():
#        #build your Bfield model
#        bfield = WMAP(coordinate,
#                      **bfield_setup['your_favourite_model_name'])
    elif list(bfield_setup)[0] not in ('WMAP','MLS','ASS','BSS'):
        raise ValueError('''
        The magnetic field name has not been recognized''')

    return bfield


#
#
#


def getBFieldDefault(*args,**kwargs):
    """
    FUNCTION

        ==== getBFieldDefault : print the used default values ===


    getBFieldDefault(*args{'WMAP','MLS','...'},
                     **kwargs{output=True,False[default]})


    This function prints the default parameter values used to build the
    Galactic magnetic field models when those are called without parameter
    value specification. It also shows a complete example of model call.

    INPUT:
    -----
     *args:
        - key word(s) to specify what model to display
          Default is empty, all (listed) models are shown.

    **kwargs:
        - output : True or False to return or not the dictionary containing
                   the Bfield model with default settings.
                   Default is False.
                   If output is True and more than one model is asked, only
                   the last one is returned.


    Create Oct 26, 2016
    @author V.Pelgrims

    """

    #for a nice printing setup:
    def __pretty(value, htchar='\t', lfchar='\n', indent=0):
        '''
        nice function from the internet:
        http://stackoverflow.com/questions/3229419/
        pretty-printing-nested-dictionaries-in-python
        
        @author : y.petremann
        '''
        nlch = lfchar + htchar * (indent + 1)
        if type(value) is dict:
            items = [
                nlch + repr(key) + ': ' + __pretty(value[key],
                                                   htchar, lfchar, indent + 1)
                for key in value
            ]
            return '{%s}' % (','.join(items) + lfchar + htchar * indent)
        elif type(value) is list:
            items = [
                nlch + __pretty(item, htchar, lfchar, indent + 1)
                for item in value
            ]
            return '[%s]' % (','.join(items) + lfchar + htchar * indent)
        elif type(value) is tuple:
            items = [
                nlch + __pretty(item, htchar, lfchar, indent + 1)
                for item in value
            ]
            return '(%s)' % (','.join(items) + lfchar + htchar * indent)
        else:
            return repr(value)
    #
    #
    #

    bfield_model_default_setting = {
        'WMAP': {'B_0' : 2.1,
                 'psi_0' : 27.0*deg2rad,
                 'psi_1' : 0.9*deg2rad,
                 'rho_0' : 8.0,
                 'Xi_0' : 25.0*deg2rad,
                 'z_0' : 1.0,
                 'withBRegular' : False,
                 'r_sun' : 8.0,
                 'R_B' : 8.5,
                 'coord_format' : 'cartesian'},
        'MLS' : {'B_0':2.1,
                 'pitch':-30.0*deg2rad,
                 'rho_0':8.0,
                 'Xi_0':25*deg2rad,
                 'z_0':1.0,
                 'withBRegular' : False,
                 'r_sun':8.0,
                 'R_B':8.5,
                 'coord_format' : 'cartesian'},
        'ASS' : {'B_0': 2.1,
                 'B_amp_param' : 8.0,
                 'B_amp_type' : 'cyl',
                 'pitch' : 11.5*deg2rad,
                 'Xi_0' : 25.0*deg2rad,
                 'z_0' : 1.0,
                 'rho_0' : 8.0,
                 'coord_format' : 'cartesian'},
        'BSS' : {'B_0': 2.1,
                 'B_amp_param' : 8.0,
                 'B_amp_type' : 'cyl',
                 'pitch' : 11.5*deg2rad,
                 'Xi_0' : 25.0*deg2rad,
                 'z_0' : 1.0,
                 'rho_0' : 8.0,
                 'coord_format' : 'cartesian'},
        'QSS' : {'B_0': 2.1,
                 'B_amp_param' : 8.0,
                 'B_amp_type' : 'cyl',
                 'pitch' : 11.5*deg2rad,
                 'Xi_0' : 25.0*deg2rad,
                 'z_0' : 1.0,
                 'rho_0' : 8.0,
                 'coord_format' : 'cartesian'},
        'LSA' : {'B_0': 2.1,
                 'B_amp_param' : 8.0,
                 'B_amp_type' : 'cyl',
                 'psi_0' : 27.0*deg2rad,
                 'psi_1' : 0.9*deg2rad,
                 'Xi_0' : 25.0*deg2rad,
                 'z_0' : 1.0,
                 'rho_0' : 8.0,
                 'coord_format' : 'cartesian'},
        'BT' : {'B_0': 2.1,
                'sig1' : 27.0*deg2rad,
                'sig2' : 0.9*deg2rad,
                'Bz_0' : 0.2,
                'rho_0' : 8.0,
                'coord_format' : 'cartesian'},
        'CCR' : {'omega':3.1,
                 'Dr':0.6,
                 'rho_0':8.0,
                 'B_0':1.6,
                 'z_0':1.0,
                 'Xi_0':25*deg2rad,
                 'B_amp_type':'cst',
                 'coord_format' : 'cartesian'},
        'RING' : {'B_0':2.1,
                 'rho_in':2.,
                 'rho_ex':5.,
                 'coord_format' : 'cartesian'},
        'ARM4' : {'B_0':2.1,
                  'B_of_rho_type':'d_exp',
                  'B_of_rho_scale':8.0,
                  'rho_cusp':.7,
                  'pitch':11.5*deg2rad,
                  'phi_00':10.*deg2rad,
                  'rho_00':1.,
                  'amp_s_i':[1.,1.,1.,1.],
                  'B_s_type':'phi',
                  'B_s_param':15*deg2rad,
                  'B_of_z_type':'cosh',
                  'B_of_z_scale':1.0,
                  'rho_min':0.0,
                  'coord_format' : 'cartesian'},
        'COMPO' : {'B_sun':2.1},
        'JAFFE' : {'B_0':1.,
                   'rho_scale':20.0,
                   'C0':2.5,
                   'rho_cc':12.0,
                   'd0':0.3,
                   'pitch':11.5*deg2rad,
                   'phi_00':10.*deg2rad,
                   'rho0_S':7.1,
                   'A_i':[3.,.5,-4.,1.2],
                   'B0_disk':0.1,
                   'B0_halo':0.83,
                   'h_disk':0.1,
                   'h_halo':3,
                   'h_c':2.,
                   'rho_ring_in':2.0,
                   'rho_ring_ex':5.0,
                   'Rho_max':20}
    }
    #
    bfield_model_to_print = {}
    if len(args) != 0:
        for i in range(len(args)):
            value = bfield_model_default_setting[args[i]]
            bfield_model_to_print[args[i]] = value
    else:
        bfield_model_to_print = bfield_model_default_setting
    #
    print(__pretty(bfield_model_to_print))

    if 'output' in kwargs.keys():
        if kwargs['output']:
            if len(bfield_model_to_print) > 1:
                print('''
            Several Bfield models are returned. The output cannot be
            plugged 'as it' in BFielder. Well, it can but only one will be
            considered.''')
        #
        return bfield_model_to_print


#
#
#


def __get_AlphaGamma(VectorField):
    """
    FUNCTION
            ===  __GET_ALPHAGAMMA : get position and inclination angles ===

    From a Galactic magnetic vector field given in the spherical
    coordinate system centred on the observer, this function evaluates
    the relevant angles for polarizations inquieries:

    - the polarization position angle, at 90 degree away from the
      projected Bfield line, with respect to the (e_theta,e_phi) basis

    - the pitch angle between the line of sight and the vector field.

    INPUT:
    ------
     - VectorField : (3,npts)-array with components of the vector field
                     expressed in spherical coord. assumed to be centred on
                     the observer

    OUTPUT:
    ------
     - Alpha : (npts,)-array with the value of the pitch angle between the
               line of sight and the the vector field at each location where
               Bfield is given [radian]
               # Alpha = 0 means the field is parallel to the los
               #       = pi/2                 perp.

     - Gamma : (npts,)-array with the value of the polarization position
               angle of the polarization vector in the plane orthogonal to
               the line-of-sight at each location where the vector field is
               given [radian ; HEALPix convention]
               The Bfield lines projected onto the polarization plane is
               pi/2 radian away from Gamma
         # Gamma = 0 means the field runs from West to the East (or reverse)
         #       = pi/2                        North       South (or reverse)


    Created on Oct 25 2016
    Based on get_AlphaGamma.py written on Jul 6 2016
    @author: V.Pelgrims

    """

    #we assume that the B vector field is given in spherical basis centred
    #on the observer:
    #\vec{B} = B_r \vec{e_r} + B_t \vec{e_\theta} + B_p \vec{e_\phi}
    #with (\vec{e_r}, \vec{e_\theta}, \vec{e_\phi}) the orthonormal basis
    #vectors of the spherical coordinate system
    #with \vec{e_theta} pointing towards the South pole.
    #
    B_r = VectorField[0]
    B_t = VectorField[1]
    B_p = VectorField[2]
    
    #inclination angle
    Alpha = np.arccos( (B_r*B_r / ( B_r*B_r + B_t*B_t + B_p*B_p ))**.5)
    
    #polarization position angle
    Gamma = .5 * np.arctan2( - 2 * B_t*B_p , B_p*B_p - B_t*B_t )
    #written this way to be in HEALPix convention.
    #Gamma is the polarization position angle expected to be
    #perfectly perpendicular to the projection on the sky of
    #the magnetic field vector
    
    #Gamma = 0 means the polarization points to the South
    #      = pi/2                                   East
    
    return Alpha,Gamma


#
#
#


def __get_AlphaGamma_fromGal(VectorField,dotproducts):
    """
    FUNCTION
        ===  __GET_ALPHAGAMMA_FROMGAL : get position and inclination angles ===

    From a Galactic magnetic vector field given in  Galactocentric
    coordinate system, this function evaluates the relevant angles for
    polarizations inquieries in the oberser reference frame

    - the polarization position angle, at 90 degree away from the
      projected Bfield line, with respect to the (e_theta,e_phi) basis

    - the pitch angle between the line of sight and the vector field.

    INPUT:
    ------
     - VectorField : (3,npts)-array with components of the vector field
                     expressed in a given Galactocentric coordinate system

     - dotproducts : (9,npts)-arrau with dot products between orthonormal
                     basis vectors of the given Galactocentric ref. frame
                     and the spherical coordinates centred on the observer
                     (see gb.GalacticTemplate() output for structure)

    OUTPUT:
    ------
     - Alpha : (npts,)-array with the value of the pitch angle between the
               line of sight and the the vector field at each location where
               Bfield is given [radian]
               # Alpha = 0 means the field is parallel to the los
               #       = pi/2                 perp.

     - Gamma : (npts,)-array with the value of the polarization position
               angle of the polarization vector in the plane orthogonal to
               the line-of-sight at each location where the vector field is
               given [radian ; HEALPix convention]
               The Bfield lines projected onto the polarization plane is
               pi/2 radian away from Gamma
         # Gamma = 0 means the field runs from West to the East (or reverse)
         #       = pi/2                        North       South (or reverse)


    Created on Oct 27 2016
    Based on __get_AlphaGamma.py
    @author: V.Pelgrims

    """

    #bfield components in GalactoCentric reference frame
    B_rho = VectorField[0]
    B_z = VectorField[1]
    B_phi = VectorField[2]

    
    #conversion of the vector field in GC ref. frame to the observer spherical
    #coordinate system
    dots = dotproducts
    B_r = B_rho * dots[0] + B_z * dots[3] + B_phi * dots[6]
    B_t = B_rho * dots[1] + B_z * dots[4] + B_phi * dots[7]
    B_p = B_rho * dots[2] + B_z * dots[5] + B_phi * dots[8]

    #NOTE: computing this conversion internally turns out to be (much) faster
    #that calling gb.__gal2sun_vector() internally or externally.
    
    #inclination angle
    Alpha = np.arccos( (B_r*B_r / ( B_r*B_r + B_t*B_t + B_p*B_p ))**.5)
    
    #polarization position angle
    Gamma = .5 * np.arctan2( - 2 * B_t*B_p , B_p*B_p - B_t*B_t )
    #written this way to be in HEALPix convention.
    #Gamma is the polarization position angle expected to be
    #perfectly perpendicular to the projection on the sky of
    #the magnetic field vector
    
    #Gamma = 0 means the polarization points to the South
    #      = pi/2                                   East
    
    return Alpha,Gamma


#
#
#


def __get_Btransverse2_fromGal(VectorField,dotproducts):
    """
    FUNCTION

        === __GET_BTRANSVERSE2_FROMGAL : get the squared norm of Btransverse

    From a Galactic magnetic vector field given in  Galactocentric coordinate
    system, this function evaluates the square of the norm of the orthogonal
    part of the magnetic field vectors in the oberser reference frame


    INPUT:
    ------
     - VectorField : (3,npts)-array with components of the vector field
                     expressed in a given Galactocentric coordinate system

     - dotproducts : (9,npts)-arrau with dot products between orthonormal
                     basis vectors of the given Galactocentric ref. frame
                     and the spherical coordinates centred on the observer
                     (see gb.GalacticTemplate() output for structure)

    OUTPUT:
    ------
     - squareBtransverse : (npts,)-array with the value of the square of the
                           norm of the projected part of the magnetic field
                           vector on the plane orthogonal to the line of sight


    Created on Oct 28 2016
    Based on __get_AlphaGamma_fromGal.py
    @author: V.Pelgrims

    """

    #bfield components in GalactoCentric reference frame
    B_rho = VectorField[0]
    B_z = VectorField[1]
    B_phi = VectorField[2]

    
    #conversion of the vector field in GC ref. frame to the observer spherical
    #coordinate system
    dots = dotproducts
    B_r = B_rho * dots[0] + B_z * dots[3] + B_phi * dots[6]
    B_t = B_rho * dots[1] + B_z * dots[4] + B_phi * dots[7]
    B_p = B_rho * dots[2] + B_z * dots[5] + B_phi * dots[8]

    #NOTE: computing this convresion internally turns out to be (much) faster
    #that calling gb.__gal2sun_vector() internally or externally.
    
    squared_Btransverse = B_t*B_t + B_p*B_p
    
    return squared_Btransverse


#
#
#


def __get_AlphaGammaBtrans2_fromGal(VectorField,dotproducts):
    """
    FUNCTION
        ===  __GET_ALPHAGAMMA_FROMGAL : get position and inclination angles ===

    From a Galactic magnetic vector field given in  Galactocentric
    coordinate system, this function evaluates the relevant angles for
    polarizations inquieries in the oberser reference frame

    - the polarization position angle, at 90 degree away from the
      projected Bfield line, with respect to the (e_theta,e_phi) basis

    - the pitch angle between the line of sight and the vector field.

    INPUT:
    ------
     - VectorField : (3,npts)-array with components of the vector field
                     expressed in a given Galactocentric coordinate system

     - dotproducts : (9,npts)-arrau with dot products between orthonormal
                     basis vectors of the given Galactocentric ref. frame
                     and the spherical coordinates centred on the observer
                     (see gb.GalacticTemplate() output for structure)

    OUTPUT:
    ------
     - Alpha : (npts,)-array with the value of the pitch angle between the
               line of sight and the the vector field at each location where
               Bfield is given [radian]
               # Alpha = 0 means the field is parallel to the los
               #       = pi/2                 perp.

     - Gamma : (npts,)-array with the value of the polarization position
               angle of the polarization vector in the plane orthogonal to
               the line-of-sight at each location where the vector field is
               given [radian ; HEALPix convention]
               The Bfield lines projected onto the polarization plane is
               pi/2 radian away from Gamma
         # Gamma = 0 means the field runs from West to the East (or reverse)
         #       = pi/2                        North       South (or reverse)

     - squareBtransverse : square of the norm of the transverse part of the
                           magnetic field vectors


    Created on Oct 28 2016
    Based on __get_AlphaGamma_fromGal.py
    @author: V.Pelgrims

    """

    #bfield components in GalactoCentric reference frame
    B_rho = VectorField[0]
    B_z = VectorField[1]
    B_phi = VectorField[2]

    
    #conversion of the vector field in GC ref. frame to the observer spherical
    #coordinate system
    dots = dotproducts
    B_r = B_rho * dots[0] + B_z * dots[3] + B_phi * dots[6]
    B_t = B_rho * dots[1] + B_z * dots[4] + B_phi * dots[7]
    B_p = B_rho * dots[2] + B_z * dots[5] + B_phi * dots[8]

    #NOTE: computing this convresion internally turns out to be (much) faster
    #that calling gb.__gal2sun_vector() internally or externally.
    
    #inclination angle
    Alpha = np.arccos( (B_r*B_r / ( B_r*B_r + B_t*B_t + B_p*B_p ))**.5)
    
    #polarization position angle
    Gamma = .5 * np.arctan2( - 2 * B_t*B_p , B_p*B_p - B_t*B_t )
    #written this way to be in HEALPix convention.
    #Gamma is the polarization position angle expected to be
    #perfectly perpendicular to the projection on the sky of
    #the magnetic field vector
    
    #Gamma = 0 means the polarization points to the South
    #      = pi/2                                   East


    squared_Btransverse = B_t*B_t + B_p*B_p

    return Alpha,Gamma,squared_Btransverse


#
#
#


def __get_AlphaGammaBtrans2(VectorField):
    """
    FUNCTION
        ===  __GET_ALPHAGAMMA_FROMGAL : get position and inclination angles ===

    From a Galactic magnetic vector field given in  Galactocentric
    coordinate system, this function evaluates the relevant angles for
    polarizations inquieries in the oberser reference frame

    - the polarization position angle, at 90 degree away from the
      projected Bfield line, with respect to the (e_theta,e_phi) basis

    - the pitch angle between the line of sight and the vector field.

    INPUT:
    ------
     - VectorField : (3,npts)-array with components of the vector field
                     expressed in a given Galactocentric coordinate system

     - dotproducts : (9,npts)-arrau with dot products between orthonormal
                     basis vectors of the given Galactocentric ref. frame
                     and the spherical coordinates centred on the observer
                     (see gb.GalacticTemplate() output for structure)

    OUTPUT:
    ------
     - Alpha : (npts,)-array with the value of the pitch angle between the
               line of sight and the the vector field at each location where
               Bfield is given [radian]
               # Alpha = 0 means the field is parallel to the los
               #       = pi/2                 perp.

     - Gamma : (npts,)-array with the value of the polarization position
               angle of the polarization vector in the plane orthogonal to
               the line-of-sight at each location where the vector field is
               given [radian ; HEALPix convention]
               The Bfield lines projected onto the polarization plane is
               pi/2 radian away from Gamma
         # Gamma = 0 means the field runs from West to the East (or reverse)
         #       = pi/2                        North       South (or reverse)

     - squareBtransverse : square of the norm of the transverse part of the
                           magnetic field vectors


    Created on Oct 28 2016
    Based on __get_AlphaGamma_fromGal.py
    @author: V.Pelgrims

    """

    #bfield components in HelioCentric reference frame
    B_r = VectorField[0]
    B_t = VectorField[1]
    B_p = VectorField[2]
    
    #inclination angle
    Alpha = np.arccos( (B_r*B_r / ( B_r*B_r + B_t*B_t + B_p*B_p ))**.5)
    
    #polarization position angle
    Gamma = .5 * np.arctan2( - 2 * B_t*B_p , B_p*B_p - B_t*B_t )
    #written this way to be in HEALPix convention.
    #Gamma is the polarization position angle expected to be
    #perfectly perpendicular to the projection on the sky of
    #the magnetic field vector
    
    #Gamma = 0 means the polarization points to the South
    #      = pi/2                                   East


    square_Btransverse = B_t*B_t + B_p*B_p

    return Alpha,Gamma,square_Btransverse


#
#
#


def __get_cylindrical_coord(coordinate,coordinate_format):
    '''
    FUNCTION

        === __get_cylindrical_coord ===

    Does what it says on coordinates with given input format.

    INPUT:
    -----
     - coordinate : (3,npts)-array with coordinates

     - coordinate_format : string that specify the coordiante format
                           given in input.

    OUTPUT:
    ------
     - cyl_coord : the cylindircal coordinates


    Created on Oct 24 2016
    @author: V.Pelgrims
    '''

    if coordinate_format == 'cylindrical':
        pass
    elif coordinate_format == 'cartesian':
        #(x,y,z) is assumed
        coordinate = gb.cart2cyl(coordinate)
    elif coordinate_format == 'spherical':
        #(r,theta,phi) is assumed
        coordinate = gb.sph2cyl(coordinate)
    elif coordinate_format not in ('cartesian','sphercial','cylindrical'):
        raise ValueError('''
        Something went wrong with the given entry coordinate format''')

    return coordinate


#
#
#



 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 #                                                                         #
 #       Nice plotting function to visualise things about the Bfield       #
 #                                                                         #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def plot_galactic_Bfield_xy_slice(bfield_model,*args,**kwargs):
    """
    FUNCTION

        === plot_galactic_Bfield_xy_slice ===


    plot_galactic_Bfield_xy_slice(bfield_model,
                                  *args{radial_step,
                                        xy_limites},
                                  **kwargs{x_sun,
                                           XYZ_sun})


    Does what it says in a squared plot centred on the Galactic Centre.
    Field lines are illustrated over the magnitude of the vector field
    which is given has background colormap.
    Negative magnitudes are set to counter-clockwise vectors,
    except if **directional=False is set.

    INPUT:
    ------
     - bfield_model : dictionary-like variable that contains the Bfield
                      model and its parameters to be used.

     *args:
       - step_p : cell size for plotting and computing precision.
                  If not given, default is .05 kpc

       - limites : maximal x- and y-coordinate to the Galactic Centre
                   that has to be plotted
                   If not given, default value is 20.0 kpc

     **kwargs:
        - x_sun : x-coordinate of the Sun in the GalactoCentric reference
                  frame. Default value = -8.0
        - XYZ_sun : cartesian coordinate of the Sun in the GalactoCentric
                     reference frame. Default values is [-8.0,0.,0.]
        - crange : list or tuple that specifies the min and max value for
                   the color range of imshow.
                   Default is from min and max of the Bmagnitude
        - directional : boolean True to set negative value for
                        counter-clockwise vectors [default]
                        False if no directional info is required

    Display the figure using imshow().

    Created on Oct 25 2016
    @author: V.Pelgrims

    """
    import matplotlib.pyplot as plt
    import matplotlib
    
    step_p = 0.05
    limite = 20.0
    x_sun = -8.0
    y_sun = 0.0
    directional = True


    if len(args) >= 1:
        step_p = args[0]
        if len(args) == 2:
            limite = args[1]

    if kwargs is not None:
        for key,val in kwargs.items():
            if key == 'x_sun':
                x_sun = val
            elif key == 'XYZ_sun':
                x_sun = val[0]
                y_sun = val[1]
            elif key == 'directional':
                directional = val
            elif key not in ('x_sun','XYZ_sun','crange','directional','colormap'):
                raise ValueError('Wrong key worded argument')
    else:
        print('''
    In this plot, the sun is supposed to be located at x_sun=-8.0 kpc.
    ''')


    #construct and plot the B field magnitude
    r = np.arange(-limite-step_p/2.,limite+step_p/2.,step_p)
    X,Y = np.meshgrid(r,r)
    XYZ = np.asarray([X.ravel(),Y.ravel(),0*X.ravel()])

    B_rho,B_z,B_phi = BFielder(XYZ,bfield_model)
    B_magnitude = ( B_rho*B_rho + B_z*B_z + B_phi*B_phi )**.5

    if 'directional' in kwargs.keys() and directional:
        B_magnitude *=  np.sign(B_phi)

    if 'crange' in kwargs.keys():
        cmin = np.min(kwargs['crange'])
        cmax = np.max(kwargs['crange'])
    else:
        cmin = B_magnitude.min()
        cmax = B_magnitude.max()
    if 'colormap' in kwargs.keys():
        cmap = kwargs['colormap']
    else:
        cmap = plt.get_cmap('jet')
        
    limites = [r.min(),r.max(),r.min(),r.max()]
    plt.figure()
    plt.imshow(np.reshape(B_magnitude,[len(r),len(r)]),
               extent=limites,
               origin='lower',
               vmin=cmin,vmax=cmax,cmap = cmap)
    plt.axis(limites)
    cb = plt.colorbar()
    print('''
    >>> Field amplitude has been computed with a
        %s [kpc] grid-spacing'''%(step_p))

    #construct and plot the B field lines
    step_p = 5*step_p
    r = np.arange(-limite-step_p/2.,limite+step_p/2.,step_p)
    X,Y = np.meshgrid(r,r)
    XYZ = np.asarray([X.ravel(),Y.ravel(),0*X.ravel()])

    B_rho,B_z,B_phi = BFielder(XYZ,bfield_model)

    u_rho,u_z,u_phi = gb.__u_cyl(XYZ)
    B_x = B_rho*u_rho[0] + B_phi*u_phi[0]
    B_y = B_rho*u_rho[1] + B_phi*u_phi[1]
    plt.streamplot(X,Y,
                   np.reshape(B_x,[len(r),len(r)]),
                   np.reshape(B_y,[len(r),len(r)]),
                   color=[0.2,.2,0.2],
                   linewidth=1.6,
                   arrowstyle='->',
                   arrowsize=1.0)
    plt.plot(0.,0.,'ok')
    plt.plot(x_sun,y_sun,'*k')
    plt.xlabel('$X$ [kpc]',fontsize=18)
    plt.ylabel('$Y$ [kpc]',fontsize=18)
    plt.title(
        '"%s" - B field model in $(x,y)$-plane'%(list(bfield_model)[0]),
        fontsize=18)
#    cb.remove()
    plt.show()
    print('''
    >>> Field lines have been computed with a
        %s [kpc] grid-spacing using streamplot'''%(step_p))


#
#
#


def plot_galactic_Bfield_xz_slice(bfield_model,*args,**kwargs):
    """
    FUNCTION

        === plot_galactic_Bfield_xz_slice ===

    Does what it says in a rectangular plot centred on the Galactic Centre.
    Field lines are illustrated over the magnitude of the vector field
    which is given has background colormap. Negative magnitudes are set to
    counter-clockwise vectors in the (x,y)-plane,
    except if **directional=False is set.

    INPUT:
    ------
     - bfield_model : dictionary-like variable that contains the Bfield
                      model and its parameters to be used.

     *args:
       - step_p : cell size for plotting and computing precision.
                  If not given, default is .05 kpc

        - x_limite : maximal x-coordinate to the Galactic Centre that has
                     to be plotted
                     If not given, default value is 20.0 kpc

        - z_limite : maximal z-coordinate to the Galactic Centre that has
                     to be plotted
                     If not given, default value is 5.0 kpc

     **kwargs:
        - x_sun : x-coordinate of the Sun in the GalactoCentric reference
                  frame. Default value = -8.0
        - XYZ_sun : cartesian coordinate of the Sun in the GalactoCentric
                     reference frame. Default values is [-8.0,0.,0.]
        - crange : list or tuple that specifies the min and max value for
                   the color range of imshow.
                   Default is from min and max of the Bmagnitude
        - directional : boolean True to set negative value for
                        counter-clockwise vectors [default]
                        False if no directional info is required



    Display the figure using imshow().

    Created on Oct 25 2016
    @author: V.Pelgrims

    """
    import matplotlib.pyplot as plt
    import matplotlib
    
    step_p = 0.05
    x_limite = 20.0
    z_limite = 5.0
    x_sun = -8.0
    z_sun = 0.0


    if len(args) >= 1:
        step_p = args[0]
        if len(args) == 3:
            x_limite = args[1]
            z_limite = args[2]

    if kwargs is not None:
        for key,val in kwargs.items():
            if key == 'x_sun':
                x_sun = val
            elif key == 'XYZ_sun':
                x_sun = val[0]
                z_sun = val[2]
            elif key not in ('x_sun','XYZ_sun','crange','directional','colormap'):
                raise ValueError('Wrong key worded argument')
    else:
        print('''
    In this plot, the sun is supposed to be located at x_sun=-8.0 kpc.
    ''')


    #construct and plot the B field magnitude
    x = np.arange(-x_limite-step_p/2.,x_limite+step_p/2.,step_p)
    z = np.arange(-z_limite-step_p/2.,z_limite+step_p/2.,step_p)
    X,Z = np.meshgrid(x,z)
    XYZ = np.asarray([X.ravel(),0*X.ravel(),Z.ravel()])

    B_rho,B_z,B_phi = BFielder(XYZ,bfield_model)
    B_magnitude = ( B_rho*B_rho + B_z*B_z + B_phi*B_phi )**.5
    if 'directional' in kwargs.keys() and kwargs['directional']:
        B_magnitude *=  np.sign(B_phi)

    if 'crange' in kwargs.keys():
        cmin = np.min(kwargs['crange'])
        cmax = np.max(kwargs['crange'])
    else:
        cmin = B_magnitude.min()
        cmax = B_magnitude.max()
    if 'colormap' in kwargs.keys():
        cmap = kwargs['colormap']
    else:
        cmap = plt.get_cmap('jet')

        
    limites = [x.min(),x.max(),z.min(),z.max()]
    plt.figure()
    plt.imshow(np.reshape(B_magnitude,[len(z),len(x)]),
               extent=limites,
               origin='lower',
               vmin=cmin,vmax=cmax,cmap=cmap)
    plt.axis(limites)
    cb = plt.colorbar()

    #construct and plot the B field lines
    step_p = 10*step_p
    x = np.arange(-x_limite-step_p/2.,x_limite+step_p/2.,step_p)
    z = np.arange(-z_limite-step_p/2.,z_limite+step_p/2.,step_p)
    X,Z = np.meshgrid(x,z)
    XYZ = np.asarray([X.ravel(),0*X.ravel(),Z.ravel()])

    B_rho,B_z,B_phi = BFielder(XYZ,bfield_model)

    u_rho,u_z,u_phi = gb.__u_cyl(XYZ)
    B_x = B_rho*u_rho[0] + B_phi*u_phi[0]
    B_z = B_z#rho*u_rho[1] + B_phi*u_phi[1]
    plt.streamplot(X,Z,
                   np.reshape(B_x,[len(z),len(x)]),
                   np.reshape(B_z,[len(z),len(x)]),
                   color=[.2,.2,.2],
                   linewidth=1.6,
                   arrowstyle='->',
                   arrowsize=1.0)
    plt.plot(0.,0.,'ok')
    plt.plot(x_sun,z_sun,'*k')
    plt.xlabel('$X$ [kpc]',fontsize=18)
    plt.ylabel('$Z$ [kpc]',fontsize=18)
    plt.title(
        '"%s" - B field model in $(x,z)$-plane'%(list(bfield_model)[0]),
        fontsize=18)
    cb.remove()
    plt.show()



#
#
#


def plot_sky_projection(bfield_model,*args,**kwargs):
    '''
    FUNCTION

        === plot_sky_projection ===

    plot_sky_projection(bfield_model,
                        *args(NSIDE,{step_r,limite}),
                        **kwargs(x_sun,XYZ_sun))

    Project the integrated magnitude of the magnetic field model on the
    sky centred on the Sun.

    INPUT:
    ------
     - bfield_model : dictionary-like variable that contains the Bfield
                      model and its parameters to be used.

     *args
       - NSIDE : NSIDE parametre of HEALPix map to be displayed

       - step_r : line-of-sight integration step
                  If not given, default is 0.2 kpc

       - limite : maximal radial distance to the Sun that has to be
                  considered. If not given, default value is 20.0 kpc

     **kwargs:
       - x_sun : x-coordinate of the Sun in the GalactoCentric reference
                 frame. Default value = -8.0
       - XYZ_sun : cartesian coordinate of the Sun in the GalactoCentric
                   reference frame. Default values is [-8.0,0.,0.]
       - crange : list or tuple that specifies the min and max value for
                  the color range of imshow.
                  Default is from min and max of the Bmagnitude


    Display the figure using healpy.mollview() and
    healpy.projaxes.SphericalProjAxes.streamplot()

    Created on Oct 25, 2016
    @author V.Pelgrims
    '''
    import healpy as hp

    step_r = 0.2
    NSIDE = 64
    if type(bfield_model) is dict:
        #everything needs to be computed
        limite = 20.0
        keyed_args = {'Bfield':True}
        if len(args) >= 1:
            NSIDE = args[0]
            if len(args) == 3:
                step_r = args[1]
                limite = args[2]
        elif (len(args) != 0 and len(args) > 3 ):
            raise ValueError('''
        Bad args entry. It has to be either NSIDE or NSIDE,step_r,limite''')
        
        for key,value in kwargs.items():
            keyed_args[key] = value

        if 'crange' in kwargs.keys():
            keyed_args.pop('crange')
        if 'colormap' in kwargs.keys():
            keyed_args.pop('colormap')

        _,XYZ_gal,dots = gb.GalacticTemplate(NSIDE,
                                        step_r,
                                        limite,
                                        **keyed_args)
        #
        B_rho,B_z,B_phi = BFielder(XYZ_gal,bfield_model)
        B_r = B_rho * dots[0] + B_z * dots[3] + B_phi * dots[6]
        B_t = B_rho * dots[1] + B_z * dots[4] + B_phi * dots[7]
        B_p = B_rho * dots[2] + B_z * dots[5] + B_phi * dots[8]
        
    else:
        print('''
        Integration of input array.
        If given, *args and **kwargs have been ignored.
        The input bfield_model is assumed to be built with:
            NSIDE = 64 and
            100 steps for line of sight construction
        and to be given in spherical coordinate centre on the
        observer such as B_r,B_theta,B_phi
        Otherwise use the dictionary-like entry.
            ''')
        B_r = bfield_model[0]
        B_t = bfield_model[1]
        B_p = bfield_model[2]
    #
    B_magnitude = (B_r*B_r + B_t*B_t + B_p*B_p)**.5
    NPIX = hp.nside2npix(NSIDE)
    rSize = B_r.size/NPIX

    themap = np.sum(np.reshape(B_magnitude,[rSize,NPIX]),axis=0)*step_r

    if 'crange' in kwargs.keys():
        cmin = np.min(kwargs['crange'])
        cmax = np.max(kwargs['crange'])
    else:
        cmin = themap.min()
        cmax = themap.max()
    if 'colormap' in kwargs.keys():
        cmap = kwargs['colormap']
    else:
        cmap = plt.get_cmap('jet')


    hp.mollview(themap,min=cmin,max=cmax,cmap=cmap)

#    In the future, it would be nice to overplot streamlines of the
#    integrated projected Bfield...
#    Better would be to include the line integral convolution (LIC)
#       to represent the field now that is implemented in healpy:
#       https://github.com/healpy/healpy/pull/617#issue-434041253




###########################################################################
#                                                                         #
#################################  E N D  #################################
#                                                                         #
###########################################################################
