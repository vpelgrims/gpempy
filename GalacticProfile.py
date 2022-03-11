# -*- coding: utf-8 -*-
"""
    gpempy -- GalacticProfile
    Copyright (C) 2022  V.Pelgrims

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++                                                                       ++
++                       GALACTIC PROFILE MODULE                         ++
++                                                                       ++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

|    Implementation of various geometrical density distributions that
|    can be used to model Galactic dust or Cosmic Ray Electron density
|    profile.
|
|    Contains also some functions to print and handle basics of the module
|    
|
|    Starting Date of Module Creation: Sep 21 2016
|    @author: V.Pelgrims
"""

import numpy as np
import GalaxyBasics as gb
from GalaxyBasics import sph2cyl

pi = np.pi
deg2rad = pi/180.0


def GalArmProfile(coord, A_0, R3, sig_rho_in, sig_rho_ex,
                  pitch, a_or, delta_i, sig_z, **kwargs):
    """
    FUNCTION

            ===  GalArmProfile : Galactic Arm Profile ===
    
    For a given set of locations and values of model parameters, this function
    returns density values given by a logarithmic spiral arm thought to model
    our Galaxy.
    This distribution has native cylindrical symmetry.
    Following the notations of [Steiman-Cameron et al 2010 ApJ 722-1460],
    the density of one arm is modeled in cylindrical coordinates as:
    
        " A(rho,z,phi) = A_0 * exp(- abs(rho - R3) / sig_rho)
                                * exp(- (phi-phi_i(rho))**2 / delta_i**2)
                                    * exp(- z**2 / (2*sig_z**2))

        where:
            phi_i(rho) = log(rho / a_or) / alpha
            characterizes the shape of the spiral arm and alpha is related
            to the pitch angle p through p = arctan(alpha)

    This function can be used to model only a piece of a spiral arm by
    passing relevant argument in kwargs.


    INPUT:
    ------
     - coord : (3,npts)-array with the coordinate values at which the
               density has to be evaluated.
               Coordinate system is assumed to be cylindrical by default
               but can also be spherical or cartesian and can be specified
               throught the optional argument "input_coord".

     - A_0 : amplitude factor of the density
     
     - R3 : [kpc] allows for the disappearance of arms in the inner Galaxy.
             The exponential scale-length sig_rho can be different inside
             and outside this radius value.
     
     - sig_rho_in : [kpc] exponential scale-length for rho <= R3
     
     - sig_rho_ex : [kpc] exponential scale-length for rho > R3
     
     - pitch : [deg] pitch angle of the arm
     
     - a_or : determines the orientation of the spiral [see S-C et al 2010]
     
     - delta_i : [deg] exponential scale-length of angular dependence
     
     - sig_z : [kpc] exponential scale length of height dependence


     **kwargs:
         - input_coord : string that specifies the format of the system of
                         coordinates given in input.
                         Can be 'cylindrical' (default), 'spherical' or
                         'cartesian'

        Parameters for piece of an arm.
        Not all are necessary. But at least three are requiered.
         - phi_min : [deg] (minimum) angular value of the starting point

         - rho_min : [deg] (minimum) radial value of the starting point

         - phi_max : [deg] (maximum) angular value of the starting point

         - rho_max : [deg] (maximum) radial value of the starting point

         - arm_angular_length : [deg] angular length of the arm.
                                can be >> 360 deg
        => given the input parameters, the longest possible piece of
           the arm is going to be built. This choice allows for
           'approximated' couples of values like (rho_min,phi_min) etc.
           which are precisely re-computed herein.
           'rho' can have an approximated value
           'phi' can be given up to 360 deg degeneracy
        The code works for the following (minimal) entries:
          - (phi_min, rho_min), arm_angular_length or phi_max or rho_max
          - (phi_max, rho_max), arm_angular_length or phi_min or rho_min

        => I NEED TO WRITE THE OTHER POSSIBILITIES!

    OUTPUT:
    -------
     - A(coord) : (npts,)-array with density values at input locations


    Created on Wed Sep 21 2016
    @author: V.Pelgrims
    """
    
    #variable initialization
    input_coord = 'cylindrical'
    piece_i = 0; #has to be three to compute only a piece of an arm.
    phi_min = +9999.
    rho_min = +9999.
    phi_max = -9999.
    rho_max = 1.0e-4
    arm_angular_length = -999.
    
    if kwargs is not None:
        for key,value in kwargs.items():
            if key == 'input_coord':
                input_coord = value
            elif key == 'phi_min':
                phi_min = value * pi/180.
                piece_i += 1
            elif key == 'phi_max':
                phi_max = value * pi/180.
                piece_i += 1
            elif key == 'rho_min':
                rho_min = value
                piece_i += 1
            elif key == 'rho_max':
                rho_max = value
                piece_i += 1
            elif key == 'arm_angular_length':
                arm_angular_length = value * pi/180.
                piece_i += 1
            else:
                print(
    '''\n
         !!! Bad entry for optional arguments, key must be:
         "input_coord", "phi_min", "phi_max", "rho_min",
         "rho_max" or "arm_angular_length"''')
                
                return                

    #coordinates' stuff
    if input_coord == 'spherical':
        #from sph2cyl import sph2cyl
        cyl = sph2cyl(coord[0],coord[1],coord[2])        
        rho = cyl[0]
        z = cyl [1]
        phi = cyl[2]
    elif input_coord == 'cylindrical':
        rho = coord[0]
        z = coord[1]
        phi = coord[2]
    elif input_coord == 'cartesian':
        rho = np.sqrt(coord[0]*coord[0] + coord[1]*coord[1])
        z = coord[2]
        phi = np.arctan2(coord[1],coord[0])
    else:
        print('Bad entry for type of "input_coord"')
        return


    #build the density profile

    #for angular dependence
    alpha = np.tan(pitch*np.pi/180.0)
    phi_rho = np.log(rho / a_or) / alpha
    diff_phi = np.arccos(np.cos(phi - phi_rho))
    #the last line here takes into account the fact that +10deg and -10deg
    #are separated by 20deg and not 340deg. This is important for the
    #exponential density below

    #for radial dependence
    sig_rho = np.ones(len(rho))
    sig_rho[np.where(rho <= R3)] *= sig_rho_in
    sig_rho[np.where(rho > R3)] *= sig_rho_ex

    #the arm density profile
    A = (A_0 * np.exp(- np.abs(rho - R3) / sig_rho)
         * np.exp(- diff_phi*diff_phi / (delta_i*np.pi/180.0)**2)
         * np.exp(- z*z / (2*sig_z*sig_z)))
    
    #
    #HERE something is needed to cut the spiral arm if only a piece has
    #to be considered
    #
    if piece_i >= 3:
        #only a piece of the arm is going to be built
        #a case-by-case study seems to be requiered
        #
        #
        #the fact that phi_min and phi_max can be determined up to 360deg
        #does not ease the things. Write a function that computes the
        #right minimum and the maximum angular bondaries given the entries
        def set2pidegeneracy(phi_in,phi_th):
            #phi_in = angular coord that can be defined up to 2pi degeneracy
            #phi_th = angular coord with no 2pi degeneracy but perhapes not
            #precise enougth
            if np.abs(phi_in - phi_th) >= pi:
                phi_in -= np.sign(phi_in - phi_th) * 2*pi
            return phi_in
        # end of function set2pidegeneracy
        
        if phi_min != 9999. and rho_min != 9999.:
            phi_min = set2pidegeneracy(phi_min,
                                       np.log(rho_min/a_or)/alpha)
            rho_min = a_or * np.exp(alpha * phi_min)
            #
            if (phi_max != -9999. and arm_angular_length == -999.
                and rho_max == 1.0e-4):
                phi_max = set2pidegeneracy(phi_max,phi_min)
                rho_max = a_or * np.exp(alpha * phi_max)
            elif (arm_angular_length != -999. and phi_max == -9999.
                  and rho_max == 1.0e-4):
                phi_max = phi_min + arm_angular_length
                rho_max = a_or * np.exp(alpha * phi_max)
            elif (rho_max != 1.0e-4 and phi_max == -9999.
                  and arm_angular_length == -999.):
                if rho_max <= rho_min:
                    print(
    '''\n    !!! rho_max has not a good value it is set
    such that the arm-length is 5deg length \n'''
                    )
                    phi_max = phi_min + 5*pi/180.
                else:
                    phi_max = set2pidegeneracy(np.log(rho_max/a_or)/alpha,
                                               phi_min)
                rho_max = a_or * np.exp(alpha * phi_max)
                #
            #The case with exclusively 3 entries with (phi_min,rho_min)
            #is done above. I should consider the other cases.
            #
            #
        elif phi_max != -9999. and rho_max != 1.0e-4:
            phi_max = set2pidegeneracy(phi_max,
                                       np.log(rho_max/a_or)/alpha)
            rho_max = a_or * np.exp(alpha * phi_max)
            #
            if phi_min != +9999.:
                phi_min = ste2pidegeneracy(phi_min,phi_max)
            elif arm_angular_length != -999.:
                phi_min = phi_max - arm_angular_length
            #
            if rho_min != +9999.:
                phi_min = setd2pidegeneracy(phi_min,
                                            np.log(rho_min/a_or)/alpha)
            #
            rho_min = a_or * np.exp(alpha * phi_min)
            #
            #

        else:
            print('''
            Something is not working (yet?) here!''')
            return



        #
        #the density A has to be set to zero in regions too far from the
        #piece of the arm.
        #if phi_rho is not included in [phi_min, phi_max] then A = 0
        #it is also necessary to cut at some radial distances from
        #the arm piece, otherwise several can show up for the angular cut
        
        A[np.where(np.mod(phi,2*pi) < np.mod(phi_min,2*pi))] *= 0.
        A[np.where(np.mod(phi,2*pi) > np.mod(phi_max,2*pi))] *= 0.
        A[np.where(rho + sig_rho < rho_min)] *= 0.
        A[np.where(rho - sig_rho > rho_max)] *= 0.
          
                    
    elif piece_i >0 and piece_i <3:
        #at least three parameters are requiered to cut the arm
        print('At least three parameters are requiered to cut the arm,',
               piece_i, 'are given.')
        return

    return A


#
#
#


def ARM4(coord,**kwargs):
    """
    FUNCTION
        === ARM4 : 4 logarithmic  Sprial Arms density profile model ===

        ARM4(coordinates,**kwargs{numerous parameters})

    'ARM' model adapted from literature
     The arms are identical with a 90 degree rotation

        "A(rho,z,phi) = A_0 * f(rho) * g(z) * s(rho,phi)
         where
            " f(rho) = exp(-|rho-rho_cusp|/sig_rho)
            " g(z) = 1./(cosh(z/sig_z))**2
            " s(rho,phi) = #if s_type='rho'
                           sum_i{a_i * exp(-|rho - rho_S_i(phi)|/rho_0) }
                           #else if s_type = 'phi' [default]
                           sum_i{a_i * exp(-(phi - phi_S_i(rho))**2)
                                            /(2(phi_0**2))}
            with
            " rho_S_i(phi) = rho0_i * exp(phi*tan(pitch))
            " phi_S_i(rho) = ln(rho/rho0_i)/tan(pitch)
                where
                   phi0_i = phi00 + i*pi/2                      #imposed
                   rho0_i = rho0_i * exp(phi0_i * tan(pitch)))  #computed
                   i = 1,2,3,4


    INPUT:
    ------
      - coord :  an (3,n)-array with coordinates at which the matter density
                 has to be evaluated. Default 'coord' format is cartesian
     **kwargs :
        - coord_format : string that specifies the coord format of the INPUT
                         coord. Can be cartesian, spherical or cylindrical
                         Default is 'cartesian'
      [model parameter]
       - A_0 : amplitude of the matter density
       - f_rho_type : functional form of the radial dependence
                      default is exp decreasing from some rho_cusp
       - f_rho_scale : [kpc] parameter of the function of the radial
                        dependence
       - rho_cusp : [kpc] gives the radius of maximum amplitude
       - pitch : [rad] pitch angle of the spiral (same for all)
       - phi_00 : [rad] angular start of the 4th arm [default = 10degree]
       - rho_00 : [kpc] 1 [default] spiral arm radial parameter
       - amp_s_i : tuple of 4 amplitudes for the density field of the arm
                   they are meant to be relative amplitudes
       - s_type : defines the amplitude dependence of the arms. Can be
                  set either with radial ('rho') or with angular ('phi')
                  dependence. (see s(rho,phi) above)
       - s_param : parameter of the function form for the arm amplitude
       - g_z_type : functional form of the height dependence (cosh)
       - g_z_scale : height scale [default = 1.]
       - rho_min : [kpc] minimum radial scale from which to compute the
                    matter density field


       > getDusterDefault('ARM4') for default setting values of parameters

    
    OUTPUT:
    ------
        ARM4_profile : amplitude of the dust profile at all locations
                       specified in 'coord'


    Created on Jul 18 2017
    @author: V.Pelgrims
    """


    #initialization
    parameter = {'A_0':2.1,
                 'f_rho_type':'d_exp',
                 'f_rho_scale':8.0,
                 'rho_cusp':.7,
                 'pitch':11.5*deg2rad,
                 'phi_00':10.*deg2rad,
                 'rho_00':1.,
                 'amp_s_i':[1.,1.,1.,1.],
                 's_type':'phi',
                 's_param':15*deg2rad,
                 'g_z_type':'cosh',
                 'g_z_scale':1.0,
                 'rho_min':0.0}

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')

    f_rho = OneParamFunc(rho-parameter['rho_cusp'],
                         parameter['f_rho_scale'],
                         parameter['f_rho_type'])

    
    ### SPIRAL FEATURE
    phi0_i = np.mod(np.asarray([parameter['phi_00']+(i+1)*np.pi/2.
                                for i in range(4)]),2*pi)
    rho0_i = parameter['rho_00'] * np.exp(phi0_i*np.tan(parameter['pitch']))

    # to not miss part of contribution...
    phi_rho = np.asarray([np.log(rho)/np.tan(parameter['pitch']) - phi_0i
                          for phi_0i in phi0_i])

    diff_phi = np.mod(phi - phi_rho + np.pi, 2*np.pi) - np.pi

    if parameter['s_type'] == 'phi':
        #amplitude follows spiral through angular parameterization
        s = np.sum([parameter['amp_s_i'][i]
                    *OneParamFunc(diff_phi[i],
                                  parameter['s_param'],
                                  'Gauss') for i in range(4)],axis=0)
        #
    elif parameter['s_type'] == 'rho':
        #amplitude follows spiral through radial parameterization
        delta_rho = [rho*np.minimum(np.abs(1-
                                           np.exp(diff_phi[i]
                                                  *np.tan(parameter['pitch']))),
                                    np.abs(1-
                                           np.exp((diff_phi[i]-2*np.pi)
                                                  *np.tan(parameter['pitch']))))
                     for i in range(4)]
        s = np.sum([parameter['amp_s_i'][i]
                    *OneParamFunc(delta_rho[i],
                                  parameter['s_param'],
                                  'd_exp') for i in range(4)],axis=0)
    else:
        raise ValueError('''
        Bad entry for optional argument 's_type'.
        Key must be one either: 'phi' or 'rho' ''')

    #height dependence
    g_z = OneParamFunc(z,
                       parameter['g_z_scale'],
                       parameter['g_z_type'])
    g_z[rho <= parameter['rho_min']] *=0

    ARM4_profile = parameter['A_0'] * f_rho * s * g_z

    return ARM4_profile



#
#
#

def RING(coord,**kwargs):
    """
    FUNCTION
        === RING : ring matter density profile ===

    RING(coord,list_of_parameter)

      Fill the space with a constant density amplitude between two
      cylindrical radii and up to some height above and below the
      Galactic plane.

    INPUT:
     - coord.
     model param
    - amplitude
    - inner radius
    - outer radius

    Created on Jul 18 2017
    @author: V.Pelgrims
    """

    #initialization
    parameter = {'A_0':2.1,
                 'rho_in':2.,
                 'rho_ex':5.,
                 'z_scale':1.
                 }

    #set parameter value to entries, if given
    for key,value in kwargs.items():
        parameter[key] = value

    #coordinate stuff
    if 'coord_format' in kwargs.keys():
        rho,z,phi = __get_cylindrical_coord(coord,kwargs['coord_format'])
    else:
        rho,z,phi = __get_cylindrical_coord(coord,'cartesian')

    RING_profile = parameter['A_0'] + 0*z

    RING_profile[rho <= parameter['rho_in']] *=0
    RING_profile[rho >= parameter['rho_ex']] *=0
    RING_profile[np.abs(z) >= parameter['z_scale']] *=0

    return RING_profile

#
#
#


#
#


def eGalProfile(coord,A_0,A_rho,A_z,**kwargs):
    """
    FUNCTION
    
            ===  eGalProfile : exponential Galactic Profile ===
    
    For a given set of locations and model-parameter values, this function
    returns density values given by an exponential distribution on the
    Galactic plane.
    This distribution class has native cylindrical symmetry
    
        " A(rho,z,phi) = A_0 * exp(-(rho / A_rho)) / (cosh(z / A_z))**2
        
    INPUT:
    ------
     - coord : (3,npts)-array with the coordinate values at which the
               density has to be evaluated. Coordinate system can be either
               cylindrical, spherical or cartesian and can be specified
               throught the optional argument "input_coord".
               Default is "cartesian" coordinate syst. even though the model
               is natively cylindric

     - A_0 : amplitude of the density at the origine
     
     - A_rho : radial (cyl.) scale of the distribution
     
     - A_z : vertical scale of the distribution

     **kwargs:
         - input_coord : string that specifies the format of the system of
                         coordinates given in input.
                         Can be 'cartesian' (default), 'cylindrical'
                         or 'spherical'.


    OUTPUT:
    -------
     - A(rho,z,phi) : (npts,)-array with density values at input locations


    Created on Jul 7, 2016
    Modified on Sep 28, 2016 [use of OneParamFunc function]
    @author: V.Pelgrims
    """


    #variable initialization
    input_coord = 'cartesian'
    
    if kwargs is not None:
        for key,value in kwargs.items():
            if key == 'input_coord':
                input_coord = value
            else:
                print(
    '''Bad entry for optional arguments, key must be "input_coord"'''
                )
                return                

    A = cylHaloProfile(coord,A_0,
                       'Decreasing_Exp',A_rho,
                       'HyperbolicSecant',A_z,
                       input_coord = input_coord)
    
    return A


#
#
#


def HaloProfile(coord,A_0,**kwargs):
    """
    FUNCTION

            ===  HaloProfile : Halo Galactic Profile ===
    
    For a given set of locations and model-parameter values, this function
    returns density values of a model with spherical symmetry that mimics
    an hypothetical halo centred on the origin of the coord. syst. with a
    density that is a function of the radial distance.
    The function follows the form specified by "form" and can be Gaussian,
    hyperbolic secant, decresaing exponential or power law.
    Default is of Gaussian form to avoid divergence.

    
        " A(coord) = A_0 * f(r,param)
          where - r is the radial (spherical) coordinate
                - param is the (only) parameter of the wanted
                        fonctional form
                Default is of Gaussian form with a standard deviation
                set to one third of the maximal r distance from coord.

    INPUT:
    ------
     - coord : (3,npts)-array with the coordinate values at which the
               density has to be evaluated. Coordinate system can be either
               cylindrical, spherical or cartesian and can be specified
               throught the optional argument "input_coord".
               Default is "cartesian" coordinate syst.

     - A_0 : amplitude of the density at the origine
     
     **kwargs:
         
         - radial_scale : value of the parameter of the functional form
                          Default is one third of max of r and is assumed\
                          to be the std of a Gaussian profile

         - radial_form : string that specifies the functional form
                         of the radial profile.
                         - Gaussian, cosh, d_exp, power, Cosine
                         Default is 'Gaussian'
     
         - input_coord : string that specifies the format of the system of
                         coordinates given in input.
                         Can be 'cartessian' (default), 'cylindrical'
                         or 'spherical'.


    OUTPUT:
    -------
     - A(xyz) : (npts,)-array with density values at input locations


    Created on Sep 22, 2016
    @author: V.Pelgrims
    """
    

    #variable initialization
    input_coord = 'cartesian'
    radial_scale = 0
    radial_form = 'Gaussian'
    
    
    if kwargs is not None:
        for key,value in kwargs.items():
            if key == 'input_coord':
                input_coord = value
            elif key == 'radial_scale':
                radial_scale = value
            elif key == 'radial_form':
                radial_form = value
            else:
                print('Bad entry for optional arguments, key must be',
                      '"input_coord", "radial_scale"',
                      'or "radial_form"')
                return                

    #coordinates' stuff
    if input_coord == 'spherical':
        #coord = (r,theta,phi) is assumed
        r = coord[0]
    elif input_coord == 'cylindrical':
        #coord = (rho,z,phi) is assumed
        r = ( coord[0]*coord[0] + coord[1]*coord[1] )**.5
    elif input_coord == 'cartesian':
        #coord=(x,y,z) is assumed
        r = ( np.sum(coord*coord,axis=0) )**.5
    else:
        print('Bad entry for type of "input_coord"')
        return

    if radial_form == 'Gaussian' and radial_scale == 0:
        #default values are used
        radial_scale = r.max() / 3.0

    #build the density profile
    Halo = A_0 * OneParamFunc(r,radial_scale,radial_form)

    return Halo


#
#
#


def cylHaloProfile(coord,A_0,rho_func,rho_param,z_func,z_param,**kwargs):
    """
    FUNCTION

        ===  cylHaloProfile : cylindrical Halo Galactic Profile ===
    
    For a given set of locations and model-parameter values, this function
    return density values of a model with rotational symmetry about the
    z-axis of the Galaxy and with an additional function of the height.
    The hypothetical halo centred on the origine of the coord. syst. has a
    density profile determined such as

        " cylHalo(coord) = A_0 * func(rho,rho_param) * func(z,z_param)

          where :
            - rho is the radial (cylindrical) coordinate
            - z is the height coordinate

            - func(rho,rho_param) = OneParamFunc(rho,rho_param,rho_func)

            - func(z,z_param) = OneParamFunc(z,z_param,z_func)
            
            FOR Gaussian, Hyperbolic Secant, Decreasing Exponential
                Power Law profile or Cosine.


    INPUT:
    ------
     - coord : (3,npts)-array with the coordinate values at which the
               density has to be evaluated. Coordinate system can be either
               cylindrical, spherical or cartesian and can be specified
               throught the optional argument "input_coord".
               Default is "cylindrical" coordinate syst.

     - A_0 : amplitude of the density at the origine

     - rho_func : string for functional form of the rho dependence

     - rho_param : the one-parameter value of the function

     - z_func : string for functional form of the z dependence

     - z_param : the one-parameter value of the function

      => rho_func and z_func have to be one of the key words of the
         OneParamFunc function.
     
     **kwargs:
         - input_coord : string that specifies the format of the system of
                         coordinates given in input.
                         Can be 'cylindircal' (default), 'cartesian'
                         or 'spherical'.


    OUTPUT:
    -------
     - cylHalo(xyz) : (npts,)-array with density values at input locations


    Created on Sep 22, 2016
    @author: V.Pelgrims
    """
    

    #variable initialization
    input_coord = 'cylindrical'
    
    if kwargs is not None:
        for key,value in kwargs.items():
            if key == 'input_coord':
                input_coord = value
            else:
                print(
    '''Bad entry for optional arguments, key must be "input_coord"''')
                return



    ###
    # THIS SHOULD BE REPLACED BY __GET_CYLINDIRCAL_COORD() FROM BFIELD MODULE
    ###
    #
    #
    #coordinates' stuff
    if input_coord == 'spherical':
        #coord = (r,theta,phi) is assumed
        cyl = sph2cyl(coord[0],coord[1],coord[2])        
        rho = cyl[0]
        z = cyl [1]
    elif input_coord == 'cylindrical':
        #coord = (r,z,phi) is assumed
        rho = coord[0]
        z = coord[1]
    elif input_coord == 'cartesian':
        #coord = (x,y,z) is assumed
        rho = np.sqrt(coord[0]*coord[0] + coord[1]*coord[1])
        z = coord[2]
    else:
        print('Bad entry for type of "input_coord"')
        return



    #build the density profile of the cylindrical halo
    cylHalo = (A_0 * OneParamFunc(rho,rho_param,rho_func)
               * OneParamFunc(z,z_param,z_func))

    return cylHalo


#
#
#


def clumpProfile(coord,x_CC,y_CC,z_CC,A0,r_func_form,r_func_param,**kwargs):
    """
    FUNCTION

        ===  clumpProfile : clump Galactic Profile ===

    For a given set of locations and model-parameter values, this function
    returns density values of a parametric model with spherical symmetry
    about the given centre of a clump.

    This function relies on the HaloProfile function in this module.

        " A(coord-clumCentre) = A_0 * func(r distance to clump centre)


    INPUT:
    ------
     - coord : (3,npts)-array with the coordinate values at which the
               density has to be evaluated. Coordinate system can be either
               cylindrical, spherical or cartesian and can be specified
               throught the optional argument "input_coord".
               Default is "cartesian" coordinate syst.

     - x_CC : x-coordinate of the clump centre
     - y_CC : y-coordinate of the clump centre
     - z_CC : z-coordinate of the clump centre [kpc are assumed, as usual]

     - A_0 : amplitude of the density at the origine

     - r_func_form : specify the form of the function to adopt
                     Can be 'power' for power law
                            'cosh' for hyperbolic secant
                            'Gauss' for Gaussian
                            'exp' for decreasing exponential
                            'Cosine' for cosine

     - r_func_param : the parameter of the function, i.e
                      - the power law index
                      - the scale-length

     **kwargs:
         - input_coord : string that specifies the format of the system of
                         coordinates given in input.
                         Can be 'cartesian' (default), 'cylindrical'
                         or 'spherical'.


    OUTPUT:
    -------
     - A(xyz) : (npts,)-array with density values at input locations


    Created on Sep 23, 2016
    @author: V.Pelgrims
    """
    

    #variable initialization
    input_coord = 'cartesian'
    
    if kwargs is not None:
        for key,value in kwargs.items():
            if key == 'input_coord':
                input_coord = value
            else:
                print('Bad entry for optional arguments, key must be',
                '"input_coord"')
                return                


    ###
    # THIS SHOULD BE REPLACED BY GET_CARTESIAN_COORD FROM GALAXYBASICS MODULE
    ###
    #
    #            
    #coordinates' stuff
    if input_coord == 'spherical':
        #coord = (r,theta,phi) is assumed
        #translate to cartesian coordinates
        #and shift it to the centre of the clump
        x = coord[0] * np.cos(coord[2]) * np.sin(coord[1]) - x_CC
        y = coord[0] * np.sin(coord[2]) * np.sin(coord[1]) - y_CC
        z = coord[0] * np.cos(coord[1]) - z_CC
        xyz = np.array([x,y,z])
    elif input_coord == 'cylindrical':
        #coord = (r,z,phi) is assumed
        #translate to cartesian coordinates
        #and shift it to the centre of the clump
        x = coord[0] * np.cos(coord[2]) - x_CC
        y = coord[0] * np.sin(coord[2]) - y_CC
        z = coord[1] - z_CC
        xyz = np.array([x,y,z])
    elif input_coord == 'cartesian':
        #coord = (x,y,z) is assumed
        #shift the coordinates to the centre of the clump
        xyz = coord - np.tile(np.array([[x_CC],[y_CC],[z_CC]]),
                              (1,len(coord[0])))
    else:
        print('Bad entry for type of "input_coord"')
        return


    #build the density profile using the HaloProfile function
    clumpProfile = HaloProfile(xyz,A0,
                               radial_form = r_func_form,
                               radial_scale = r_func_param,
                               input_coord = 'cartesian')

    return clumpProfile


#
#
#


def loopProfile(coord,x_LC,y_LC,z_LC,A0,
                shellRadius,r_func_form,r_func_param,**kwargs):

    """
    FUNCTION

        ===  loopProfile : loop Galactic Profile ===

    For a given set of locations and model-parameter values, this
    function returns a shell-like density profile of a parametric
    model with spherical symmetry about the given centre of a loop
    and some radial dependence.


        " LoopDensity = A_0 * func(r distance to loop centre)


    INPUT:
    ------
     - coord : (3,npts)-array with the coordinate values at which the
               density has to be evaluated. Coordinate system can be either
               cylindrical, spherical or cartesian and can be specified
               throught the optional argument "input_coord".
               Default is "cartesian" coordinate syst.

     - x_CC : x-coordinate of the loop centre
     - y_CC : y-coordinate of the loop centre
     - z_CC : z-coordinate of the loop centre [kpc are assumed, as usual]

     - A_0 : amplitude of the density at the origine

     - shellRadius : [kpc] the radius of the shell that produces the loop
                     i.e. the distance between the origine of the shell
                     centred on (x_LC,...) and the locations of density
                     maxima

     - r_func_form : specify the form of the function to adopt
                     Can be 'power'    for Power Law
                            'cosh'     for Hyperbolic Secant
                            'Gauss'    for Gaussian
                            'exp'      for Decreasing Exponential
                            'Cosine'   for Cosine

     - r_func_param : the parameter of the one-parameter function, e.g
                      - the power law index
                      - the scale-length

     **kwargs:
         - input_coord : string that specifies the format of the system of
                         coordinates given in input.
                         Can be 'cartesian' (default), 'cylindrical'
                         or 'spherical'.


    OUTPUT:
    -------
    - Loop(xyz) : (npts,)-array with density values at input locations


    Created on Sep 28, 2016
    @author: V.Pelgrims
    """
    

    #variable initialization
    input_coord = 'cartesian'
    
    if kwargs is not None:
        for key,value in kwargs.items():
            if key == 'input_coord':
                input_coord = value
            else:
                print('Bad entry for optional arguments, key must be',
                '"input_coord"')
                return                

    ###
    # THIS SHOULD BE REPLACED BY GET_CARTESIAN_COORD FROM GALAXYBASICS MODULE
    ### BUT ... WE ONLY NEED THE (SPHERICAL) RADIUS ABOUT THE LOOP CENTRE ...
    #
    #            
    #coordinates' stuff
    if input_coord == 'spherical':
        #coord = (r,theta,phi) is assumed
        #translate to cartesian coordinates
        #and shift it to the centre of the clump
        x = coord[0] * np.cos(coord[2]) * np.sin(coord[1]) - x_LC
        y = coord[0] * np.sin(coord[2]) * np.sin(coord[1]) - y_LC
        z = coord[0] * np.cos(coord[1]) - z_LC
        xyz = np.array([x,y,z])
    elif input_coord == 'cylindrical':
        #coord = (r,z,phi) is assumed
        #translate to cartesian coordiantes
        #and shift it to the centre of the clump
        x = coord[0] * np.cos(coord[2]) - x_LC
        y = coord[0] * np.sin(coord[2]) - y_LC
        z = coord[1] - z_LC
        xyz = np.array([x,y,z])
    elif input_coord == 'cartesian':
        #coord = (x,y,z) is assumed
        #shift the coordinates to the centre of the clump
        xyz = coord - np.tile(np.array([[x_LC],[y_LC],[z_LC]]),
                              (1,len(coord[0])))
    else:
        print('Bad entry for type of "input_coord"')
        return


    r = np.sum(xyz*xyz,axis=0)**.5
    #shift the vales of r for shell-like density
    r_shifted = r-shellRadius
    
    #build the density profile using the HaloProfile function
    loop = A0 * OneParamFunc(r_shifted,r_func_param,r_func_form)

    return loop


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
        print('The one-parameter functional referenced by',
              form,
              'is not recognized. \n Wrong entry or not',
              'yet implemented. Try again!')
        return

    return f_of_var


#
#
#


def Duster(coord,list_of_dusty_things):
    """
    FUNCTION

        === DUSTER : populates the space with (dust) density profile(s) ===

    This function handles the various possibilities of geometrical profiles
    for the (dust) density distribution in a friendly user way.

    The aim is to be able to plug several dust components in an homogeneous
    and standardized way such as:
    Dust = Duster(XYZ,{'Halo',{halo_parameters},
                       'cylHalo',[{cylHalo_parameters}],
                       'Arm',[{Arm1_parameters},
                              {Arm2_parameters},
                              {Arm3_parameters},
                              {Arm4_parameters}],
                       'Loop',[{Loop_parameters}], etc. })

    Of course the user need to know how to declare the value of the
    parameters of the wanted profiles...
        >>> getDusterDefault() will help on this matter.

    It can also be used using yaml configuration file such as:
    > import yaml
    > with open('best_model_config.yml','r') as config:
    >    dust_model = yaml.load(config)
    > Dust = Duster(XYZ,dust_model)


    INPUT:
    ------
     - coord : (3,Npts)-array with cartesian coordinates of the space to be
               filled by mater (like dust).

     - list_of_dusty_things : dictionary with (key,value)-couple.

           - 'key' is a string that must be either:
                   "Halo"      for spherical halo profile
                   "cylHalo"   for cylindrical halo profile
                   "Arm"       for logarithmic spiral arm profile 
                   "Loop"      for a loop profile
                   "Clump"     for a clump profile

           - 'value' is a dictionary that contains the parameters of the
                     corresponding profile in 'key' if default values are
                     not to be used.

             >>>  See getDusterDefault('key') for default values <<<

    OUTPUT:
    -------
     - Dust_Density : the dust density computed as being the sum of the
                      profiles listed in list_of_dusty_things
                      (Npts,)-array

    Created on Oct 14 2016
    @author: V.Pelgrims

    """

    n = len(coord[0])

    Dust_density = np.zeros(n)
    
    print('''\n''')
    for key,value in list_of_dusty_things.items():
        if key == 'Halo':
            for i in range(len(list_of_dusty_things[key])):
                #default parameter values
                amp = 1.
                keyed_args = {}
                #initialization to input if given
                for key_p,val_p in list_of_dusty_things[key][i].items():
                    if key_p == 'amplitude':
                        amp = val_p
                    elif (key_p == 'radial_form' or
                          key_p == 'radial_scale'):
                        keyed_args[key_p] = val_p
                #evaluation of dust density with given model
                Dust_density += HaloProfile(coord,
                                            amp,
                                            **keyed_args)
                print('''  - ''',key,i+1)
        #
        elif key == 'cylHalo':
            for i in range(len(list_of_dusty_things[key])):
                #default parameter values
                amp = 1.
                rho_func_form = 'Decreasing_Exp'
                rho_func_param = 3.
                z_func_form = 'HyperbolicSecant'
                z_func_param = 1.
                #initialization to input if given
                for key_p,val_p in list_of_dusty_things[key][i].items():
                    if key_p == 'amplitude':
                        amp = val_p
                    elif key_p == 'radial_form':
                        rho_func_form = val_p
                    elif key_p == 'radial_scale':
                        rho_func_param = val_p
                    elif key_p == 'height_form':
                        z_func_form = val_p
                    elif key_p == 'height_scale':
                        z_func_param = val_p
                #evaluation of dust density with given model
                Dust_density += cylHaloProfile(coord,
                                               amp,
                                               rho_func_form,
                                               rho_func_param,
                                               z_func_form,
                                               z_func_param,
                                               input_coord='cartesian')
                print('''  - ''',key,i+1)
        #
        elif key == 'Loop':
            for i in range(len(list_of_dusty_things[key])):
                #default parameter values
                amp = 1.
                radial_func_form = 'Gaussian'
                radial_func_param = 0.03
                shell_radius = 0.14
                x_loopCentre = -8.0
                y_loopCentre = -0.045
                z_loopCentre = +0.07
                #initialization to input if given
                for key_p,val_p in list_of_dusty_things[key][i].items():
                    if key_p == 'amplitude':
                        amp = val_p
                    elif key_p == 'radial_form':
                        radial_func_form = val_p
                    elif key_p == 'radial_scale':
                        radial_func_param = val_p
                    elif key_p == 'shell_radius':
                        shell_radius = val_p
                    elif key_p == 'loopCentre':
                        x_loopCentre = val_p[0]
                        y_loopCentre = val_p[1]
                        z_loopCentre = val_p[2]
                #evaluation of dust density with given model
                Dust_density += loopProfile(coord,
                                            x_loopCentre,
                                            y_loopCentre,
                                            z_loopCentre,
                                            amp,
                                            shell_radius,
                                            radial_func_form,
                                            radial_func_param)
                print('''  - ''',key,i+1)
        #
        elif key == 'Clump':
            for i in range(len(list_of_dusty_things[key])):
                #default parameter values
                amp = 1.
                radial_form = 'Gaussian'
                radial_scale = .3
                x_clumpCentre = -4.0
                y_clumpCentre = -6.2
                z_clumpCentre = -3.0
                #initialization to input if given
                for key_p,val_p in list_of_dusty_things[key][i].items():
                    if key_p == 'amplitude':
                        amp = val_p
                    elif key_p == 'radial_form':
                        radial_form = val_p
                    elif key_p == 'radial_scale':
                        radial_scale = val_p
                    elif key_p == 'clumpCentre':
                        x_clumpCentre = val_p[0]
                        y_clumpCentre = val_p[1]
                        z_clumpCentre = val_p[2]
                #evaluation of dust density with given model
                Dust_density += clumpProfile(coord,
                                             x_clumpCentre,
                                             y_clumpCentre,
                                             z_clumpCentre,
                                             amp,
                                             radial_form,
                                             radial_scale)
                print('''  - ''',key,i+1)
        #
        elif key == 'Arm':
            for i in range(len(list_of_dusty_things[key])):
                amp = 1.
                rho_cusp = 2.9
                radial_scale_in = 0.7
                radial_scale_ex = 3.1
                pitch_angle = 13.6
                a_orientation = 0.246
                angular_scale = 15.
                height_scale = 0.07
                keyed_args = {'input_coord':'cartesian'}
                #initialization to input if given
                for key_p,val_p in list_of_dusty_things[key][i].items():
                    if key_p == 'amplitude':
                        amp = val_p
                    elif key_p == 'rho_cusp':
                        rho_cusp = val_p
                    elif key_p == 'radial_scale_in':
                        radial_scale_in = val_p
                    elif key_p == 'radial_scale_ex':
                        radial_scale_ex = val_p
                    elif key_p == 'pitch_angle':
                        pitch_angle = val_p
                    elif key_p == 'a_orientation':
                        a_orientation = val_p
                    elif key_p == 'angular_scale':
                        angular_scale = val_p
                    elif key_p =='height_scale':
                        height_scale = val_p
                    elif (key_p == 'phi_min' or
                          key_p == 'phi_max' or
                          key_p == 'rho_min' or
                          key_p == 'rho_max' or
                          key_p == 'arm_angular_length'):
                        keyed_args[key_p] = val_p
                #evaluation of dust density with given model
                Dust_density += GalArmProfile(coord,
                                              amp,
                                              rho_cusp,
                                              radial_scale_in,
                                              radial_scale_ex,
                                              pitch_angle,
                                              a_orientation,
                                              angular_scale,
                                              height_scale,
                                              **keyed_args)
                print('''  - ''',key,i+1)
        #
        elif key == 'ARM4':
            list_of_values = list_of_dusty_things['ARM4']
            for i in range(len(list_of_values)):
                Dust_density += ARM4(coord,**list_of_values[i])
                                        
        #
        elif key == 'RING':
            list_of_values = list_of_dusty_things['RING']
            for i in range(len(list_of_values)):
                Dust_density += RING(coord,**list_of_values[i])
                                        
        #
        elif key not in ('Halo','cylHalo','Arm','Loop','Clump',
                         'ARM4','RING'):
            raise ValueError(
    '''\n
         !!! Bad entry for dusty component name. Key must be:
         "Halo", "cylHalo", "Arm", "Loop", "Clump", "ARM4" or "RING".''')


    return Dust_density


#
#
#


def getDusterDefault(*args,**kwargs):
    '''
    FUNCTION

        ==== getDusterDefault : print the used default values ===


    getDusterDefault(*args{'Halo','cylHalo','Clump','Loop','Arm'},
                     **kwargs{output=True,False[default]})


    This function prints the default parameter values used to build the
    different dust components when those are called without parameter
    value specification. It also shows a complete example of component
    initialization.

    INPUT:
    -----
     *args:
        - key word(s) to specify what component to display
          Default is empty, all components are shown.

    **kwargs:
        - output : True or False to return or not the dictionary containing
                   the dust component with default settings.
                   Default is False.


    Create Oct 19, 2016
    @author V.Pelgrims

    '''

    #for a nice printing setup:
    def __pretty(value, htchar='\t', lfchar='\n', indent=0):
        '''
        function from:
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
    dust_component_default_setting = {
        'Halo':[{'amplitude' : 1.0,
                 'radial_form' : 'Gaussian',
                 'radial_scale' : 10.0}],
        'cylHalo':[{'amplitude' : 1.0,
                    'radial_form' : 'Decreasing_Exp',
                    'radial_scale' : 3.0,
                    'height_form' : 'HyperbolicSecant',
                    'height_scale' : 1.0}],
        'Clump':[{'amplitude' : 1.0,
                  'radial_form' : 'Gaussian',
                  'radial_scale' : 0.3,
                  'clumpCentre' : [-4.0,-6.2,-3.0]}],
        'Loop':[{'amplitude' : 1.0,
                 'radial_form' : 'Gaussian',
                 'radial_scale' : 0.03,
                 'shell_radius' : 0.14,
                 'loopCentre' : [-8.0,-0.045,0.07]}],
        'Arm':[{'amplitude' : 1.0,
                'rho_cusp' : 2.9,
                'radial_scale_in' : 0.7,
                'radial_scale_ex' : 3.1,
                'pitch_angle' : 13.6,
                'a_orientation' : 0.246,
                'angular_scale' : 15.0,
                'height_scale' : 0.07}],
        'ARM4':[{'A_0':2.1,
                 'f_rho_type':'d_exp',
                 'f_rho_scale':8.0,
                 'rho_cusp':.7,
                 'pitch':11.5*deg2rad,
                 'phi_00':10.*deg2rad,
                 'rho_00':1.,
                 'amp_s_i':[1.,1.,1.,1.],
                 's_type':'phi',
                 's_param':15*deg2rad,
                 'g_z_type':'cosh',
                 'g_z_scale':1.0,
                 'rho_min':0.0}],
        'RING':[{'A_0':2.1,
                 'rho_in':2.,
                 'rho_ex':5.,
                 'z_scale':1.
                 }]
    }
    #
    dust_component_to_print = {}
    if len(args) != 0:
        for i in range(len(args)):
            value = dust_component_default_setting[args[i]]
            dust_component_to_print[args[i]] = value
    else:
        dust_component_to_print = dust_component_default_setting
    #
    print(__pretty(dust_component_to_print))

    send_output = False
    if kwargs is not None:
        for key,value in kwargs.items():
            if key == 'output':
                send_output = value
            else:
                raise ValueError('Bad entry for key-worded argument')

    if send_output:
        return dust_component_to_print


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
 #       Nice plotting function to visualize matter density profile        #
 #                                                                         #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def plot_galactic_dust_xy_slice(dust_model,*args,**kwargs):
    '''
    FUNCTION

        === plot_galactic_dust_xy_slice ===

    Does what it says in a squared plot centred on the Galactic Centre.

    INPUT:
    ------
      - dust_model : dictionary-like variable that contains the dust
                     components and their parameter initialization.

      *args
        - step : cell size for plotting precision.
                 If not given, default is 0.05 kpc

        - limite : maximal radial distance to the Galactic Centre that has
                  to be plotted.
                  If not given, default value is 20.0 kpc

      **kwargs:
        - x_sun : x-coordinate of the Sun in the GalactoCentric reference
                  frame. Default value = -8.0
        - XYZ_sun : cartesian coordinate of the Sun in the GalactoCentric
                     reference frame. Default values is [-8.0,0.,0.]


    Display the figure using imshow().

    Created on Oct 19, 2016
    @author V.Pelgrims
    '''
    import matplotlib.pyplot as plt
    import matplotlib

    step_p = 0.05
    limite = 20.0
    x_sun = -8.0
    y_sun = 0.0

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
            else:
                raise ValueError('Wrong key worded argument')
    else:
        print('''
    In this plot, the sun is supposed to be located at x_sun=-8.0 kpc.
    ''')

    
    r = np.arange(-limite-step_p/2.,limite+step_p/2.,step_p)
    X,Y = np.meshgrid(r,r)
    XYZ = np.asarray([X.ravel(),Y.ravel(),0*X.ravel()])

    dust_density = Duster(XYZ,dust_model)

    #plotting stuff
    limites = [r.min(),r.max(),r.min(),r.max()]
    vmin = dust_density.min()
    vmax = dust_density.max()
    plt.figure()
    plt.imshow(np.reshape(dust_density,[len(r),len(r)]),
               extent=limites,
               origin='lower',
               norm=matplotlib.colors.LogNorm(vmin=vmin,
                                              vmax=vmax))
    plt.axis(limites)
    plt.colorbar()
    plt.plot(0,0,'ok')
    plt.plot(x_sun,y_sun,'*k')
    plt.xlabel('$X$ [kpc]',fontsize=18)
    plt.ylabel('$Y$ [kpc]',fontsize=18)
    plt.show()


#
#
#

def plot_galactic_dust_xz_slice(dust_model,*args,**kwargs):
    '''
    FUNCTION

        === plot_galactic_dust_xz_slice ===

    Does what it says in a rectangular plot centred on the Galactic Centre.

    INPUT:
    ------
      - dust_model : dictionary-like variable that contains the dust
                     components and their parameter initialization.

      *args:
        - step_p : cell size for plotting precision.
                   If not given, default is 0.05 kpc

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

    Display the figure using imshow().

    Created on Oct 20, 2016
    @author V.Pelgrims
    '''
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
            else:
                raise ValueError('Wrong key worded argument')
    else:
        print('''
    In this plot, the sun is supposed to be located at x_sun=-8.0 kpc.
    ''')


    x = np.arange(-x_limite-step_p/2.,x_limite+step_p/2.,step_p)
    z = np.arange(-z_limite-step_p/2.,z_limite+step_p/2.,step_p)
    X,Z = np.meshgrid(x,z)
    XYZ = np.asarray([X.ravel(),0*X.ravel(),Z.ravel()])

    dust_density = Duster(XYZ,dust_model)

    #plotting stuff
    limites = [x.min(),x.max(),z.min(),z.max()]
    vmin = dust_density.min()
    vmax = dust_density.max()
    plt.figure()
    plt.imshow(np.reshape(dust_density,[len(z),len(x)]),
               extent=limites,
               origin='lower',
               norm=matplotlib.colors.LogNorm(vmin=vmin,
                                              vmax=vmax))

#               norm=matplotlib.colors.LogNorm(vmin=5.0e-3,
#                                              vmax=1.1))
    plt.axis(limites)
    plt.colorbar()
    plt.plot(0,0,'ok')
    plt.plot(x_sun,z_sun,'*k')
    plt.xlabel('$X$ [kpc]',fontsize=18)
    plt.ylabel('$Z$ [kpc]',fontsize=18)
    plt.show()


#
#
#


def plot_sky_projection(dust_model,*args,**kwargs):
    '''
    FUNCTION

        === plot_sky_projection ===

    plot_sky_projection(dust_model,
                        *args(NSIDE,{step_r,limite}),
                        **kwargs(x_sun))

    Project the integrated dust density model on the sky centred on the Sun.

    INPUT:
    ------
      - dust_model : dictionary-like variable that contains the dust
                     components and their parameter initialization.

      *args
        - NSIDE : NSIDE parametre of HEALPix map to be displayed

        - step_r : line-of-sight integration step
                   If not given, default is 0.2 kpc

        - limite : maximal radial distance to the Sun that has to be
                   considered. If not given, default value is 20.0 kpc

       **kwargs:
            Same as for the gb.GalacticTemplate function


    Display the figure using healpy.mollview()

    Created on Oct 20, 2016
    @author V.Pelgrims
    '''
    import healpy as hp


    step_r = 0.2
    NSIDE = 64
    if type(dust_model) is dict:
        #everything needs to be computed
        import GalaxyBasics as gb
        limite = 20.0
        keyed_args = {'Bfield':False}
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
                
        _,XYZ_gal = gb.GalacticTemplate(NSIDE,
                                        step_r,
                                        limite,
                                        **keyed_args)
        #
        dust_model = Duster(XYZ_gal,dust_model)
    else:
        print('''
        Integration of input array.
        If given, *args and **kwargs have been ignored.
        The input dust_model is assumed to be built with:
            NSIDE = 64 and
            100 steps for line of sight construction
        Otherwise use the dictionary-like entry. Sorry about that.
            ''')
    #
    NPIX = hp.nside2npix(NSIDE)
    rSize = np.int(dust_model.size/NPIX)
    hp.mollview(np.sum(np.reshape(dust_model,[rSize,NPIX]),
                       axis=0)*step_r,
                norm='log')


#
#
#


###########################################################################
#                                                                         #
################################# E N D ###################################
#                                                                         #
###########################################################################
