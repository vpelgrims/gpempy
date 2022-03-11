# -*- coding: utf-8 -*-
"""
    gpempy -- GalaxyBasics
    Copyright (C) 2022  V.Pelgrims

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++                                                                       ++
++                         GALAXYBASICS MODULE                           ++
++                                                                       ++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Basic fonctions to setup a Galactic template in two differents
    system of coordinates --centred on the Galactic centre and on
    the sun-- and to go back and forth between the two with scalar
    and vectorial quantities.


Starting Date of Module Creation: Oct 12 2016
@author: V.Pelgrims
"""

import numpy as np
import healpy as hp

pi = np.pi


def GalacticTemplate(*args,**kwargs):
    """
    FUNCTION

            === GalacticTemplate : builds a Galactic template ===

    GalacticTemplate(*args(NSIDE,radial_step,radial_max),
                     **kwargs('XYZ_sun','x_sun','Bfield'))

    This function sets up two sets of (not homogeneously distributed)
    cartesian coordinates for given angular and radial precisions.
        The first set is centred on the Sun and consists of
    12 * NSIDE**2 lines of sight with a radial sampling given by
    np.arange(radial_step, radial_max+radial_step, radial_step)
        The second set of coordinates is the same of the first but
    translated to put the Galactic centre at the origin.

    Additionally, because the large-scale Galactic magnetic field is
    generally described in cylindrical coordinates of a reference frame
    centred on the Galactic Centre, it is necessary to project the
    magnetic 3D vector field onto the observer spherical coordinate.
    We therefore provide all scalar products between the cylindrical-basis
    vectors of the Galaxy and the spherical-basis vectors of the observer.


    INPUT:
    ------
     *args:

         - NSIDE : HEALPix parameter for all-sky map.
                   Default = 64

         - radial_step : [kpc] value for the radial sampling, i.e. for the
                         integration precision
                         Default = 0.2 kpc

         - radial_max : [kpc] maximal value up to which the Galaxy is supposed
                        to extend (radially) from the Sun
                        Default = 20.0 kpc

         [radial_step and radial_max are expected to be given together,
          if given]


     **kwargs:

         - XYZ_sun : [kpc] cartesian coordinates of the Sun in the
                     Galacto-centric coordinates system. (3,)-array
                     Default = [-8., 0., 0.]
                     !!! if either y_sun or z_sun != 0 then the Galactic
                     centre is no more at (l,b) = (0,0)

         - x_sun : [kpc] x-coordinate of the Sun in the Galacto-centric
                   coordinates system.
                   Default = 8.

         - Bfield : boolean.
                    If True [default], compute the dotproducts between basis
                    vectors and return them along with other outputs.
                    Else, do not compute the dotproduct and the function
                    only return XYZ_sun and XYZ_gal.

    OUTPUT:
    -------
     - XYZ_sun : the set of cartesian coordinates in heliocentric
                 coordinate system.
                 If rSize is the number of radial steps and NPIX the number
                 of lines of sight, then the structure of the output is
                 such that there are rSize blocks of NPIX elements.
                 The first NPIX vectors have a norm equal to radial_step,
                 the next NPIX following ones have a norm of 2*radial_step
                 etc etc.
                 (3,NPIX*rSize)-array

     - XYZ_gal : the set of cartesian coordinates in the galacto-centric
                 coordinate system. The structure is the same.
                 There is just a translation between XYZ_sun and XYZ_gal.
                 (3,NPIX*rSize)-array

     - dotproducts : contains the dot products between the cylindrical basis
                     vectors of the reference frame centred on the Galaxy
                     with those of the spherical coordinate system centred
                     on the Sun.
                     dotproduct.shape = (9,NPIX*rSize)
                     dotproduct[0] = u_rhoG.u_rS
                     dotproduct[1] = u_rhoG.u_tS
                     dotproduct[2] = u_rhoG.u_pS
                     dotproduct[3] = u_zG.u_rS
                     ...
                     dotproduct[8] = u_phiG.u_phiS

    Created on Oct 12 2016

    @author: V.Pelgrims

    """


    #variable initialization
    if (len(args) != 0 and len(args) != 1 and len(args) != 3):
        raise ValueError(
            '''Wrong assignment of argument.
               They should be either:
                - the NSIDE parameter of the healpy map
                - the NSIDE, the radial step and the radial maximum [kpc]''')
    else:
        NSIDE = 64
        radial_step = .2
        radial_max = 20.
        if len(args) == 1:
            NSIDE = args[0]
        elif len(args) == 3:
            NSIDE = args[0]
            radial_step = args[1]
            radial_max = args[2]

    #check input parameters
    if not hp.isnsideok(NSIDE):
        raise ValueError(' ! Not a good value for the NSIDE parameter.')
    if radial_max < 20.:
        print('You are going to miss a large part of the Galaxy.')


    Bfield = True
    x_sun = -8.0    # [kpc]
    y_sun = 0.0
    z_sun = 0.0
    if kwargs is not None:
        for key,value in kwargs.items():
            if key == 'Bfield':
                Bfield = value
            elif key == 'x_sun':
                x_sun = value
            elif key == 'XYZ_sun':
                x_sun = value[0]
                y_sun = value[1]
                z_sun = value[2]
                
            elif key not in ('Bfield',
                             'x_sun','XYZ_sun'):
                raise ValueError(
    '''\n
         !!! Bad entry for optional arguments, key must be:
         "Bfield", "x_sun" or "XYZ_sun".''')

    if y_sun != 0 or z_sun != 0:
        print('''y_sun and z_sun are different from zero thus the
Galactic centre is not towards (l,b) = (0,0)''')
   
    NPIX = hp.nside2npix(NSIDE)
    r = np.arange(radial_step,radial_max+radial_step,radial_step)
    rSize = r.size

    #unit pointing vectors
    xyz_n = np.asarray(hp.pix2vec(NSIDE,np.arange(NPIX)))

    #pointing vectors to all wanted positions
#    XYZ_sun = (np.tile(xyz_n,(1,rSize)) *
#               np.tile(np.reshape(np.tile(r.T,(NPIX,1)),
#                                  NPIX*rSize,1).T,(3,1)))
#
    XYZ_sun = np.reshape((xyz_n.T * r[:,None,None]).T,[3,NPIX*rSize],order='F')
        #the structure is such that there is rSize blocks of NPIX elements.
        #the first NPIX elements (vectors) have a norm equal to radial_step.
        #The next NPIX following ones have a norm of 2*radial_step etc.
        #In other words, its structure is (3,nPix*rSize).
        #Each block of (3,nPix) draws a sphere with fixed radius.

    #conversion from the heliocentric coord syst to Galactocentric one
    XYZ_gal = (XYZ_sun +
               np.tile(np.array([x_sun,y_sun,z_sun]),(rSize*NPIX,1)).T)

    #products between unit vectors from both ref frame need also to be computed
    print('''
    The Galaxy Template has been built with:
        NSIDE = ''', NSIDE,'''
        radial_step = ''', radial_step, '''kpc
        radial_max = ''', radial_max, '''kpc

    The Galactic space is thus populated by''', NPIX*rSize, '''points
    distributed spherically around the Sun.
    ''')

    #OUTPUT and dotproduct evaluation, if wanted
    if Bfield:
        dots = __ifBfield(XYZ_sun,XYZ_gal)
        print('''    Dot products between basis vectors have been computed''')
        return XYZ_sun,XYZ_gal,dots
    elif not Bfield:
        print('''    Dot products between basis vectors have not been computed''')
        return XYZ_sun,XYZ_gal


    # +++ end of GalacticTemplate function +++


#
#
#


def __ifBfield(XYZ_sun,XYZ_gal):
    """
    FUNCTION
        Comptutes the dotproducts between the basis vectors of the cylindrical
    coordinate system centred on the Galactic centre and those of the spherical
    system centred on the Sun.
    """
    
    #compute the cylindrical-basis vectors in the GC reference frame
    u_rhoG,u_zG,u_phiG = __u_cyl(XYZ_gal)

    #compute the spherical-basis vectors in the heliocentric reference frame
    u_rS,u_tS,u_pS = __u_sph(XYZ_sun)

    #compute the requiered scalar products between basis vectors
    d_rGrS = np.sum(u_rhoG*u_rS, axis=0)    # u_rhoG . u_rS
    d_rGtS = np.sum(u_rhoG*u_tS, axis=0)    # u_rhoG . u_thetaS
    d_rGpS = np.sum(u_rhoG*u_pS, axis=0)    # u_rhoG . u_phiS
    #
    d_zGrS = np.sum(u_zG*u_rS, axis=0)      # u_zG . u_rS
    d_zGtS = np.sum(u_zG*u_tS, axis=0)      # u_zG . u_thetaS
    d_zGpS = np.sum(u_zG*u_pS, axis=0)      # u_zG . u_phiS
    #
    d_pGrS = np.sum(u_phiG*u_rS, axis=0)    # u_phiG . u_rS
    d_pGtS = np.sum(u_phiG*u_tS, axis=0)    # u_phiG . u_thetaS
    d_pGpS = np.sum(u_phiG*u_pS, axis=0)    # u_phiG . u_phiS
    #
    #
    dotproducts = np.array([d_rGrS,d_rGtS,d_rGpS,
                            d_zGrS,d_zGtS,d_zGpS,
                            d_pGrS,d_pGtS,d_pGpS])
    
    return dotproducts


#
#
#


def __u_cyl(coord):
    """
    For input cartesian coordinates, returns the 3-basis vectors of the
    cylindrical coordinate system.

    INPUT:
    ------
     - coord : an (3, N)-array of the coordinates at which the vectors
               have to be computed.

    OUTPUT:
    -------
    - u_rho, u_z, r_phi : three (3, N)-array containing the basis vectors

    Created on Jun 22 2016
    @author: V.Pelgrims
    """

    phi = np.arctan2(coord[1],coord[0])
 
    #compute the stuff    
    u_rho1 = np.cos(phi)
    u_rho2 = np.sin(phi)
    u_rho3 = np.zeros(phi.size)
    
    u_rho = np.array([u_rho1, u_rho2, u_rho3])

    u_phi = np.array([-u_rho2, u_rho1, u_rho3])
    
    u_z = np.array([u_rho3, u_rho3, u_rho3 + 1.])

    return u_rho, u_z, u_phi


#
#
#


def __u_sph(coord):
    """
    For input cartesian coordinates, returns the 3-basis vectors of the
    spherical coordinate system.

    INPUT:
    ------
     - coord : an (3, N)-array of the coordinates at which the vectors
               have to be computed.

    OUTPUT:
    -------
     - u_r, u_theta, u_phi : three (3, N)-array containing the basis vectors

    Created on Jun 22 2016
    @author: V.Pelgrims
    """
    
    theta = np.arctan2((coord[0]**2 + coord[1]**2)**.5,coord[2])        
    phi = np.arctan2(coord[1],coord[0])

    #compute the stuff
    Cphi = np.cos(phi)
    Sphi = np.sin(phi)
    Ctheta = np.cos(theta)
    Stheta = np.sin(theta)

    u_r = np.array([Cphi*Stheta, Sphi*Stheta, Ctheta])
    
    u_theta = np.array([Cphi*Ctheta, Sphi*Ctheta, -Stheta])

    u_phi = np.array([-Sphi, Cphi, np.zeros(Cphi.size)])

    return u_r, u_theta, u_phi


#
#
#


def sun2gal(XYZ,**kwargs):
    """
    FUNCTION

        === sun2gal : heliocentric to Galactocentric coordinates ===

    This function converts coordinates centred on the Sun to coordinates
    in a Galactic reference frame with the Galactic Centre at the origin.

    INPUT :
    -------
     - XYZ : (3,npst)-array with coordinates in the reference frame
             centred on the Sun [kpc are assumed]

     **kwargs:
         - coord_format : specifies the format of input and out coord.
                          Can be either 'spherical', 'cylindrical' or 
                          'cartesian' (default)

         - x_Sun : distance of the Sun from the Galactic Centre [kpc].
                   Default = 8. [kpc]

         - XYZ_sun : cartesian coord. of the sun in the Galacto Centric
                     system of coord. [kpc],
                     Default is (-8., 0., 0.)


    OUTPUT:
    -------

     - XYZ_Gal : (3,npts)-array with coordinates in the reference
                 frame centred on the GC.
                 Same format as the input (cart., sph. or cyl.)
 
    Created on Jun 20 2016
    @author: V.Pelgrims
    """

    #initializing variable(s)
    x_Sun = -8.  #in [kpc], the sun location in Galactocentric reference frame
    y_Sun = 0.
    z_Sun = 0.
    coord_format = 'cartesian'

    if kwargs is not None:
        if 'coord_format' in kwargs.keys():
            XYZ = get_cartesian_coord(XYZ,kwargs['coord_format'])
            coord_format = kwargs['coord_format']
            
        for key,value in kwargs.items():
            if key == 'x_Sun':
                x_Sun = value
            elif key == 'XYZ_sun':
                XYZ_sun = value
                if XYZ_sun[1] != 0:
                    #rotate the Galactic disk such that the sun lies in the
                    #(x,z)-plane of the Galactocentric coordinate system
                    XYZ_sun[0] = (XYZ_sun[0]*XYZ_sun[0] +
                                  XYZ_sun[1]*XYZ_sun[1])**.5
                    XYZ_sun[1] = 0.
                    print('''
    !!!  The Galactic disk has been rotated such that the sun lies in the
         (x,z)-plane of the cartesian Galactocentric coordinate system''')

                x_Sun = XYZ_sun[0]
                y_Sun = XYZ_sun[1]
                z_Sun = XYZ_sun[2]
                alpha = np.arctan2(z_Sun,x_Sun)
            #
            elif key not in ('coord_format','x_sun','XYZ_sun'):
                raise ValueError('''
                Bad entry for optional arguments, key must be either
                "x_Sun", "XYZ_sun", "coord_format"''')

    XYZ_GC = XYZ + np.array([[x_sun],[y_sun],[z_sun]])

    if z_sun != 0:
        #a rotation is requiered to have the GC at b=0
        x = np.cos(alpha)*XYZ_GC[0] + np.sin(alpha)*XYZ_GC[2]
        z = np.cos(alpha)*XYZ_GC[2] - np.sin(alpha)*XYZ_GC[0]
        XYZ_GC[0] = x
        XYZ_GC[2] = z

    #
    #output the coordinates in the same format as the where given in input
    if  coord_format == 'cylindrical':
        #(rho,z,phi) is expected
        rho_GC = (XYZ_GC[0]*XYZ_GC[0] + XYZ_GC[1]*XYZ_GC[1])**.5
        phi_GC = np.arctan2(XYZ_GC[1],XYZ_GC[0])
        z_GC = XYZ_GC[2]
        XYZ_GC = np.array([rho_GC, z_GC, phi_GC])
    elif coord_format == 'spherical':
        #(r,theta,phi) is expected
        r_GC = (XYZ_GC[0]*XYZ_GC[0] + XYZ_GC[1]*XYZ_GC[1] +
                XYZ_GC[2]*XYZ_GC[2])**.5
        theta_GC = np.arctan2((XYZ_GC[0]*XYZ_GC[0]
                               + XYZ_GC[1]*XYZ_GC[1])**.5,
                              XYZ_GC[2])
        phi_GC = np.arctan2(XYZ_GC[1], XYZ_GC[0])
        XYZ_GC = np.array([r_GC, theta_GC, phi_GC])
    elif coord_format not in ('cartesian','spherical','cylindrical'):
        raise ValueError('''
        Something is going wrong with the coord_format type!''')
        
        #would be better to have small functions somewhere to call here
        #such as cart2cyl, cyl2cart and so on and so forth

    return XYZ_GC


#
#
#


def get_cartesian_coord(coordinate,coordinate_format):
    '''
    FUNCTION

        === get_cartesian_coord ===

    Does what it says on coordinates with given input format.

    INPUT:
    -----
     - coordinate : (3,npts)-array with coordinates

     - coordinate_format : string that specify the coordiante format
                           given in input.

    OUTPUT:
    ------
     - cart_coord : the cartesian coordinates


    Created on Oct 21 2016
    @author: V.Pelgrims
    '''

    if coordinate_format == 'cartesian':
        pass
    elif coordinate_format == 'cylindrical':
        #(rho,z,phi) is assumed
        coordinate = cyl2cart(coordinate)
    elif coordinate_format == 'spherical':
        #(r,theta,phi) is assumed
        coordinate = sph2cart(coordinate)
    elif coordinate_format not in ('cartesian','sphercial','cylindrical'):
        raise ValueError('''
        Something went wrong with the given entry coordinate format''')

    return coordinate


#
#
#


def cyl2cart(coordinate):
    '''
    Converts cylindrical coordinates to Cartesian coordinates.

    INPUT:
    -----
     - coordinate: (3,npts)-array with cylindrical coordinates
                   [rho, z, phi] is assumed

    OUTPUT:
    -------
     - ouput : (3,npts)-array of cartesian coordinates

    Created on Oct 21 2016
    @author: V.Pelgrims

    '''
    x = coordinate[0] * np.cos(coordinate[2])
    y = coordinate[0] * np.sin(coordinate[2])
    z = coordinate[1]

    return np.array([x,y,z])


#
#
#


def sph2cart(coordinate):
    '''
    Translate spherical coordinates in cartesian coordinates.

    INPUT:
    -----
     - coordinate: (3,npts)-array with spherical coordinates
                   [r, theta, phi] is assumed

    OUTPUT:
    -------
     - ouput : (3,npts)-array of cartesian coordinates

    Created on Oct 21 2016
    @author: V.Pelgrims

    '''
    x = coordinate[0] * np.cos(coordinate[2]) * np.sin(coordinate[1])
    y = coordinate[0] * np.sin(coordinate[2]) * np.sin(coordinate[1])
    z = coordinate[0] * np.cos(coordinate[1])

    return np.array([x,y,z])


#
#
#


def cart2sph(coordinate):
    '''
    Converts Cartesian coordinates to spherical coordinates.

    INPUT:
    -----
     - coordinate: (3,npts)-array with cartesian coordinates
                   [x,y,z] is assumed

    OUTPUT:
    -------
     - ouput : (3,npts)-array of spherical coordinates
               [r, theta, phi] is assumed

    Created on Oct 21 2016
    @author: V.Pelgrims

    '''

    r = (coordinate[0]*coordinate[0] +
         coordinate[1]*coordinate[1] +
         coordinate[2]*coordinate[2])**.5

    theta = np.arctan2((coordinate[0]*coordinate[0] +
                        coordinate[1]*coordinate[1])**.5,
                       coordinate[2])

    phi = np.arctan2(coordinate[1],coordinate[0])
   
    return np.array([r,theta,phi])


#
#
#


def cart2cyl(coordinate):
    '''
    Converts Cartesian coordinates to cylindrical coordinates.

    INPUT:
    -----
     - coordinate: (3,npts)-array with cartesian coordinates
                   [x,y,z] is assumed

    OUTPUT:
    -------
     - ouput : (3,npts)-array of cylindrical coordinates
               [r, z, phi] is assumed

    Created on Oct 21 2016
    @author: V.Pelgrims

    '''

    rho = (coordinate[0]*coordinate[0] +
           coordinate[1]*coordinate[1])**.5

    z = coordinate[2]
    
    phi = np.arctan2(coordinate[1],coordinate[0])
   
    return np.array([rho,z,phi])


#
#
#


def sph2cyl(coordinate):
    '''
    Converts spherical coordinates to cylindrical coordinates.

    INPUT:
    ------
     - coordinate: (3,npts)-array with spherical coordinates
                   [r, theta, phi] is assumed

    OUTPUT:
    -------
     - outut: (3,npts)-array of cylindrical coordinates
              [rho, z, phi] is assumed

    Created on Oct 24 2016
    @author: V.Pelgrims
    '''

    rho = coordinate[0] * np.sin(coordinate[1])
    z = coordinate[0] * np.cos(coordinate[1])
    phi = coordinate[2]

    return np.array([rho,z,phi])


#
#
#


def cyl2sph(coordinate):
    '''
    Converts cylindrical coordinates to spherical coordinates.

    INPUT:
    ------
     - coordinate :(3,npts)-array with cylindrical coordinates
                   [rho, z, phi] is assumed

    OUTPUT:
    -------
     - output: (3,npts)-array of spherical coordinates
               [r, theta, phi] is assumed

    Created on Oct 24 2016
    @author: V.Pelgrims
    '''

    r = ( coordinate[0]*coordinate[0] + coordinate[1]*coordinate[1] )**.5
    theta = np.arctan2(coordinate[0],coordinate[1])
    phi = coordinate[2]

    return np.array([r,theta,phi])


#
#
#


def Grid2Template_indices(x_vector,XYZ_sun):
    """
    FUNCTION

        === Grid2Template_indices : fetching two space grids ===

    Grid2Template_indices(x_vector,XYZ_sun)


    This function assumes a Cartesian and cubic grid in one side and a
    another grid on the other size, such as the spherical one returned
    by gb.GalacticTemplate() for instance, but can be whatever you want.
    Then, for each element of the latter grid, it finds out the index
    corresponding to the spatially closest element in the cubic Cartesian
    grid.

    INPUT:
    ------
     - x_vector : (n,)-array with a linear and symmetric sampling that is
                  assumed to be used to generate a cubic and centered grid
                  that avoids the centre such as:
                      x_vector = np.arange(-x_max + x_step/2., x_max, x_step)
                      X,Y,Z = np.meshgrid(x_vector,x_vector,x_vector)
                      XYZ = np.asarray([X.reshape(-1),
                                        Y.reshape(-1),
                                        Z.reshape(-1)])
                  XYZ is (n**3,3)-array representing position vectors of the
                 cartesian grid.

     - XYZ_sun : (3,M)-array representing the M vectors of the second grid
                 that we want to connect to the Cartesian and centered one


    OUTPUT:
    ------
     - indices : (M,)-array with the indices that connect the closest
                 elements in XYZ to each XYZ_sun


    >>> You have XYZ_sun close to XYZ[indices]


    Created on Feb 09 2017
    @author: V.Pelgrims

    """

    x_step = x_vector[1]-x_vector[0]
    x_max = x_vector.max()
    len_x = len(x_vector)
    ind_x = np.round((XYZ_sun[0]+x_max)/x_step).astype(int)
    ind_y = np.round((XYZ_sun[1]+x_max)/x_step).astype(int)
    ind_z = np.round((XYZ_sun[2]+x_max)/x_step).astype(int)

    indices = ind_z + ind_y * len_x**2 + ind_x * len_x

    #need to account for eventual 'out of the grid' cells.
    #this appends when ind_* > len_x for * == x,y,z
    #in this case it is better to flag the unexisting cells of the grid
    out = (ind_x>len_x or ind_y>len_x or ind_z>len_x)
    if np.sum(out)>0:
        print('''
        The Template runs out of the grid of reference.
        You will need to clear out the indices to proceed further.
        The value assigned to the 'out of the gird' cells is''', len_x**3)

        indices[np.where(out)] = len_x**3
    #
    
    return indices


#
#
#


def __gal2sun_vector(vector_field_G,dotproducts):
    """
    FUNCTION
        === __gal2sun_vector : convert vectors from Galactic to Sun RF ===

    INPUT:
    ------
     - vector_field_G : (3,npts)-array with vector components expressed
                        in Galactocentric coordinate system

     - dotproducts : (9,npts)-array with dotproducts between basis vectors
                     of the two coordinate system. Expected to be given by:
                      _,__,dotproducts = gb.GalacticTemplate() for example

    OUTPUT:
    -------
     - vector_field_S : (3,npts)-array with vector components expressed
                        in Heliocentric coordinate system

    NOTE :
    This is independent of the choice of coordinate system but works well
    with the assumption that dotproducts contains exactly what is expected.
    See gb.GalacticTemplate() for the expected structure

    Created on Oct 26 2016
    @author: V.Pelgrims

    """

    B_rho = vector_field_G[0]
    B_z = vector_field_G[1]
    B_phi = vector_field_G[2]
    dots = dotproducts

    B_r = B_rho * dots[0] + B_z * dots[3] + B_phi * dots[6]
    B_t = B_rho * dots[1] + B_z * dots[4] + B_phi * dots[7]
    B_p = B_rho * dots[2] + B_z * dots[5] + B_phi * dots[8]


    return np.array([B_r,B_t,B_p])


#
#
#

################################ E N D ####################################
