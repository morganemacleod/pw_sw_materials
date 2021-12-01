import numpy as np
from astropy.io import ascii
import athena_read as ar

def read_trackfile(fn,m1=0,m2=0):
    orb=ascii.read(fn)
    print( "reading orbit file for planet wind simulation...")
    if m1==0:
        m1 = orb['m1']
    if m2==0:
        m2 = orb['m2']

    orb['sep'] = np.sqrt(orb['x']**2 + orb['y']**2 + orb['z']**2)

    orb['r'] = np.array([orb['x'],orb['y'],orb['z']]).T
    orb['rhat'] = np.array([orb['x']/orb['sep'],orb['y']/orb['sep'],orb['z']/orb['sep']]).T

    orb['v'] = np.array([orb['vx'],orb['vy'],orb['vz']]).T
    orb['vmag'] = np.linalg.norm(orb['v'],axis=1)
    orb['vhat'] = np.array([orb['vx']/orb['vmag'],orb['vy']/orb['vmag'],orb['vz']/orb['vmag']]).T

    orb['xcom'] = m2*orb['x']/(m1+m2)
    orb['ycom'] = m2*orb['y']/(m1+m2)
    orb['zcom'] = m2*orb['z']/(m1+m2)
    
    orb['vxcom'] = m2*orb['vx']/(m1+m2)
    orb['vycom'] = m2*orb['vy']/(m1+m2)
    orb['vzcom'] = m2*orb['vz']/(m1+m2)
    
    orb['rcom'] = np.array([orb['xcom'],orb['ycom'],orb['zcom']]).T
    orb['vcom'] = np.array([orb['vxcom'],orb['vycom'],orb['vzcom']]).T
    
    return orb


def read_data(fn,orb,
              m1=0,m2=0,rsoft2=0.1,level=0,
              get_cartesian=True,get_cartesian_vel=True,
             x1_min=None,x1_max=None,
             x2_min=None,x2_max=None,
             x3_min=None,x3_max=None,
              gamma=5./3.,
              pole_dir=2,
              dens_pres_scale_factor=1.0):
    """ Read spherical data and reconstruct cartesian mesh for analysis/plotting """
    
    print("read_data...reading file",fn)
    
    
    d = ar.athdf(fn,level=level,subsample=True,
                 x1_min=x1_min,x1_max=x1_max,
                 x2_min=x2_min,x2_max=x2_max,
                 x3_min=x3_min,x3_max=x3_max) # approximate arrays by subsampling if level < max
    print(" ...file read, constructing arrays")
    print(" ...gamma=",gamma)

    # SCALE DENSITY AND PRESSURE
    d['rho'] = dens_pres_scale_factor*d['rho']
    d['press'] = dens_pres_scale_factor*d['press']

    
    # current time
    t = d['Time']
    # get properties of orbit
    rcom,vcom = rcom_vcom(orb,t)

    if m1==0:
        m1 = np.interp(t,orb['time'],orb['m1'])
    if m2==0:
        m2 = np.interp(t,orb['time'],orb['m2'])

    data_shape = (len(d['x3v']),len(d['x2v']),len(d['x1v']))
   
    d['gx1v']=np.broadcast_to(d['x1v'],(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
    d['gx2v']=np.swapaxes(np.broadcast_to(d['x2v'],(len(d['x3v']),len(d['x1v']),len(d['x2v'])) ),1,2)
    d['gx3v']=np.swapaxes(np.broadcast_to(d['x3v'],(len(d['x1v']),len(d['x2v']),len(d['x3v'])) ) ,0,2 )
    
    ####
    # GET THE VOLUME 
    ####
    
    ## dr, dth, dph
    d1 = d['x1f'][1:] - d['x1f'][:-1]
    d2 = d['x2f'][1:] - d['x2f'][:-1]
    d3 = d['x3f'][1:] - d['x3f'][:-1]
    
    gd1=np.broadcast_to(d1,(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
    gd2=np.swapaxes(np.broadcast_to(d2,(len(d['x3v']),len(d['x1v']),len(d['x2v'])) ),1,2)
    gd3=np.swapaxes(np.broadcast_to(d3,(len(d['x1v']),len(d['x2v']),len(d['x3v'])) ) ,0,2 )
    
    
    # AREA / VOLUME 
    sin_th = np.sin(d['gx2v'])
    d['dA'] = d['gx1v']**2 * sin_th * gd2*gd3
    d['dvol'] = d['dA'] * gd1
    
    # free up d1,d2,d3
    del d1,d2,d3
    del gd1,gd2,gd3
    
    
    ### 
    # CARTESIAN VALUES
    ###
    if(get_cartesian or get_torque or get_energy):
        print("...getting cartesian arrays...")
        # angles
        cos_th = np.cos(d['gx2v'])
        sin_ph = np.sin(d['gx3v'])
        cos_ph = np.cos(d['gx3v']) 
        
        # cartesian coordinates
        if(pole_dir==2):
            d['x'] = d['gx1v'] * sin_th * cos_ph 
            d['y'] = d['gx1v'] * sin_th * sin_ph 
            d['z'] = d['gx1v'] * cos_th
        if(pole_dir==0):
            d['y'] = d['gx1v'] * sin_th * cos_ph 
            d['z'] = d['gx1v'] * sin_th * sin_ph 
            d['x'] = d['gx1v'] * cos_th

        if(get_cartesian_vel):
            # cartesian velocities
            if(pole_dir==2):
                d['vx'] = sin_th*cos_ph*d['vel1'] + cos_th*cos_ph*d['vel2'] - sin_ph*d['vel3'] 
                d['vy'] = sin_th*sin_ph*d['vel1'] + cos_th*sin_ph*d['vel2'] + cos_ph*d['vel3'] 
                d['vz'] = cos_th*d['vel1'] - sin_th*d['vel2']  
            if(pole_dir==0):
                d['vy'] = sin_th*cos_ph*d['vel1'] + cos_th*cos_ph*d['vel2'] - sin_ph*d['vel3'] 
                d['vz'] = sin_th*sin_ph*d['vel1'] + cos_th*sin_ph*d['vel2'] + cos_ph*d['vel3'] 
                d['vx'] = cos_th*d['vel1'] - sin_th*d['vel2'] 
            
        del cos_th, sin_th, cos_ph, sin_ph
    
    return d




def read_data_for_rt(fn,orb,
              m1=0,m2=0,rsoft2=0.1,level=0,
              get_cartesian=True,get_cartesian_vel=True,
             x1_min=None,x1_max=None,
             x2_min=None,x2_max=None,
             x3_min=None,x3_max=None,
              gamma=5./3.,
             pole_dir=2,
             dens_pres_scale_factor=1.0):
    """ Read spherical data and reconstruct cartesian mesh for analysis/plotting """
    
    print("read_data...reading file",fn)
    
    
    d = ar.athdf(fn,level=level,subsample=True,
                 x1_min=x1_min,x1_max=x1_max,
                 x2_min=x2_min,x2_max=x2_max,
                 x3_min=x3_min,x3_max=x3_max,
                 quantities=['rho','press','vel1','vel2','vel3','r0']) # approximate arrays by subsampling if level < max
    print(" ...file read, constructing arrays")
    print(" ...gamma=",gamma)

    # SCALE DENSITY AND PRESSURE
    d['rho'] = dens_pres_scale_factor*d['rho']
    d['press'] = dens_pres_scale_factor*d['press']

    # current time
    t = d['Time']
    # get properties of orbit
    rcom,vcom = rcom_vcom(orb,t)

    if m1==0:
        m1 = np.interp(t,orb['time'],orb['m1'])
    if m2==0:
        m2 = np.interp(t,orb['time'],orb['m2'])

    #data_shape = (len(d['x3v']),len(d['x2v']),len(d['x1v']))
   
    d['gx1v']=np.broadcast_to(d['x1v'],(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
    d['gx2v']=np.swapaxes(np.broadcast_to(d['x2v'],(len(d['x3v']),len(d['x1v']),len(d['x2v'])) ),1,2)
    d['gx3v']=np.swapaxes(np.broadcast_to(d['x3v'],(len(d['x1v']),len(d['x2v']),len(d['x3v'])) ) ,0,2 )
    
    ####
    # GET THE VOLUME 
    ####
    
    ## dr, dth, dph
    #d1 = d['x1f'][1:] - d['x1f'][:-1]
    #d2 = d['x2f'][1:] - d['x2f'][:-1]
    #d3 = d['x3f'][1:] - d['x3f'][:-1]
    
    #gd1=np.broadcast_to(d1,(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
    #gd2=np.swapaxes(np.broadcast_to(d2,(len(d['x3v']),len(d['x1v']),len(d['x2v'])) ),1,2)
    #gd3=np.swapaxes(np.broadcast_to(d3,(len(d['x1v']),len(d['x2v']),len(d['x3v'])) ) ,0,2 )
    
    
    # AREA / VOLUME 
    sin_th = np.sin(d['gx2v'])
    #d['dA'] = d['gx1v']**2 * sin_th * gd2*gd3
    #d['dvol'] = d['dA'] * gd1
    
    # free up d1,d2,d3
    #del d1,d2,d3
    #del gd1,gd2,gd3
    
    
    ### 
    # CARTESIAN VALUES
    ###
    if(get_cartesian or get_torque or get_energy):
        print("...getting cartesian arrays...")
        # angles
        cos_th = np.cos(d['gx2v'])
        sin_ph = np.sin(d['gx3v'])
        cos_ph = np.cos(d['gx3v'])
        del d['gx2v'],d['gx3v']
                
        # cartesian coordinates
        if(pole_dir==2):
            d['x'] = d['gx1v'] * sin_th * cos_ph 
            d['y'] = d['gx1v'] * sin_th * sin_ph 
            d['z'] = d['gx1v'] * cos_th
      
        if(get_cartesian_vel):
            # cartesian velocities
            if(pole_dir==2):
                d['vx'] = sin_th*cos_ph*d['vel1'] + cos_th*cos_ph*d['vel2'] - sin_ph*d['vel3'] 
                d['vy'] = sin_th*sin_ph*d['vel1'] + cos_th*sin_ph*d['vel2'] + cos_ph*d['vel3'] 
                d['vz'] = cos_th*d['vel1'] - sin_th*d['vel2']  

            
        del d['vel1'],d['vel2'],d['vel3']    
        del cos_th, sin_th, cos_ph, sin_ph
    
    return d





def get_midplane_theta(myfile,level=0):
    dblank=ar.athdf(myfile,level=level,quantities=[],subsample=True)

    # get closest to midplane value
    return dblank['x2v'][ np.argmin(np.abs(dblank['x2v']-np.pi/2.) ) ]


def get_plot_array_midplane(arr):
    return np.append(arr,[arr[0]],axis=0)


def rcom_vcom(orb,t):
    """pass a pm_trackfile.dat that has been read, time t"""
    rcom =  np.array([np.interp(t,orb['time'],orb['rcom'][:,0]),
                  np.interp(t,orb['time'],orb['rcom'][:,1]),
                  np.interp(t,orb['time'],orb['rcom'][:,2])])
    vcom =  np.array([np.interp(t,orb['time'],orb['vcom'][:,0]),
                  np.interp(t,orb['time'],orb['vcom'][:,1]),
                  np.interp(t,orb['time'],orb['vcom'][:,2])])
    
    return rcom,vcom

def pos_secondary(orb,t):
    x2 = np.interp(t,orb['time'],orb['x'])
    y2 = np.interp(t,orb['time'],orb['y'])
    z2 = np.interp(t,orb['time'],orb['z'])
    return x2,y2,z2


### Ray tracing
from scipy.interpolate import RegularGridInterpolator

def get_interp_function(d,var):
    dph = np.gradient(d['x3v'])[0]
    x3v = np.append(d['x3v'][0]-dph,d['x3v'])
    x3v = np.append(x3v,x3v[-1]+dph)
    
    var_data = np.append([d[var][-1]],d[var],axis=0)
    var_data = np.append(var_data,[var_data[0]],axis=0)
    
    var_interp = RegularGridInterpolator((x3v,d['x2v'],d['x1v']),var_data,bounds_error=False,method='nearest')
    return var_interp



def cart_to_polar(x,y,z):
    """ returns phi in range 0-2pi"""
    r = np.sqrt(x**2 + y**2 +z**2)
    th = np.arccos(z/r)
    phi=np.where(np.arctan2(y,x)<0,np.arctan2(y,x) + 2.0*np.pi, np.arctan2(y,x) )
    return phi,th,r


def get_ray(planet_pos, ydart, zdart, azim_angle, pol_angle, rstar, rplanet, fstep, inner_lim,outer_lim):

    xdart = np.sqrt(1.0 - ydart*ydart - zdart*zdart)
    dart1 = np.array([xdart*np.sign(planet_pos[0]), ydart, zdart])*rstar
    
    rotz = np.array([[np.cos(-1.0*azim_angle),-np.sin(-1.0*azim_angle),0.0],
                     [np.sin(-1.0*azim_angle),np.cos(-1.0*azim_angle),0.0],
                     [0.0,0.0,1.0]])
    rotY = np.array([[np.cos(-pol_angle),0.0,np.sin(-pol_angle)],
                     [0.0,1.0,0.0],
                     [-np.sin(-pol_angle),0.0,np.cos(-pol_angle)]])     


    # define origin
    origin = np.matmul( dart1, rotz)
   
    # ray 
    ray={}

    """
    fmin = 0.5
    #points= np.linspace(3*rstar, outer_lim, npoints) 
    length1 = np.sqrt((inner_lim*np.sign(planet_pos[0]) -planet_pos[0])*(inner_lim*np.sign(planet_pos[0]) -planet_pos[0]))
    length2 = np.sqrt((outer_lim*np.sign(planet_pos[0]) -planet_pos[0])*(outer_lim*np.sign(planet_pos[0]) -planet_pos[0]))
    points_aux1 = -1.0*np.logspace(np.log10(length1),np.log10(fmin*rplanet), np.int(npoints/2) ) + fmin*rplanet + abs(planet_pos[0])
    points_aux2 = np.logspace(np.log10(fmin*rplanet), np.log10(length2), np.int(npoints/2)  ) - fmin*rplanet + abs(planet_pos[0])
    #print (points_aux1, points_aux2)
    points = np.concatenate([points_aux1[0:-1], points_aux2])
    #print points
    """

    #pxrot = np.matmul(planet_pos,rotz)
    #print("planet_pos =",planet_pos,"pxrot = ",pxrot)
    

    points = [inner_lim]
    while(points[-1] < outer_lim):
        x = points[-1]*np.cos(azim_angle)*np.cos(pol_angle) + origin[0]
        y = points[-1]*np.sin(azim_angle)*np.cos(pol_angle) + origin[1]
        z = points[-1]*np.sin(pol_angle) + origin[2]
        dpl = np.sqrt((x-planet_pos[0])**2 + (y-planet_pos[1])**2 + (z-planet_pos[2])**2 )
        points.append(points[-1] + fstep*dpl)
        

    points = np.array(points)
        
    ray['dl'] = points[1:]-points[0:-1]
    ray['l'] = 0.5*(points[1:]+points[0:-1])
    ray['x'] = ray['l']*np.cos(azim_angle)*np.cos(pol_angle) + origin[0] 
    ray['y'] = ray['l']*np.sin(azim_angle)*np.cos(pol_angle) + origin[1]
    ray['z'] = ray['l']*np.sin(pol_angle) + origin[2]
    #print (ray['x'],ray['y'])
    
    # spherical polar
    ray['phi'],ray['theta'],ray['r'] = cart_to_polar(ray['x'],ray['y'],ray['z'])

    print(" ... ray has ",len(ray['r']),"points")
    
    return ray
