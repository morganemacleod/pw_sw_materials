import argparse
from planet_wind_constants import *
from scipy.special import wofz
import time
from scipy.optimize import newton
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
import planet_wind_utils_v6 as pw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import deepdish as dd # pip install deepdish
mpl.use('Agg')

#from scipy.optimize import root


def I(mu, ld1, ld2):
    return np.where(mu == 0.0, 0.0, (1. - ld1 * (1. - mu) - ld2 * (1. - mu)**2))


def Voigt(x, alpha, gamma):
    sigma = alpha / np.sqrt(2.0*np.log(2.0))
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2.0)))/sigma/np.sqrt(2.0*np.pi)


def New_get_interp_function(d, var):
    dph = np.gradient(d['x3v'])[0]
    x3v = np.append(d['x3v'][0]-dph, d['x3v'])
    x3v = np.append(x3v, x3v[-1]+dph)

    var_data = np.append([var[-1]], var, axis=0)
    var_data = np.append(var_data, [var_data[0]], axis=0)

    var_interp = RegularGridInterpolator(
        (x3v, d['x2v'], d['x1v']), var_data, bounds_error=True)
    return var_interp


def initial_guess(floor_val = 1.e-30):
    
    # initial guess for electron number density
    ne_init = 0.1*d['rho']*Xuni/c.mp

    #print("ne_init:",np.sum(ne_init))

    # initial guess for number density of H I
    nh1 = np.copy(d['rho']*Xuni/c.mp - ne_init)

    nhe1 = np.copy(d['rho']*Yuni/(4.0*c.mp))
    nhe3 = np.ones_like(nhe1)*floor_val

    #print("nh1:",np.sum(nh1))
    #print("nhe1:",np.sum(nhe1))
    #print("nhe3:",np.sum(nhe3))

    # initial guess for optical depth in each cell
    tau_integral = np.clip(nh1*sigma_photo_nu0*d['gx1v'],floor_val,100)
    tau1_integral = np.clip(nhe1*sigma_photo_nu1*d['gx1v'],floor_val,100)
    tau3_integral = np.clip(nhe3*sigma_photo_nu3*d['gx1v'],floor_val,100)

    #print ("tau:", np.sum(tau_integral)  )
    #print ("tau1:", np.sum(tau1_integral)  )
    #print ("tau3:", np.sum(tau3_integral)  )
    
    ne = apply_lim( phi*np.exp(-tau_integral)/(2.0*alpha) * (np.sqrt(1.0 + apply_lim(4.0*d['rho']*Xuni*alpha/(c.mp*phi*np.exp(-tau_integral)), floor_val)) - 1.0), floor_val)  # electrons from hydrogen ionization


    nh1= apply_lim(d['rho']*Xuni/c.mp - ne, floor_val)  # neutral hydrogen

    return tau_integral, tau1_integral, tau3_integral, ne, nh1


def new_guess(tau_integral, tau1_integral, tau3_integral, ne, nh1,
              floor_val = 1.e-30):
    
    # H number desnity
    nh_plus = apply_lim( phi*np.exp(-1.0*tau_integral)*d['rho']*Xuni/c.mp / (phi*np.exp(-1.0*tau_integral) + ne*alpha), floor_val ) # ionized hydrogen
    print('diff nh1  (med, av):',np.median(d['rho']*Xuni/c.mp - nh_plus - nh1), np.average(d['rho']*Xuni/c.mp - nh_plus - nh1))
    nh1 = apply_lim(d['rho']*Xuni/c.mp - nh_plus, floor_val) 	# neutral hydrogen
    
    # helium number densities
    f3 = apply_lim((ne*alpha1 - (ne*alpha3)*(ne*alpha1 + phi1*np.exp(-1.0*tau1_integral) + ne*q13a)/(ne*alpha3 - ne*q13a)) / (ne*alpha1 - A31 - ne*q31a - ne*q31b - nh1*Q31 -
                                                                                                                              (ne*alpha1 + phi1*np.exp(-1.0*tau1_integral) + ne*q13a)*(ne*alpha3 + A31 + phi3*np.exp(-1.0*tau3_integral) + ne*q31a + ne*q31b + nh1*Q31)/(ne*alpha3 - ne*q13a)),
                   floor_val)
    
        
    f1 = apply_lim((ne*alpha3 - f3*(ne*alpha3 + A31 + phi3*np.exp(-tau3_integral) + ne*q31a + ne*q31b + nh1*Q31)) / (ne*alpha3 - ne*q13a), floor_val)
    
    
    nhe1 = apply_lim(f1*d['rho']*Yuni/(4.0*c.mp), floor_val) 	# ground-state helium
    nhe3 = apply_lim(f3*d['rho']*Yuni/(4.0*c.mp), floor_val)  # metastable-state helium
    # ionized helium
    nhe_plus = apply_lim((1.0 - f1 - f3)*d['rho']*Yuni/(4.0*c.mp), floor_val)

    # optical depth
    tau_integral = np.clip(nh1*sigma_photo_nu0*d['gx1v'],floor_val,100)
    tau1_integral = np.clip(nhe1*sigma_photo_nu1*d['gx1v'],floor_val,100)
    tau3_integral = np.clip(nhe3*sigma_photo_nu3*d['gx1v'],floor_val,100)
    
    ne = np.copy(nh_plus + nhe_plus)

    return ne, nh1, nh_plus, nhe1, nhe3, nhe_plus, tau_integral, tau1_integral, tau3_integral 


def generate_random(N_mc):
    theta = 2*np.pi*np.random.random_sample(N_mc)
    r = np.sqrt(np.random.random_sample(N_mc))

    yrandom = r*np.cos(theta)
    zrandom = r*np.sin(theta)

    return yrandom, zrandom


def sum_tau_LOS(ray):
    nu_array = np.broadcast_to(nu, (len(ray['vx']), len(nu)))
    vx_array = np.broadcast_to(ray['vx'], (len(nu), len(ray['vx']))).T
    vy_array = np.broadcast_to(ray['vy'], (len(nu), len(ray['vx']))).T
    nhe3_array = np.broadcast_to(ray['nhe3'], (len(nu), len(ray['vx']))).T
    dl_array = np.broadcast_to(ray['dl'], (len(nu), len(ray['vx']))).T
    da1_array = np.broadcast_to(ray['da1'], (len(nu), len(ray['vx']))).T
    da2_array = np.broadcast_to(ray['da2'], (len(nu), len(ray['vx']))).T
    da3_array = np.broadcast_to(ray['da3'], (len(nu), len(ray['vx']))).T

    delta_u1 = np.copy(c.c*(nu_array-nu1)/nu1 + (vx_array *
                                                 np.cos(azim_angle) + vy_array*np.sin(azim_angle))*np.sign(x2))
    xx1 = np.copy(delta_u1*nu1/c.c)
    delta_u2 = np.copy(c.c*(nu_array-nu2)/nu2 + (vx_array *
                                                 np.cos(azim_angle) + vy_array*np.sin(azim_angle))*np.sign(x2))
    xx2 = np.copy(delta_u2*nu2/c.c	)
    delta_u3 = np.copy(c.c*(nu_array-nu3)/nu3 + (vx_array *
                                                 np.cos(azim_angle) + vy_array*np.sin(azim_angle))*np.sign(x2))
    xx3 = np.copy(delta_u3*nu3/c.c)

    tauLOS1 = np.sum(nhe3_array*dl_array*cs1 *
                     Voigt(xx1, da1_array, natural_gamma), axis=0)
    tauLOS2 = np.sum(nhe3_array*dl_array*cs2 *
                     Voigt(xx2, da2_array, natural_gamma), axis=0)
    tauLOS3 = np.sum(nhe3_array*dl_array*cs3 *
                     Voigt(xx3, da3_array, natural_gamma), axis=0)

    return tauLOS1, tauLOS2, tauLOS3


def MC_ray(dart):
    """ computes sum of tau along LOS of a ray defined by integer 'dart' """
    ydart = yrandom[dart]
    zdart = zrandom[dart]
    print('dart: ', dart, ydart, zdart)

    ray = pw.get_ray(planet_pos=(x2, y2, z2),
                     ydart=ydart,
                     zdart=zdart,
                     azim_angle=azim_angle,
                     pol_angle=0.0,
                     rstar=rad_star,
                     rplanet=rp,
                     npoints=nraypoints,
                     inner_lim=in_lim,
                     outer_lim=out_lim)


    # boolean, whether the ray intersects the planet
    throughplanet = (np.amin( np.sqrt( (ray['x']-x2)**2 + (ray['y']-y2)**2 + (ray['z']-z2)**2 ) ) < rp)

    
    ray['nhe3'] = nhe3_interp((ray['phi'], ray['theta'], ray['r']))
    ray['vx'] = vx_interp((ray['phi'], ray['theta'], ray['r']))
    ray['vy'] = vy_interp((ray['phi'], ray['theta'], ray['r']))
    ray['vz'] = vz_interp((ray['phi'], ray['theta'], ray['r']))
    ray['temp'] = temp_interp((ray['phi'], ray['theta'], ray['r']))
    ray['da1'] = np.sqrt(2.0*np.log(2.0))*nu1 * \
        np.sqrt(0.25*c.kB*ray['temp']/c.mp)/c.c
    ray['da2'] = np.sqrt(2.0*np.log(2.0))*nu2 * \
        np.sqrt(0.25*c.kB*ray['temp']/c.mp)/c.c
    ray['da3'] = np.sqrt(2.0*np.log(2.0))*nu3 * \
        np.sqrt(0.25*c.kB*ray['temp']/c.mp)/c.c

    tauLOS1, tauLOS2, tauLOS3 = sum_tau_LOS(ray)

    if throughplanet:
        expfac = np.zeros_like(tauLOS1)
        print ("...planet crossing ray!")
    else:
        expfac = np.exp(-tauLOS1 - tauLOS2 - tauLOS3) 

    return expfac


def make_plots(it_num, jcoord):
    # temp
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(d['x'][:, jcoord, :], d['y'][:, jcoord, :], np.log10(
        d['temp'][:, jcoord, :]), cmap=plt.cm.Spectral, vmax=6, vmin=2,shading='auto')
    plt.colorbar(label=r"T [K]")
    plt.axis('equal')
    plt.plot(x2, y2, 'w*')
    plt.xlim(-lim/2-a, lim/2-a)
    plt.ylim(-lim/2, lim/2)
    plt.savefig(base_dir+'temp_'+str(it_num+1)+'.png')
    plt.close()

    # tau hydrogen
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(d['x'][:, jcoord, :], d['y'][:, jcoord, :], np.log10(
        tau_integral[:, jcoord, :]), cmap=plt.cm.magma, vmax=-2.0, vmin=-7.0,shading='auto')
    plt.colorbar(label=r"$\tau$")
    plt.axis('equal')
    plt.plot(x2, y2, 'w*')
    plt.xlim(-lim/2-a, lim/2-a)
    plt.ylim(-lim/2, lim/2)
    plt.savefig(base_dir+'tauHI_'+str(it_num+1)+'.png')
    plt.close()

    # tau helium singlet
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(d['x'][:, jcoord, :], d['y'][:, jcoord, :], np.log10(
        tau1_integral[:, jcoord, :]), cmap=plt.cm.magma, vmax=-2.0, vmin=-7.0,shading='auto')
    plt.colorbar(label=r"$\tau1$")
    plt.axis('equal')
    plt.plot(x2, y2, 'w*')
    plt.xlim(-lim/2-a, lim/2-a)
    plt.ylim(-lim/2, lim/2)
    plt.savefig(base_dir+'tauHe1_'+str(it_num+1)+'.png')
    plt.close()

    # tau helium triplet
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(d['x'][:, jcoord, :], d['y'][:, jcoord, :], np.log10(
        tau3_integral[:, jcoord, :]), cmap=plt.cm.magma, vmax=0.0, vmin=-7.0,shading='auto')
    plt.colorbar(label=r"$\tau3$")
    plt.axis('equal')
    plt.plot(x2, y2, 'w*')
    plt.xlim(-lim/2-a, lim/2-a)
    plt.ylim(-lim/2, lim/2)
    plt.savefig(base_dir+'tauHe3_'+str(it_num+1)+'.png')
    plt.close()

    # electron number density
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(d['x'][:, jcoord, :], d['y'][:, jcoord, :], np.log10(
        ne[:, jcoord, :]), cmap=plt.cm.magma, vmax=10.0, vmin=-5.0,shading='auto')
    plt.colorbar(label=r"ne")
    plt.axis('equal')
    plt.plot(x2, y2, 'w*')
    plt.xlim(-lim/2-a, lim/2-a)
    plt.ylim(-lim/2, lim/2)
    plt.savefig(base_dir+'ne_'+str(it_num+1)+'.png')
    plt.close()

    # H I number density
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(d['x'][:, jcoord, :], d['y'][:, jcoord, :], np.log10(
        nh1[:, jcoord, :]), cmap=plt.cm.magma, vmax=10.0, vmin=-5.0,shading='auto')
    plt.colorbar(label=r"nh1")
    plt.axis('equal')
    plt.plot(x2, y2, 'w*')
    plt.xlim(-lim/2-a, lim/2-a)
    plt.ylim(-lim/2, lim/2)
    plt.savefig(base_dir+'nh1_'+str(it_num+1)+'.png')
    plt.close()

    # He I singlet number density
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(d['x'][:, jcoord, :], d['y'][:, jcoord, :], np.log10(
        nhe1[:, jcoord, :]), cmap=plt.cm.magma, vmax=10.0, vmin=-5.0,shading='auto')
    plt.colorbar(label=r"nhe1")
    plt.axis('equal')
    plt.plot(x2, y2, 'w*')
    plt.xlim(-lim/2-a, lim/2-a)
    plt.ylim(-lim/2, lim/2)
    plt.savefig(base_dir+'nhe1_'+str(it_num+1)+'.png')
    plt.close()

    # He I triplet number density
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(d['x'][:, jcoord, :], d['y'][:, jcoord, :], np.log10(
        nhe3[:, jcoord, :]), cmap=plt.cm.magma, vmax=5.0, vmin=-7.0,shading='auto')
    plt.colorbar(label=r"nhe3")
    plt.axis('equal')
    plt.plot(x2, y2, 'w*')
    plt.xlim(-lim/2-a, lim/2-a)
    plt.ylim(-lim/2, lim/2)
    plt.savefig(base_dir+'nhe3_'+str(it_num+1)+'.png')
    plt.close()


def make_side_plots(it_num, icoord):
    # tau
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(d['x'][icoord, :, :], d['z'][icoord, :, :], np.log10(
        tau_integral[icoord, :, :]), cmap=plt.cm.magma, vmax=0.0, vmin=-7.0,shading='auto')
    plt.colorbar(label=r"$\tau$")
    plt.axis('equal')
    plt.plot(x2, y2, 'w*')
    plt.xlim(-lim/2-a, lim/2-a)
    plt.ylim(-lim/2, lim/2)
    plt.savefig(base_dir+'stau05_'+str(it_num+1)+'.png')
    plt.close()

    # H I number density
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(d['x'][icoord, :, :], d['z'][icoord, :, :], np.log10(
        nh1[icoord, :, :]), cmap=plt.cm.magma, vmax=10.0, vmin=-5.0,shading='auto')
    plt.colorbar(label=r"nh1")
    plt.axis('equal')
    plt.plot(x2, y2, 'w*')
    plt.xlim(-lim/2-a, lim/2-a)
    plt.ylim(-lim/2, lim/2)
    plt.savefig(base_dir+'snh05_'+str(it_num+1)+'.png')
    plt.close()


def get_BC_prop(d):
    bc_ind = np.argmin(d['rp'].flatten())
    print("rp/Rp=", d['rp'].flatten()[bc_ind] / rp)
    pp = d['press'].flatten()[bc_ind]
    rhop = d['rho'].flatten()[bc_ind]
    Bern = gamma/(gamma-1.0)*pp/rhop - c.G*m2/rp
    print("Bern = ", Bern)
    K = pp/rhop**gamma
    print("K =", K)
    lambda_planet = c.G*m2*rhop/(gamma*pp*rp)
    print("lambda = ", lambda_planet)
    mdot_est = np.pi*rhop * \
        np.sqrt(c.G*m2*(rp*lambda_planet)**3)*np.exp(1.5-lambda_planet)
    print("mdot_est =", mdot_est)

    return Bern, K, lambda_planet, mdot_est


def parker_fv(v, r):
    return v*np.exp(-0.5*v*v/(vS*vS))/vS - rS*rS*np.exp(-2.0*rS/r + 1.5)/(r*r)


def parker_frho(r, v):
    return rhoS*np.exp(2.0*rS/r - 1.5 - 0.5*v*v/(vS*vS))


def get_Parker_rho_v_func(r_out=1.e11, num=1000):
    vguess = 1.e5
    r_aux = np.linspace(1.0/(0.9*rp), 1/r_out, num)
    r = 1.0/r_aux
    res_v = np.zeros(num)
    res_rho = np.zeros(num)

    for i in range(num):
        if i > 0:
            vguess = 1.001*res_v[i-1]
        res_v[i] = newton(parker_fv, vguess, args=(r[i],))
        res_rho[i] = parker_frho(r[i], res_v[i])

    rho_func = interp1d(r, res_rho, fill_value='extrapolate')
    v_func = interp1d(r, res_v, fill_value='extrapolate')

    plt.figure()
    plt.plot(r/rp, res_v/1.e5, '-')
    plt.plot(r/rp, v_func(r)/1.e5, '--')
    plt.axvline(1, color='grey')
    plt.axvline(rS/rp, color='grey', ls='--')
    plt.loglog()
    plt.savefig(base_dir+"parker_solution.png")

    return rho_func, v_func

def phi_planet(x,y):
    ph = np.arctan2(y-y2,x-x2)
    return np.where(ph<0,ph+2*np.pi,ph)


def apply_Parker(d):
    # refill arrays with analytic solution...
    R2 = np.sqrt( (d['x']-x2)**2 + (d['y']-y2)**2 )
    #Rcyl = np.sqrt(d['x']**2 + d['y']**2) 
    php = phi_planet(d['x'],d['y'])
    thp = np.arccos( (d['z']-z2)/d['rp'] )
    d['rho'] = density(d['rp'])
    d['press'] = K*d['rho']**gamma
    # constant angular momentum of the surface (in rot frame)
    d['vx'] = velocity(d['rp'])*(d['x']-x2)/d['rp'] - np.sin(php)*(omega_planet*(np.sin(thp)*rp)**2 / R2 -Omega_orb*R2)  
    d['vy'] = velocity(d['rp'])*(d['y']-y2)/d['rp'] + np.cos(php)*(omega_planet*(np.sin(thp)*rp)**2 / R2 -Omega_orb*R2)
    d['vz'] = velocity(d['rp'])*(d['z']-z2)/d['rp']

    return d

def apply_lim(arr,fv):
    return np.clip(arr,fv,1./fv)


####################################################

# set some global options
plt.rcParams['figure.figsize'] = (6, 5)
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.borderpad'] = 0.2
plt.rcParams['legend.labelspacing'] = 0.2
plt.rcParams['legend.handletextpad'] = 0.2
plt.rcParams['font.family'] = 'stixgeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 16

######################################################

parser = argparse.ArgumentParser(
    description='Read input/output directories, MC ray properties, example usage: "python Planet_Wind_Ray_Star_Origin23.py --base_dir ~/Dropbox/PlanetWind/Analysis/testdata/ --snapshot PW_W107.out1.00100.athdf --level 1 --N_mc 100 --N_raypoints 200 --angles 0" ')

parser.add_argument("--base_dir", help="data directory (should end with / )")
parser.add_argument("--snapshot", help="filename of snapshot to be processed")
parser.add_argument("--angles", type=float, nargs='+',
                    help="angles at which to perform the spectral synthesis (radians, mid-transit=0)", required=True)
parser.add_argument("--level", default=1, type=int,
                    help="refinement level to read the snapshot at")
parser.add_argument("--N_mc", default=1000, type=int, help="number of MC rays")
parser.add_argument("--N_raypoints", default=200, type=int,
                    help="number of points along a ray")
parser.add_argument("--parker", action='store_true',
                    help="apply the analytic solution to the data: True/False")
parser.add_argument("--plots", action='store_true', help="plot slices")
parser.add_argument("--savedata", action='store_true', help="save slice as hdf5 file")
parser.add_argument("--scale",default=1.0,type=float,help="scale density and pressure by this factor")


args = parser.parse_args()
base_dir = args.base_dir
snapshot = args.snapshot
mylevel = args.level
N_mc = args.N_mc
nraypoints = args.N_raypoints
angles = args.angles
parker = args.parker
plots = args.plots
savedata = args.savedata
dens_pres_scale = args.scale

# small value
fv = 1.e-30

# read file
start_read_time = time.time()


orb = pw.read_trackfile(base_dir+"pm_trackfile.dat")

# NOTE: needs to be a full 3D output, not a slice!!!
myfile = base_dir+snapshot
out_lim = 1.2e12 #2.e12 #8.9e11 #2.0e12
in_lim = 5.5e11 #2.1e11 #7.556e11
gamma = 1.0001
#d = pw.read_data_for_rt(myfile, orb, level=mylevel, x2_min=3*np.pi/8, x2_max=5.*np.pi/8.,
#                        x1_max=1.1*out_lim,
#                        x3_min=3.5*np.pi/4, x3_max=4.5*np.pi/4,
#                        gamma=gamma)
d = pw.read_data_for_rt(myfile, orb, level=mylevel,
                        x2_min=3.5*np.pi/8, x2_max=4.5*np.pi/8.,
                        x1_min=0.8*in_lim,x1_max=1.1*out_lim,
                        x3_min=np.pi-0.5, x3_max=np.pi+0.5,
                        gamma=gamma,dens_pres_scale_factor=dens_pres_scale)
# d=pw.read_data(myfile,orb,level=mylevel,x1_max=1.1*out_lim,gamma=1.01)

t = d['Time']
rcom, vcom = pw.rcom_vcom(orb, t)
x2, y2, z2 = pw.pos_secondary(orb, t)
print('Time:', t)
print('Position of secondary: ', x2, y2, z2)
print('time to read file:', time.time() - start_read_time)

################################################################

rad_star = 4.67e10 #7.0e+10

#Omega_orb = 1.28241e-05
#omega_planet = Omega_orb
#a = 8.228e+11
rp = 6.72e9
m1 = orb['m1'][0]
m2 = orb['m2'][0]
a = orb['sep'][0]  ## THIS IS ONLY TRUE IN THE CIRCULAR LIMIT! 
Omega_orb = np.sqrt(a**3/(6.674e-8*(m1+m2)))
omega_planet = Omega_orb

print("a = ", a, " Omega_orb = ", Omega_orb)

d2 = np.sqrt((d['x']-x2)**2 + (d['y']-y2)**2 + (d['z']-z2)**2)

dr = np.broadcast_to(d['x1f'][1:]-d['x1f'][0:-1],
                          (len(d['x3v']), len(d['x2v']), len(d['x1v'])))

#################################################################
if(parker):
    # apply the spherically-sym analytic solution
    d['rp'] = d2
    Bern, K, lambda_planet, md = get_BC_prop(d)
    rS = lambda_planet/2. * rp
    vS = np.sqrt(6.674e-8*m2/(lambda_planet*rp))
    rhoS = md/(4.0*np.pi*rS*rS*vS)
    density, velocity = get_Parker_rho_v_func(r_out=np.max(d['rp']), num=1000)
    d = apply_Parker(d)

#################################################################
# Convert rotating -> Inertial frame
d['vx'] = d['vx']-Omega_orb*d['y']
d['vy'] = d['vy']+Omega_orb*d['x']


################################################################
# change the planet density -- e.g. make sure the planet is completely opaque or transparent
rhoplanet=1.0e-24
d['rho']  = np.where(d2 < rp,rhoplanet,d['rho'])
d['press']= np.where(d2 < rp,6.67e-8*m2/5.*rhoplanet,d['press'])


### DELETE THE SW
#d['rho'] = d['rho']*d['r0']
#d['press'] = d['press']*d['r0']
    
#################################################################
start_rt = time.time()

# distance of every point from the star:
#dist_star = np.sqrt((d['x'])**2 + (d['y'])**2 + (d['z'])**2)
#dist_star_au = dist_star/1.496e+13


# photoionization
phi  = phi0_1au/pow(d['gx1v']/1.496e+13, 2)
phi1 = phi1_1au/pow(d['gx1v']/1.496e+13, 2)
phi3 = phi3_1au/pow(d['gx1v']/1.496e+13, 2)

#################################################################
# initial guess

tau_integral0, tau1_integral0, tau3_integral0, ne0, nh10 = initial_guess(floor_val=fv)

###################################################################
# new guess

ne, nh1, nh_plus, nhe1, nhe3, nhe_plus, tau_integral, tau1_integral, tau3_integral = new_guess(
    tau_integral0, tau1_integral0, tau3_integral0, ne0, nh10,floor_val=fv)



####################################################################
# iterations
# number density of metals (average mass of 15.5mp; from Carroll & Ostlie), assuming all metals are neutral
nz = d['rho']*Zuni/(15.5*c.mp)

#min_cell_size = 1.0e12
for _ in range(4):
    print(_)

    tau_integral =  np.clip(np.cumsum(nh1*sigma_photo_nu0*dr,  axis=2),0.0,100.)
    tau1_integral = np.clip(np.cumsum(nhe1*sigma_photo_nu1*dr, axis=2),0.0,100.)
    tau3_integral = np.clip(np.cumsum(nhe3*sigma_photo_nu3*dr, axis=2),0.0,100.)
    #cdhe3 = np.copy(tau3_integral / sigma_nu3)

    
    #d['nhe3'] = np.copy(nhe3)
    mean_mol_weight = np.copy((nh1 + nh_plus + (nhe1 + nhe3 + nhe_plus)
                               * 4.0 + nz*15.5)/(ne+nh1+nh_plus+nhe1+nhe3+nhe_plus+nz))
    d['temp'] = np.copy(mean_mol_weight*c.mp/c.kB * (gamma*d['press']/d['rho']))

    # new H number density

    nh_plus = np.copy(phi*np.exp(-1.0*tau_integral)*d['rho']*Xuni/c.mp / (
        phi*np.exp(-1.0*tau_integral) + ne*alpha*pow(d['temp']/1.0e4, -0.8)))  # ionized hydrogen
    #diff_nh1 = np.copy(abs(d['rho']*Xuni/c.mp - nh_plus - nh1))

    print('(diff nh1)  (med, av):',np.median(abs(d['rho']*Xuni/c.mp - nh_plus - nh1)), np.average(abs(d['rho']*Xuni/c.mp - nh_plus - nh1)) )

    nh1 = np.copy(np.maximum(d['rho']*Xuni/c.mp - nh_plus, fv))  # neutral hydrogen
    nh1_interp = New_get_interp_function(d, nh1)

       
    # new He singlet and triplet densities
    temp_m08 = pow(d['temp']/1.0e4, -0.8)
    q13a_temp = q13a_approx_func(d['temp'])
    q31a_temp = q31a_approx_func(d['temp'])
    q31b_temp = q31b_approx_func(d['temp'])

    f3 = np.copy((ne*alpha1*temp_m08 - (ne*alpha3*temp_m08)*(ne*alpha1*temp_m08 + phi1*np.exp(-1.0*tau1_integral) + ne*q13a_temp)/(ne*alpha3*temp_m08 - ne*q13a_temp)) / (ne*alpha1*temp_m08 - A31 - ne*q31a_temp - ne*q31b_temp -
                                                                                                                                                                          nh1*Q31 - (ne*alpha1*temp_m08 + phi1*np.exp(-1.0*tau1_integral) + ne*q13a_temp)*(ne*alpha3*temp_m08 + A31 + phi3*np.exp(-1.0*tau3_integral) + ne*q31a_temp + ne*q31b_temp + nh1*Q31)/(ne*alpha3*temp_m08 - ne*q13a_temp)))
    f3 = np.maximum(f3,fv)
        
    f1 = np.copy((ne*alpha3*temp_m08 - f3*(ne*alpha3*temp_m08 + A31 + phi3*np.exp(-1.0*tau3_integral) +
                                           ne*q31a_temp + ne*q31b_temp + nh1*Q31)) / (ne*alpha3*temp_m08 - ne*q13a_temp))
    f1 = np.maximum(f1,fv)
    
    nhe1 = np.copy(f1*d['rho']*Yuni/(4.0*c.mp)) 	# ground state helium
    nhe3 = np.copy(f3*d['rho']*Yuni/(4.0*c.mp)) 	# metastable state helium

    
    # (singly) ionized helium
    nhe_plus = np.copy((1.0-f1-f3)*d['rho']*Yuni/(4.0*c.mp))

    ne = np.copy(nh_plus + nhe_plus)  # free electrons

    
    if(plots == True):
        lim = 1.e11 #out_lim
        make_plots(_, int(len(d['x2v'])/2) )
        make_side_plots(_, int(len(d['rho'][:, 0, 0])/2) )




print('time for rt:', time.time() - start_rt)

########################################################################

#nhe3 = np.where(d2<rp,np.nan,nhe3)
nhe3 = np.where(d2<rp,1.e30,nhe3) # Note: must be used with "nearest" interpolation or can affect solution outside of planet


d['nhe3'] = np.copy(nhe3)
mean_mol_weight = np.copy((nh1 + nh_plus + (nhe1 + nhe3 + nhe_plus)
                           * 4.0 + nz*15.5)/(ne+nh1+nh_plus+nhe1+nhe3+nhe_plus+nz))
d['temp'] = np.copy(mean_mol_weight*c.mp/c.kB * (gamma*d['press']/d['rho']))

if(savedata == True):
    jcoord = int(len(d['x2v'])/2)
    dsave = {}
    dsave['dr'] = dr[:,jcoord,:]
    dsave['x'] = d['x'][:,jcoord,:]
    dsave['y'] = d['y'][:,jcoord,:]
    dsave['z'] = d['z'][:,jcoord,:]
    dsave['vx'] = d['vx'][:,jcoord,:]
    dsave['vy'] = d['vy'][:,jcoord,:]
    dsave['vz'] = d['vz'][:,jcoord,:]
    dsave['rho'] = d['rho'][:,jcoord,:]
    dsave['nh1'] = nh1[:,jcoord,:]
    dsave['nh_plus'] = nh_plus[:,jcoord,:]
    dsave['nhe1'] = nhe1[:,jcoord,:]
    dsave['nhe3'] = nhe3[:,jcoord,:]
    dsave['nhe_plus'] = nhe_plus[:,jcoord,:]
    dsave['ne'] = ne[:,jcoord,:]
    dsave['temp'] = d['temp'][:,jcoord,:]
    dsave['press'] = d['press'][:,jcoord,:]
    dsave['mu'] = mean_mol_weight[:,jcoord,:]
    # save hdf5 file with deepdish
    if(parker):
        print("... saving slice output as ",myfile+'.rt_s'+str(np.round(dens_pres_scale,2))+'parker.h5')
        dd.io.save(myfile+'.rt_s'+str(np.round(dens_pres_scale,2))+'parker.h5',dsave)
    else:
        print("... saving slice output as ",myfile+'.rt_s'+str(np.round(dens_pres_scale,2))+'.h5')
        dd.io.save(myfile+'.rt_s'+str(np.round(dens_pres_scale,2))+'.h5',dsave)
        

del d['press'], d['rho']


nhe3_interp = pw.get_interp_function(d, "nhe3")
vx_interp = pw.get_interp_function(d, "vx")
vy_interp = pw.get_interp_function(d, "vy")
vz_interp = pw.get_interp_function(d, "vz")
temp_interp = pw.get_interp_function(d, "temp")





########################################################################

# ray tracing
print("ray tracing for", len(angles), "angles =", angles)

aind = 0
for aa in angles:
    start_angle = time.time()
    print("#####\n angle=", aa, "######")
    aind += 1
    azim_angle = aa + np.pi

    # get random ray positions
    yrandom, zrandom = generate_random(N_mc)
    # FOR TESTING w/N_mc = 10
    #yrandom = np.array([-0.52469469,-0.28468887,-0.63962719,0.42682868,0.29410005,0.91207035,0.10792988,-0.17645031,-0.19454244,0.48970835])
    #zrandom = np.array([0.71603157,-0.0282534,-0.36738675,-0.23474991,-0.81483935,0.36357166,-0.00422779,0.67878842,0.92808775,0.58591747])

    r_prime_mag = np.sqrt(yrandom*yrandom + zrandom*zrandom)

    # calculate stellar intensity profile
    m = np.sqrt(1.-r_prime_mag**2)
    stellar_intensity = I(m, ld1, ld2)
    total_stellar_intensity = np.sum(stellar_intensity)
    # print "total_stellar_intensity=",total_stellar_intensity

    total = np.zeros(len(lamb))
    control = np.zeros(len(lamb))
    control2 = np.zeros(len(lamb))

    control_num = 0.0
    control2_num = 0.0

    for dart in range(N_mc):
        exp_fac = MC_ray(dart)

        if np.isnan(np.sum(exp_fac)):
            control2 += stellar_intensity[dart]
            control2_num += 1
            print('nan dart !!!!! ')
        else:
            total += stellar_intensity[dart]*exp_fac
            control += stellar_intensity[dart]
            control_num += 1
            #print(total[int(len(nu)/2)]/control[int(len(nu)/2)])
            print("... tau=",-np.log(exp_fac[int(len(nu)/2)]))

    final_intensity = total / N_mc
    final_control = control / N_mc
    print("control_num , control2_num, N_mc= ",
          control_num, control2_num, N_mc)

    ###### make plot ######
    plt.axvline(10830.33977, color='black', ls=':')
    plt.axvline(10830.2501, color='black', ls=':')
    plt.axvline(10829.09114, color='black', ls=':')
    plt.plot(lamb, final_intensity/final_control, lw=2)
    plt.plot(lamb, final_control/final_control)

    if(parker):
        pfn = base_dir+snapshot+'_s'+str(np.round(dens_pres_scale,2))+'_l' + \
            str(mylevel)+'_Nmc'+str(N_mc)+'_Nr'+str(nraypoints) + \
            '_spectrum23_a'+str(aa)+'_parker.png'
    else:
        pfn = base_dir+snapshot+'_s'+str(np.round(dens_pres_scale,2))+'_l' + \
            str(mylevel)+'_Nmc'+str(N_mc)+'_Nr' + \
            str(nraypoints)+'_spectrum23_a'+str(aa)+'.png'

    plt.savefig(pfn)
    plt.close()

    #######################

    ### Save file ##########
    if(parker):
        fil = open(base_dir+snapshot+'_s'+str(np.round(dens_pres_scale,2))+'_l'+str(mylevel)+'_Nmc'+str(N_mc) +
                   '_Nr'+str(nraypoints)+'_spectrum23_a'+str(aa)+'_parker.txt', 'w')
    else:
        fil = open(base_dir+snapshot+'_s'+str(np.round(dens_pres_scale,2))+'_l'+str(mylevel)+'_Nmc'+str(N_mc) +
                   '_Nr'+str(nraypoints)+'_spectrum23_a'+str(aa)+'.txt', 'w')
    for i in range(len(lamb)):
        fil.write(str(lamb[i])+'\t' +
                  str(final_intensity[i]/final_intensity[0])+'\n')
    fil.close()
    print('time for one angle:', time.time() - start_angle)

    ########################
