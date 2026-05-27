import astropy.constants as const
import astropy.units as u

c = float(const.c.to(u.cm/u.s).value)
cm_per_mpc = u.Mpc.to(u.cm)
m_per_mpc = u.Mpc.to(u.m)
erg_per_s_per_nW = u.nW.to(u.erg/u.s)
sqdeg_per_std = u.sr.to(u.deg**2)

lsun = float(const.L_sun.to(u.erg/u.s).value)
r_ab = 2.86 # Balmer decrement in absence of dust
r_HaPab = 17.6
r_PaHb = 0.33
r_PaHa = r_PaHb / r_ab

# Ha L-SFR calibrations
lsfr_kennicutt98 = 1.27e41 # Kennicutt 1998
lsfr_murphy11 = 1.862e41 # Murphy et al 2011 / Kennicutt and Evans 2012
