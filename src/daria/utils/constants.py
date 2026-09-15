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
# Osterbrock + Ferland table 4.4 T 10kK ne 1e2
r_PagHb = 0.0904
r_BraHb = 0.0802
r_BrbHb = 0.0455
r_BrgHb = 0.0278

convert_L_to_planck = (0.7/0.6766)**2
# Ha L-SFR calibrations
lsfr_kennicutt98 = 1.27e41 # Kennicutt 1998
lsfr_murphy11 = 1.862e41 # Murphy et al 2011 / Kennicutt and Evans 2012
