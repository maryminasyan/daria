from .constants import lsun, r_ab, r_PaHa, r_HaPab, lsfr_kennicutt98

# Avoid double-counting OIII
line_names = ('Lya','Ha','Hb','Hg','Hd','He','OIII_5007','OIII_4959','OII',\
              'Pa','PAH','NII','Pab')

line_lpersfr = \
    {
        'Lya': 1.21e42,
        'Ha': lsfr_kennicutt98, # 1.27e41
        'Hb': lsfr_kennicutt98 / r_ab,
        'Hg': 0.468 * 0.44e41,
        'Hd': 0.259 * 0.44e41,
        'He': 0.159 * 0.44e41,
        'OIII': 1.32e41,
        'OIII_5007': 0.75 * 1.32e41,
        'OIII_4959': 0.25 * 1.32e41,
        'OII': 0.71e41,
        # 'Pa': 2.174e40, # (Neufeld+2024 section 3.3) 0.123 * lsfr_kennicutt98
        'Pa': lsfr_kennicutt98 * r_PaHa,
        'PAH': lsun * 10**6.6, # 3.3 um (Lai+ 2020)
        'NII': lsfr_kennicutt98 * 0.25,
        'Pab': lsfr_kennicutt98 / r_HaPab
    }

line_rwaves = \
    {
        'Lya': 1216,
        'Ha': 6563,
        'Hb': 4861,
        'Hg': 4340,
        'Hd': 4102,
        'He': 3970,
        'OII': 3727,
        'OIII_5007': 5007,
        'OIII_4959': 4959,
        'Pa': 1.87e4,
        'PAH': 3.3e4,
        'NII': 6584,
        'Pab': 1.28e4
    }        
        
        
                
