from pathlib import Path
import joblib
import pyneb as pn
import numpy as np

DATA_DIR = "/home/vero/Respaldo/Palta/NGC6153/renamed/maps"
OBJ_NAME = "NGC6153"

OBS_NAME = Path(DATA_DIR) / Path(f'{OBJ_NAME}_MUSE_b_*.fits')

FILE_FORMAT = 'fits_IFU'
CORRECTED = False
ERR_IS_RELATIVE = False
ADD_ERR_DEFAULT = True
CORREC_LAW = 'F99'
ERROR_STR = "error" # etiqueta para identificar el archivo que contiene los errores (alfa-like o 1-sigma)
ERROR_DEFAULT = 0.05 # error por defecto
CUTOUT2D_POSITION = None # posición central del pixel (x_pix, y_pix), en caso de querer cortar la imagen 2D
CUTOUT2D_SIZE = None # tamaño (n_x_pix, n_y_pix) de la imagen a seleccionar

OBS_INT_FILE = "/home/vero/Respaldo/Palta/NGC6153/NGC6153_int_line_fluxes.dat"
FILE_FORMAT_INT = "lines_in_rows_err_cols"

FLUX_NORM = 1e-20 # Normalización del flujo
CLEAN_ERROR = 1e-5

N_MC = None
N_MC_INT = 500

TEM_CHB = 8000 # Based on Te[S III]
DEN_CHB = 3400 # Based on Ne[Cl III]
R_V = 3.1
EBV_MIN = 0. 
EXTINCTION_LABEL = ("H1r_6563A", "H1r_9229A", "H1r_8750A", 'H1r_8863A', 'H1r_9015A')

CUT_HB = 0.005

TE_CORR = 6000
DEN_CORR = 1e4
REC_LABEL = 'O2r_4649.13A' #rec_label may be 'O2r_4661.63A' or 'O2r_4649.13A' al utilizar 4649.13 está considerando 4649.13 + 4650.84 del mult V1 de OII

TENE_HIGH = 'S3Cl3'

RANDOM_SEED = 42
ANN_INST_KWARGS = {'RM_type' : 'SK_ANN', 
                                'verbose' : False, 
                                'scaling' : True,
                                'use_log' : True, # In pipieline.py was set as False, but it didn't work
                                'random_seed' : RANDOM_SEED
                                }

ANN_INIT_KWARGS = {'solver' : 'lbfgs', 
                                'activation' : 'tanh', 
                                'hidden_layer_sizes' : (10, 20, 10), 
                                'max_iter' : 20000
                                }

USE_ANN = True
ANN_FOLDER = "./ai4neb/"

DIAGNOSTICS_DICT = {'N2S2': ('[NII] 5755/6548', '[SII] 6731/6716'),
                    'S3Cl3': ('[SIII] 6312/9069', '[ClIII] 5538/5518'),
                    'S3S2': ('[SIII] 6312/9069', '[SII] 6731/6716'),
                    'S3Ar4': ('[SIII] 6312/9069', '[ArIV] 4740/4711'),
                    'Ar3Cl3': ('[ArIII] 5192/7136','[ClIII] 5538/5518'),
                    'Ar3S2': ('[ArIII] 5192/7136', '[SII] 6731/6716'),
                    'Ar4Cl3': ('[ArIV] 7170/4740','[ClIII] 5538/5518'),
                    'Ar4S2': ('[ArIV] 7170/4740', '[SII] 6731/6716'),
                    'Ar4Ar4': ('[ArIV] 7170/4740','[ArIV] 4740/4711')
                    }

PLOT_DIAGNOSTICS = True
VMIN_DEN = 3
VMAX_DEN = 4
VMIN_TEM = 7500
VMAX_TEM = 12000

DEN_PJ = 1e4 
Hep_PJ = 0.12
Hepp_PJ = 0.004

FITS_DIR = "/home/vgomez/Datos/MUSE/NGC6153/maps/gauss_fit/"
C_KMS = 299792.46 #km/s taken from astropy
V_SYS = 35 #Richer et al. (2022)
VMIN = -35
VMAX = 35
FLUX_LIM = -18

FITS_INFO = ( #("NGC6153_MUSE_[6301.0].fits", 6300.304, "O1_6300A", "[O I] 6300", 0),
              ("NGC6153_MUSE_[6364.0].fits", 6363.777, "O1_6364A", "[O I] 6364", 0),
              #("NGC6153_MUSE_[5192.0,5198.0,5200.0].fits", 5197.9016, "N1_5198A", "[N I] 5198", 4),
              ("NGC6153_MUSE_[5192.0,5198.0,5200.0].fits", 5200.2574, "N1_5200A", "[N I] 5200", 8),
              ("NGC6153_MUSE_[6717.0,6731.0].fits", 6716.44, "S2_6716A", "[S II] 6716", 0),
              #("NGC6153_MUSE_[6717.0,6731.0].fits", 6730.81, "S2_6731A", "[S II] 6731", 4),              
              ("NGC6153_MUSE_[6548.0].fits", 6548.04, "N2_6548A", "[N II] 6548", 0),
              ("NGC6153_MUSE_[5518.0,5538.0].fits", 5517.72, "Cl3_5518A", "[Cl III] 5518", 0),
              #("NGC6153_MUSE_[5518.0,5538.0].fits", 5537.89, "Cl3_5538A", "[Cl III] 5538", 4),
              #("NGC6153_MUSE_[5667,5676,5680,5686.0].fits", 5666.63, "N2r_5666.63A", "N II 5666.63", 0),
              #("NGC6153_MUSE_[5667,5676,5680,5686.0].fits", 5676.02, "N2r_5676.02A", "N II 5676.02", 4),
              ("NGC6153_MUSE_[5667,5676,5680,5686.0].fits", 5679.56, "N2r_5679.56A", "N II 5679.56", 8),
              #("NGC6153_MUSE_[5667,5676,5680,5686.0].fits", 5686.21, "N2r_5686.21A", "N II 5686.21", 12), 
              ("NGC6153_MUSE_[4659.0,4663.0].fits", 4661.63, "O2r_4661.63A", "O II 4661", 4),
              ("NGC6153_MUSE_[4959.0].fits", 4958.911, "O3_4959A", "[O III] 4959", 0),
              #("NGC6153_MUSE_[4634.0,4640.0,4649.0].fits", 4640.8, "N3r_4641A", "N III 4641", 4)
                            
              #("NGC6153_MUSE_[9069.0].fits", 9069.2, "S3_9069A", "[S III] 9069", 0),
              

              #("NGC6153_MUSE_[4634.0,4640.0,4649.0].fits", 4649.3, "O2r_4649.13A", "O II 4649+", 8),

              )

IP_CUT = 17

wcs_file = 'constants/wcs_NGC6153.joblib'
WCS = joblib.load(wcs_file)
