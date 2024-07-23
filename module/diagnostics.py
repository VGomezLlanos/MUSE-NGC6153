import pyneb as pn
import numpy as np
import os
import matplotlib.pyplot as plt
from observation import get_obs
from scipy.interpolate import interp1d
try:
    from ai4neb import manage_RM
    AI4NEB_INSTALLED = True
    pn.config.import_ai4neb()
except:
    AI4NEB_INSTALLED = False
from utils.plots import create_axis, plot_image
from constants.observation_parameters import *

def make_diags(obs, use_ANN):    
    diags = pn.Diagnostics()
    if use_ANN:
        diags.ANN_inst_kwargs = ANN_INST_KWARGS
        diags.ANN_init_kwargs = ANN_INIT_KWARGS
    diags.addDiagsFromObs(obs)
    diags.addDiag('[ArIV] 7170/4740', ('Ar4', 'L(7170)/L(4740)', 'RMS([E(4740), E(7170)])'))
    return diags

def add_gCTD(diags, obs, label, diag1, diag2, TeNe, use_ANN=True, limit_res=True, force=False, save=True, **kwargs):
    """
    Add the values obtained from getCrossTemDen.
    The first time it is run (or if force=True) the ANN is created, trained and stored. 
    Next uses only call the already trained ANN.
    """
    if not AI4NEB_INSTALLED and use_ANN:
        print('ai4neb not installed')
    if force:
        ANN = None        
    else:
        ANN = manage_RM(RM_filename='new_ai4neb/'+label)        
        if not ANN.model_read:
            ANN = None
    Te, Ne = diags.getCrossTemDen(diag1, diag2, obs=obs, use_ANN=use_ANN, ANN=ANN,
                                       limit_res=limit_res, **kwargs)
    if use_ANN and ANN is None and save:
        diags.ANN.save_RM(filename='new_ai4neb/'+label, save_train=True, save_test=True)
    TeNe[label] = {'Te': Te, 'Ne': Ne}
    return TeNe

def plot_tem_den(TeNe, key_diag, vmin_den = VMIN_DEN, vmax_den = VMAX_DEN, vmin_tem = VMIN_TEM, vmax_tem = VMAX_TEM, **kwargs):
    fig, ax = create_axis(2, n_columns=2, **kwargs)
    ax_den = ax[0]
    ax_tem = ax[1]
    plot_image(np.log10(TeNe[key_diag]["Ne"]), fig = fig, ax = ax_den, 
    label = 'Ne {}'.format(DIAGNOSTICS_DICT[key_diag][1]), vmin = vmin_den, vmax = vmax_den)
    plot_image(TeNe[key_diag]["Te"], fig = fig, ax = ax_tem, 
    label = 'Te {}'.format(DIAGNOSTICS_DICT[key_diag][0]), vmin = vmin_tem, vmax = vmax_tem)

def add_T_PJ(TeNe, obs, den, Hep, Hepp):
    cont = pn.Continuum()
    tab_tem = np.linspace(500, 30000, 100)
    tab_den = np.ones_like(tab_tem) * den
    tab_Hep = np.ones_like(tab_tem) * Hep
    tab_Hepp = np.ones_like(tab_tem) * Hepp

    tab_PJ =  cont.BJ_HI(tab_tem, tab_den, tab_Hep, tab_Hepp, wl_bbj = 8100, wl_abj = 8400, HI_label='9_3')
    tem_inter = interp1d(tab_PJ, tab_tem, bounds_error=False)

    TeNe['PJ'] = {}
    C_8100 = obs.getIntens()['H1r_8100A']
    C_8400 = obs.getIntens()['H1r_8400A']
    HI = obs.getIntens()['H1r_9229A']
    with np.errstate(divide='ignore', invalid='ignore'):
        PJ_HI = (C_8100 - C_8400) /  HI
    TeNe['PJ']['Te'] = tem_inter(PJ_HI)
    return TeNe

def add_T_He(TeNe, obs):
    """
    Mendez Delgado 2021
    """
    TeNe['He1'] = {}
    dens = np.asarray((100,   500,  1000,  2000,  3000,  4000,  5000,  6000,  7000,
                       8000,  9000, 10000, 12000, 15000, 20000, 25000, 30000, 40000,
                       45000, 50000))
    alpha = np.asarray((92984, 81830, 77896, 69126, 65040, 62517, 60744, 59402, 58334,
                        57456, 56715, 56077, 55637, 55087, 54364, 53796, 53329, 52591,
                        52289, 52019))
    beta = np.asarray((-7455, -6031, -5527, -4378, -3851, -3529, -3305 ,-3137, -3004, 
                       -2895, -2804, -2726, -2676, -2611, -2523, -2452, -2392, -2297, 
                       -2257, -2222))
    alpha_int = interp1d(dens, alpha, bounds_error=False)
    beta_int = interp1d(dens, beta, bounds_error=False)

    alphas = alpha_int(TeNe['N2S2']['Ne'])
    betas = beta_int(TeNe['N2S2']['Ne'])
    with np.errstate(divide='ignore', invalid='ignore'):
        R_He = obs.getIntens()['He1r_7281A'] / obs.getIntens()['He1r_6678A']

    Te = alphas * R_He + betas
    Te[np.isinf(Te)] = np.nan

    TeNe['He1']['Te'] = Te
    return TeNe

def get_TeNe(obs = None, use_ANN = USE_ANN, plot = PLOT_DIAGNOSTICS):
    if obs is None:
        obs = get_obs()
    diags = make_diags(obs, use_ANN = use_ANN)
    TeNe = {}

    if not os.path.exists(ANN_FOLDER):
        os.mkdir(ANN_FOLDER)
        
    pn.log_.timer('Starting', quiet=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        for key in DIAGNOSTICS_DICT.keys():
            diagnostic = DIAGNOSTICS_DICT[key]
            TeNe = add_gCTD(diags, obs, key, diagnostic[0], diagnostic[1], TeNe, use_ANN = use_ANN)
    pn.log_.timer('ANN getCrossTemDen done') 

    if plot:
        for key in DIAGNOSTICS_DICT.keys():
            plot_tem_den(TeNe, key)

    TeNe = add_T_PJ(TeNe, obs, DEN_PJ, Hep_PJ, Hepp_PJ)
    TeNe = add_T_He(TeNe, obs)

    return TeNe, obs

def plot_PJ(TeNe, obs, **kwargs):
    if "PJ" not in TeNe.keys():
        TeNe = add_T_PJ(TeNe, obs, DEN_PJ, Hep_PJ, Hepp_PJ)
    plot_image(TeNe["PJ"]["Te"], **kwargs)
    return TeNe

def plot_He(TeNe, obs, **kwargs):
    if "He1" not in TeNe.keys():
        TeNe = add_T_He(TeNe, obs)
    plot_image(TeNe["He1"]["Te"], **kwargs)
    return TeNe
