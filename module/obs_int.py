import pyneb as pn
import numpy as np
from utils.plots import plot_image
from constants.observation_parameters import *

class Obs_int(object):
    def __init__(self):
        self.obs_int = None
        self.isNormalized_obs_int = False
        self.areLinesRedefined = False
        self.isExtinctionCorrected = False
        self.isNIICorrected = False
        self.isOIICorrected = False
        self.N_MC = N_MC_INT
    
    def read_obs_int(self):
        if self.obs_int is None:
            self.obs_int = pn.Observation(OBS_INT_FILE, 
                            fileFormat = FILE_FORMAT_INT, 
                            corrected = CORRECTED, 
                            errIsRelative=ERR_IS_RELATIVE, 
                            err_default = ERROR_DEFAULT, 
                            addErrDefault = ADD_ERR_DEFAULT) #ADD_ERR_DEFAULT

    def norm_obs_int(self):
        if (self.obs_int is not None) & (not self.isNormalized_obs_int):
            for line in self.obs_int.getSortedLines():
                line.obsIntens *= FLUX_NORM
            self.isNormalized_obs_int = True
        else:
            self.read_obs_int()
            self.norm_obs_int()

    def redefine_lines(self):
        if (self.obs_int is not None) & (not self.areLinesRedefined):
            # Se agrega la suma de estas 3 líneas y se le pone la etiqueta correspondiente
            self.obs_int.addSum(('O1r_7771A', 'O1r_7773A', 'O1r_7775A'), 'O1r_7773+')
            # Se eliminan las líneas individuales
            self.obs_int.removeLine('O1r_7771A')
            self.obs_int.removeLine('O1r_7773A')
            self.obs_int.removeLine('O1r_7775A')
            # Se define 4649.13 como la suma de esas dos líneas
            self.obs_int.getLine(label='O2r_4649.13A').to_eval = 'L(4649.13) + L(4650.84)'
            # Se define 4726+ como la suma de las transiciones 4-3 y 5-3 de Ne4
            self.obs_int.getLine(label='Ne4_4726A+').to_eval = 'I(4,3)+I(5,3)'
            self.areLinesRedefined = True
        else:
            print('Redefine lines not done. Observations must be read first')

    def add_MC(self):
        if (self.obs_int is not None) & (self.N_MC is not None):
            print(f"Adding N = {self.N_MC} Monte Carlo")
            self.obs_int.addMonteCarloObs(self.N_MC, random_seed = RANDOM_SEED)
        else:
            print('MC not added. Observations must be read first or N_MC is None')

    def red_cor_obs(self, EBV_min=None, r_theo = 2.86,  
                    label1="H1r_6563A", label2="H1r_4861A"):
        if (self.obs_int is not None) & (not self.isExtinctionCorrected):
            self.obs_int.extinction.R_V = R_V
            print('Redifined R_V = '+str(self.obs_int.extinction.R_V))
            try:
                _ = r_theo[0]
                # Esto debe ser para el caso en que se utilicen varios cocientes de Balmer para calcular la extinción como en el caso de M1-42
                # donde se utilizan estas lineas: "H1r_6563A", "H1r_9229A", "H1r_8750A", 'H1r_8863A', 'H1r_9015A'
                r_theo_is_iterable = True
            except:
                r_theo_is_iterable = False

            if r_theo_is_iterable:
                self.EBV = []
                for l1, r in zip(label1, r_theo):
                    self.obs_int.def_EBV(label1=l1, label2=label2, r_theo=r)
                    self.EBV.append(self.obs_int.extinction.E_BV)
                self.obs_int.extinction.E_BV = np.nanmedian(self.EBV, 0)       
            else:
                self.obs_int.def_EBV(label1=label1, label2=label2, r_theo=r_theo)

            if EBV_min is not None:
                mask = self.obs_int.extinction.E_BV < EBV_min
                # Esto solo va a funcionar cuando se tenga la clase definida, por el momento se deja como comentario
                """pn.log_.message('number of spaxels with EBV < {} : {}/{}'.format(EBV_min, mask.sum(),len(mask)),
                                calling='PipeLine.red_cor_obs')"""
                self.obs_int.extinction.E_BV[mask] = 0.

            self.obs_int.correctData()
    
    def correct_by_cHb(self, tem, den):
        if (self.obs_int is not None) & (not self.isExtinctionCorrected):
            HI = pn.RecAtom('H',1)
            get_r_theo = lambda label:HI.getEmissivity(tem, den, label=label, product=False)  / (
                                    HI.getEmissivity(tem, den, label='4_2', product=False) )
            R_THEO = [get_r_theo('3_2'), get_r_theo('9_3'), get_r_theo('12_3'), get_r_theo('11_3'), get_r_theo('10_3')]

            self.red_cor_obs(EBV_min = EBV_MIN,  
                            label1=EXTINCTION_LABEL,
                            r_theo=R_THEO)
            self.isExtinctionCorrected = True
        else:
            print('Correction not done, observations must be read or correction was already done.')

    def correct_NII_recomb(self, tem_rec, den_rec): 
        if (self.obs_int is not None) & (not self.isNIICorrected):
            I_5755 = self.obs_int.getIntens()['N2_5755A']
            I_5679 = self.obs_int.getIntens()['N2r_5679.56A']        
            pn.atomicData.setDataFile('n_ii_rec_P91.func')
            N2rP = pn.RecAtom('N', 2, case='B')
            pn.atomicData.setDataFile('n_ii_rec_FSL11.func')
            N2rF = pn.RecAtom('N', 2, case='B')
            R_5755_5679 = (N2rP.getEmissivity(tem_rec, den_rec, label='5755.', product=False) / 
                        N2rF.getEmissivity(tem_rec, den_rec, label='5679.56', product=False))
            with np.errstate(divide='ignore', invalid='ignore'):
                I_5755R = R_5755_5679 * I_5679
                I_5755_new = I_5755 - I_5755R
            for line in self.obs_int.lines:
                if line.label == 'N2_5755A':
                    line.corrIntens = I_5755_new
            self.isNIICorrected = True
        else:
            print('NII recombination correction not done, observations must be read or correction was already done.')

    def correct_OII_recomb(self, tem_rec, den_rec, rec_label):
        if (self.obs_int is not None) & (not self.isOIICorrected):
            I_7320 = self.obs_int.getIntens()['O2_7319A+']
            I_7330 = self.obs_int.getIntens()['O2_7330A+']
            I_7325 = I_7320 + I_7330
            
            I_REC = self.obs_int.getIntens()[rec_label]
            
            pn.atomicData.setDataFile('o_ii_rec_P91.func')
            O2rP = pn.RecAtom('O', 2, case='B') 
            pn.atomicData.setDataFile('o_ii_rec_SSB17-B-opt.hdf5')
            O2rS = pn.RecAtom('O', 2, case='B')
            wave_str = rec_label.split('_')[1][:-1]
            emisP = O2rP.getEmissivity(tem_rec, den_rec, label='7325+', product=False)
            if rec_label == 'O2r_4649.13A':
                emisR = O2rS.getEmissivity(tem_rec, den_rec, label='4649.13', product=False) \
                    + O2rS.getEmissivity(tem_rec, den_rec, label='4650.84', product=False)
            else:
                emisR = O2rS.getEmissivity(tem_rec, den_rec, label=wave_str, product=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                R_7325_REC = emisP / emisR
                I_7325_new = I_7325 - R_7325_REC * I_REC

            for line in self.obs_int.lines:
                with np.errstate(divide='ignore', invalid='ignore'):
                    if line.label == 'O2_7319A+':
                        line.corrIntens = I_7325_new * I_7320 / I_7325
                    if line.label == 'O2_7330A+':
                        line.corrIntens = I_7325_new * I_7330 / I_7325
            self.isOIICorrected = True
        else:
            print('OII recombination correction not done, observations must be read or correction was already done.')

    def get_obs(self, tem_cHb = TEM_CHB, den_cHb = DEN_CHB, tem_rec = TE_CORR, den_rec = DEN_CORR, oii_rec_label = REC_LABEL,
                corr_NII = True, corr_OII = True):
        self.__init__()
        self.read_obs_int()
        self.norm_obs_int()
        self.redefine_lines()
        self.add_MC()
        self.correct_by_cHb(tem = tem_cHb, den = den_cHb)
        if corr_NII:
            self.correct_NII_recomb(tem_rec = tem_rec, den_rec = den_rec)
        if corr_OII:
            self.correct_OII_recomb(tem_rec = tem_rec, den_rec = den_rec, rec_label = oii_rec_label)


def get_obs_int(**kwargs):
    obj = Obs_int()
    obj.get_obs(**kwargs)
    return obj.obs_int
