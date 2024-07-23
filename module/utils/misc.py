from pyneb.utils.misc import parseAtom, int_to_roman
import numpy as np

def get_label_str(label, split = False, latex = False):
    atom = label.split('_')[0]
    wl = label.split('_')[1]
    if atom[-1] == 'r':
        atom = '{}'.format(atom[0:-1])
        forb = False
    else:
        forb = True
        
    elem, spec = parseAtom(atom)

    spec_rom = int_to_roman(int(spec))
    
    if forb:
        if latex:
            line = '\\forb{%s}{%s}'%(elem, spec_rom.lower())
        else:
            line = '[{} {}]'.format(elem, spec_rom)
    else:
        if latex:
            line = '\\perm{%s}{%s}'%(elem, spec_rom.lower())
        else:
            line = '{} {}'.format(elem, spec_rom)
        
    if wl[-1] == '+':
        wl = wl[0:-1]
        blend_str = '+'
    else:
        blend_str=''
    if wl[-1] == 'A':
        wl = wl[0:-1]
    wl = wl.split('.')[0]
    if split:
        return elem, spec_rom, wl + blend_str, forb
    else:
        return '{} {}{}'.format(line, wl, blend_str)

def get_label_str_ori(label, split = False):
    lab1 = label.split('_')[0]
    lab3 = label.split('_')[1]
    if lab1[-1] == 'r':
        lab1 = '{}'.format(lab1[0:-1])
        forb = False
    else:
        forb = True
        
    lab1, lab2 = parseAtom(lab1)

    lab2 = int_to_roman(int(lab2))
    
    if forb:
        lab1 = '[{} {}]'.format(lab1, lab2)
    else:
        lab1 = '{} {}'.format(lab1, lab2)
        
    if lab3[-1] == '+':
        lab3 = lab3[0:-1]
        blend_str = '+'
    else:
        blend_str=''
    if lab3[-1] == 'A':
        lab3 = lab3[0:-1]
    lab3 = lab3.split('.')[0]
    if split:
        return lab1, lab3 + blend_str
    else:
        return '{} {}{}'.format(lab1, lab3, blend_str)

def check_recomb(label:str):
    ion = label.split("_")[0]
    return ion[-1] == "r"


def get_image(obs, data=None, label=None, N_MC = None, type_='median', returnObs=True):
    """
    Parameters:
    -----------
        type_ [str]:
            'median': median of the MC and observations.
            'mean': mean of the MC and observations.
            'std': std of the MC and observations.
            
        returnObs [bool]:
            True returns the observations.
            False returns the corrected observations.
    """
    if label is not None:
        if isinstance(label, tuple):
            with np.errstate(divide='ignore', invalid='ignore'):
                to_return = (get_image(label=label[0], type_=type_ ,returnObs=returnObs) / 
                             get_image(label=label[1], type_=type_, returnObs=returnObs))
            return to_return
        d2return = obs.getIntens(returnObs=returnObs)[label]
    else:
        d2return = data # Por qué manda data para regresar data? Para cuando solo se quiere hacer reshape
    if N_MC is None:
        return d2return.reshape(obs.data_shape)
    else:
        if type_ == 'median':
            return np.nanmedian(d2return.reshape(obs.data_shape), 2)
        if type_ == 'mean':
            return np.nanmean(d2return.reshape(obs.data_shape), 2)
        elif type_ == 'std':
            return np.nanstd(d2return.reshape(obs.data_shape), 2)
        elif type_ == 'orig':
            return d2return.reshape(obs.data_shape)[:,:,0]
        else:
            print('type_ must be one of median, mean, std, or orig')