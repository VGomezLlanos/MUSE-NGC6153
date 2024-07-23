import pyneb as pn
import numpy as np

def create_rec_atom(line):
    if line.atom in ('C2r', 'O1r'):
        case = 'A'
    else:
        case = 'B'
    atom = pn.RecAtom(line.elem, line.spec, case=case, extrapolate=True)
    IP = pn.utils.physics.IP[atom.elem][atom.spec-1]
    return atom, IP

def create_col_atom(line):
    atom = pn.Atom(line.elem, line.spec)
    if atom.spec-2 < 0:
        IP = 0.
    else:
        IP = pn.utils.physics.IP[atom.elem][atom.spec-2]
    return atom, IP

def create_atom(line):
    if line.atom[-1] == 'r':
        rec_line = True
        atom, IP = create_rec_atom(line)
    else:
        rec_line = False
        atom, IP = create_col_atom(line)
    return atom, IP, rec_line

def select_TeNe(TeNe, IP, consider_3zones = False, use_ar3 = False):
    if IP < 17:
        Te = TeNe['N2S2']['Te']
        Ne = TeNe['N2S2']['Ne']
    else:
        Te = TeNe['S3Cl3']['Te']
        Ne = TeNe['S3Cl3']['Ne']
    if (consider_3zones) & (IP >= 35):
        if use_ar3:
            Te = TeNe['Ar3Cl3']['Te']
            Ne = TeNe['Ar3Cl3']['Ne']
        else:
            ar_pp = 3e-6
            ar_ppp = 1e-6
            w_ar = ar_pp / (ar_pp + ar_ppp)
            Te = w_ar * (TeNe['Ar3Cl3']['Te']) + (1-w_ar) * (TeNe['Ar4Ar4']['Te'])
            Ne = TeNe['Ar3Cl3']['Ne']
    return Te, Ne

def select_Te_rec(TeNe, Te, Ne, Te_rec):
    if Te_rec == 'He':
        return TeNe['He1']['Te']
    elif Te_rec == 'PJ':
        return TeNe['PJ']['Te']
    elif Te_rec == 'PJ_ANN':
        return TeNe['PJ_ANN']['Te']
    elif Te_rec is None:
        return Te
    else:
        return Te_rec * np.ones_like(Ne)

def set_abunds_ori(TeNe, obs, label = None, tem_HI=None, exclude_elem=('H',),
                   Te_rec = None, abund_dic = None, Ne_cte = None):
    if abund_dic is None:
        abund_dic = {}
    Hbeta = obs.getIntens()['H1r_4861A']

    atom_dic = {}

    for line in obs.getSortedLines():
        if (line.label == label or label is None) & (line.elem not in exclude_elem) & (line.is_valid): 
            if line.atom not in atom_dic:
                atom, IP, rec_line = create_atom(line)
                atom_dic[line.atom] = (atom, IP, rec_line)
            else:
                atom, IP, rec_line = atom_dic[line.atom]

            Te, Ne = select_TeNe(TeNe, IP)
            
            if Ne_cte is not None:
                Ne = Ne_cte * np.ones_like(Te)

            if rec_line:
                Te = select_Te_rec(TeNe, Te, Ne, Te_rec)

            abund_dic[line.label] = atom.getIonAbundance(line.corrIntens/Hbeta, Te, Ne, 
                                                         to_eval=line.to_eval, Hbeta=1., tem_HI=tem_HI)
        elif line.label not in abund_dic.keys():
            abund_dic[line.label] = None
    return abund_dic

def set_abunds(TeNe, obs, w = None, w_1 = None, label = None, tem_HI=None, tem_HI_RLs = None, exclude_elem=('H',),
                   Te_rec = None, abund_dic = None, Ne_rec = None, **kwargs):
    if abund_dic is None:
        abund_dic = {}
    Hbeta = obs.getIntens()['H1r_4861A']

    atom_dic = {}

    for line in obs.getSortedLines():
        if (line.label == label or label is None) & (line.elem not in exclude_elem) & (line.is_valid): 
            if line.atom not in atom_dic:
                atom, IP, rec_line = create_atom(line)
                atom_dic[line.atom] = (atom, IP, rec_line)
            else:
                atom, IP, rec_line = atom_dic[line.atom]

            Te, Ne = select_TeNe(TeNe, IP, **kwargs)
            
            if rec_line:
                tem_HI_adopted = tem_HI_RLs
                if 'He1r' in line.atom:
                    Te = TeNe['He1']['Te']
                    Ne = TeNe['N2S2']['Ne']
                elif 'He2r' in line.atom:
                    pass
                else:
                    Te = select_Te_rec(TeNe, Te, Ne, Te_rec)

                if Ne_rec is not None:
                    Ne = Ne_rec * np.ones_like(Te)

                if w is not None:
                    Hb_w = w
                else:
                    Hb_w = 1
            else:
                tem_HI_adopted = tem_HI
                if w_1 is not None:
                    Hb_w = w_1
                else:
                    Hb_w = 1

            abund_dic[line.label] = atom.getIonAbundance(line.corrIntens/Hbeta, Te, Ne, 
                                                         to_eval=line.to_eval, Hbeta=Hb_w, tem_HI=tem_HI_adopted)
        elif line.label not in abund_dic.keys():
            abund_dic[line.label] = None
    return abund_dic
    