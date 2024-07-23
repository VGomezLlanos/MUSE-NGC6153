import numpy as np
import matplotlib.pyplot as plt
from constants.observation_parameters import WCS, N_MC
from utils.misc import check_recomb, get_image, get_label_str

def create_axis(n_line_maps, suptitle:str = "", n_columns = 3, scale_x = 5, scale_y = 4, 
                size = 30, top = 0.95, bottom = 0.05, hspace = 0.3, wspace = 0.2, show_ticks=False):
    n_rows = n_line_maps // n_columns
    residue = n_line_maps % n_columns
    if residue > 0:
        n_rows += 1
    figsize = (int(scale_x*n_columns), int(scale_y*n_rows))
    if show_ticks:
        fig, axs = plt.subplots(n_rows, n_columns, figsize = figsize, subplot_kw={'projection': WCS})
    else:
        fig, axs = plt.subplots(n_rows, n_columns, figsize = figsize)
    fig.suptitle(suptitle, size = size)
    fig.subplots_adjust(top = top, bottom = bottom, hspace = hspace, wspace = wspace)
    return fig, axs

def plot_image(image:np.ndarray, label:str="", fig = None, ax = None, show_ticks = False,
               create_colorbar = True, cmap = 'viridis', title_size = 12, **kwargs):
    if np.ndim(image) == 1:
        if np.size(image) == 40000:
            image_plot = image.reshape(200,200)
        else:
            image_plot = image.reshape(200,200,N_MC+1)[:,:,0]
    else:
        image_plot = image
    if ax is None:
        fig, ax = create_axis(1, n_columns=1)
    
    im = ax.imshow(image_plot, cmap = cmap, origin = "lower", **kwargs)
    if show_ticks:
        ax.grid(color='gray', linestyle='--', linewidth=1)
    else:
        ax.grid(color='gray', linestyle='--', linewidth=1, alpha =0.5)
        ax.set_xticks([18, 72, 126, 180])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_yticks([25, 75, 125, 175])
        ax.set_yticklabels(['', '', '', ''])
    if create_colorbar:
        cax = fig.add_axes([ax.get_position().x1,
                    ax.get_position().y0,
                    0.01,
                    ax.get_position().height])
    ax.set_title(label, size = title_size)
    if show_ticks:
        ax.set_xlabel("Right Ascension")
        ax.set_ylabel("Declination")
    if create_colorbar:
        fig.colorbar(im, cax = cax)

def plot_ionic_ab(abund_dic, abund_keys:list=None, dex_range = 0.5):
    if abund_keys is None:
        n_maps = len(abund_dic)
        lines = list(abund_dic.keys())
    else:
        n_maps = len(abund_keys)
        lines = abund_keys
    fig, axs = create_axis(n_maps)
    for index, line in enumerate(lines):
        try:
            ab = abund_dic[line]
            log_median = np.log10(np.nanmedian(ab))
            vmin = log_median - dex_range
            vmax = log_median + dex_range
            plot_image(np.log10(ab), vmin = vmin, vmax = vmax, label = line, fig = fig, ax = axs.ravel()[index])
        except:
            print(line)


def plot_fluxes(obs, returnObs=True, **kwargs):
    lines_labels = [obs.getSortedLines()[index].label for index in range(len(obs.getSortedLines()))]
    n_recomb = sum([check_recomb(label) for label in lines_labels])
    n_coll = len(lines_labels) - n_recomb
    
    fig_r, axs_r = create_axis(n_recomb, suptitle = "Recombination lines")
    index_r = 0

    fig_c, axs_c = create_axis(n_coll, suptitle = "Collisionaly excited lines")
    index_c = 0

    for label in lines_labels:
        if check_recomb(label):
            image = get_image(obs = obs, label=label, type_='orig', returnObs=returnObs)
            plot_image(np.log10(image), get_label_str(label), fig = fig_r, ax = axs_r.ravel()[index_r], **kwargs)
            index_r += 1
        else:
            image = get_image(obs = obs, label=label, type_='orig', returnObs=returnObs)
            plot_image(np.log10(image), get_label_str(label), fig = fig_c, ax = axs_c.ravel()[index_c], **kwargs)
            index_c += 1

def plot_ann_test(pred, ann, tem_diag:str = 'OII 4649/4089', den_diag:str = 'O II 4649/mult V1'):
    fig, ax = plt.subplots(1,2, figsize = (17,7))
    fontsize = 14
    lim_min = (np.min(pred[:,0]))*1e4 - 100
    lim_max = (np.max(pred[:,0]))*1e4 + 100
    cb0 = ax[0].scatter(ann.y_test[:,0]*1e4, pred[:,0]*1e4, c = ann.y_test[:,1], cmap = 'jet')
    cbar1 = fig.colorbar(cb0, ax = ax[0])
    cbar1.ax.tick_params(labelsize=fontsize)
    cbar1.set_label(label='log (Ne) [cm^-3]', size=fontsize)
    ax[0].set_title('Temperature diagnostic {}'.format(tem_diag), size = fontsize, weight='bold')
    ax[0].tick_params(axis='y', labelsize=fontsize)
    ax[0].tick_params(axis='x', labelsize=fontsize, rotation= 45)
    ax[0].plot((lim_min, lim_max),(lim_min, lim_max), color ='k')
    ax[0].set_xlim(lim_min, lim_max)
    ax[0].set_ylim(lim_min, lim_max)
    ax[0].set_xlabel('Te [K] (Real)', size = fontsize)
    ax[0].set_ylabel('Te [K] (Predicted)', size = fontsize)

    lim_min = np.min(pred[:,1])-0.1
    lim_max = np.max(pred[:,1])+0.1
    cb1 = ax[1].scatter(ann.y_test[:,1], pred[:,1], c = 1e4 * ann.y_test[:,0], cmap = 'jet')
    cbar2 = fig.colorbar(cb1, ax = ax[1], label = 'Te [K]')
    cbar2.ax.tick_params(labelsize=fontsize)
    cbar2.set_label(label='Te [K]', size=fontsize)
    ax[1].tick_params(axis='both', labelsize=fontsize)
    ax[1].set_title('Density diagnostic {}'.format(den_diag), size = fontsize, weight='bold')
    ax[1].plot((lim_min, lim_max),(lim_min, lim_max), color ='k')
    ax[1].set_xlim(lim_min, lim_max)
    ax[1].set_ylim(lim_min, lim_max)
    ax[1].set_xlabel('log(Ne [cm^-3]) (Real)', size = fontsize)
    ax[1].set_ylabel('log(Ne [cm^-3]) (Predicted)', size = fontsize);