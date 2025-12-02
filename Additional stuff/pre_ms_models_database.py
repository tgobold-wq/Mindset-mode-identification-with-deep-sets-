import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.spatial
import sys
import os
import boundaries as bd
from matplotlib.lines import Line2D
from shapely.geometry import Point, Polygon
import matplotlib.colors as cl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes # we need this to place the colorbar
import corner
import pickle
import warnings
warnings.filterwarnings("ignore", module = "matplotlib\\..*")
active_ridges = []


# model_data = np.loadtxt('pre-ms-models_database/results_10_dim_model_extension_for_database.txt')
# model_data = pd.DataFrame(model_data,
#                           columns=['model_id', 'mass', 'Teff', 'log_g', 'radius', 'L', 'age', 'alphas', 'zs',
#                                    'e_fold',
#                                    'betas', 'mlins', 'mdots', 'Dmixs', 'f1s', 'hes', 'l_val', 'F', 'O1', 'O2',
#                                    'O3',
#                                    'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'F_drot', 'O1_drot', 'O2_drot',
#                                    'O3_drot',
#                                    'O4_drot', 'O5_drot', 'O6_drot', 'O7_drot', 'O8_drot', 'O9_drot', "parent_model_numbers"])

# print(model_data['O1'])
# print(model_data)


def get_marker( lval):
    if lval == 0:
        return 'ro'
    elif lval == 1:
        return 'g^'
    else:
        return 'bP'


# def markerblack( lval):
#     if lval == 0:
#         return 'ko'
#     elif lval == 1:
#         return 'k^'
#     else:
#         return 'kP'


def plot_echelle(ax, ridges, observed_fs):
    dnu = 0
    keys = ['F', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9']
    for r in ridges:
        dnu += r.delta_nu
    dnu = dnu / len(ridges)
    dnuplus = 1.2 * dnu
    for r, k in zip(ridges, keys):
        marker = get_marker(int(r.l))
        ax.plot(r.fvals % dnu, r.fvals, marker, ms=10)
        ax.plot(r.fvals % dnu + dnu, r.fvals, marker, ms=10)

    ax.plot(observed_fs % dnu, observed_fs, "kx", ms=10)
    ax.axvline(dnu, color='black', ls='--', lw=2)
    # pemmpt = Line2D([0], [0], color="w")

    ax.text(0.99, 0.95, r"$(\nu \mathrm{mod} \Delta\nu) + \Delta\nu$", 
                ha='right', va='center', transform=ax.transAxes)

    ax.text(0.8, 0.95, r"$(\nu \mathrm{mod} \Delta\nu) + \Delta\nu$", 
                ha='right', va='center', transform=ax.transAxes, fontsize='xx-small')


    ax.set_ylabel('Frequency $(d^{-1})$')
    ax.set_xlabel('Frequency modulo ' + '{:.3f}'.format(dnu) + '$ (d^{-1})$')
    ax.set_xlim(0, dnuplus)

class modelling_ridge():
    def __init__(self, f_dict, l):
        self.data = f_dict
        self.l = l
        self.delta_nu_vals = []
        self.fvals = []
        for key in ['F', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7']:
            if f_dict[key] != None:
                self.fvals.append(f_dict[key])
        self.fvals = np.array(self.fvals)

        self.delta_nu_vals = self.fvals[1:] - self.fvals[:-1]
        self.delta_nu = np.median(self.delta_nu_vals)

class observed_star():
    def __init__(self, log_Teff, log_Teff_err, log_g, log_g_err, frequencies, name, rl):
        self.logTeff = log_Teff
        self.Teff = 10**log_Teff
        self.errlogTeff = log_Teff_err
        self.errTeff = 10**(log_Teff+log_Teff_err)-10**log_Teff
        # print(self.Teff, self.errTeff)
        self.logg = log_g
        self.errlogg = log_g_err
        self.frequencies = np.array(frequencies)
        self.name = name
        self.rl = rl



def closest_point_distance(ckdtree, x, y):
    #returns distance to closest point
    return ckdtree.query([x, y])[0]

def closest_point_id(ckdtree, x, y):
    #returns index of closest point
    return ckdtree.query([x, y])[1]

def closest_point_coords(ckdtree, x, y):
    # returns coordinates of closest point
    return ckdtree.data[closest_point_id(ckdtree, x, y)]

def parameter_overview_plot(model_data, observed_star = None):

    mosaic = """
                AAAAAABBCCDD
                AAAAAABBCCDD
                AAAAAAEEFFGG
                AAAAAAEEFFGG
                AAAAAAHHIIJJ
                AAAAAAHHIIJJ
                KKLLMMNNOOPP
                KKLLMMNNOOPP
                QQQQQQRRRRRR
                """

    fig = plt.figure(constrained_layout=True, figsize=(10,6), dpi=150)
    ax = fig.subplot_mosaic(mosaic)

    # ax["A"].plot(model_data2['Teff'], model_data2['log_g'], ls='', marker='x', color='palegreen', ms=2, zorder=0,
    #              rasterized=True)

    ax["A"].plot(model_data['Teff'], model_data['log_g'], 'kx', ms=2, zorder=0, rasterized=True)
    observed_fs = np.array([])
    if observed_star != None:
        observed_fs = observed_star.frequencies
        errbox_Teff = [observed_star.Teff - observed_star.errTeff, observed_star.Teff + observed_star.errTeff, observed_star.Teff + observed_star.errTeff, observed_star.Teff - observed_star.errTeff, observed_star.Teff - observed_star.errTeff]
        errbox_logg = [observed_star.logg - observed_star.errlogg, observed_star.logg - observed_star.errlogg, observed_star.logg + observed_star.errlogg, observed_star.logg + observed_star.errlogg, observed_star.logg - observed_star.errlogg]
        # print(errbox_Teff, errbox_logg)
        ax["A"].plot(errbox_Teff, errbox_logg, color='red', marker='', lw=2)


    x_scale = np.max(model_data['Teff'])-np.min(model_data['Teff'])
    y_scale = np.max(model_data['log_g'])-np.min(model_data['log_g'])
    ckdtree = scipy.spatial.cKDTree(np.array([model_data['Teff']/x_scale, model_data['log_g']/y_scale]).T)
    ax["A"].invert_xaxis()
    ax["A"].invert_yaxis()
    # for i in [601, 801, 201, 11, 802, 9]:
    #     hist = np.genfromtxt(f'../astrocluster/dscuti/grid/model_{i}/LOGS/history.data', skip_header=5, names=True)
    #     ax["A"].plot(10 ** hist['log_Teff'], hist['log_g'], 'm-', lw=2, zorder=1)
    # ax["A"].set_ylim(4.55, 3.25)
    # ax["A"].set_xlim(10500, 6100)
    ax["A"].set_ylabel('$\\log \\,g$')
    ax["A"].set_xlabel(r'$T_{eff}$')

    for a, key, name in zip(["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
                            ['mass', 'Teff', 'log_g', 'radius', 'L', 'age', 'alphas', 'zs', 'hes', 'e_fold', 'betas',
                             'mlins', 'mdots', 'Dmixs', 'f1s'], [r'star mass ($M_\odot$)', r'$T_{eff}$',
                                                                 r'$log \,g$', r'radius ($R_\odot)$', r'$L \,(L_\odot)$',
                                                                 'star age (Myr)', r'$\alpha_{MLT}$', 'Z', 'Y',
                                                                 r'$\tau (Myr)$', r'$\beta$',
                                                                 r'$M_{outer}$', r'$\dot{M}_0 $',
                                                                 r'$D_{mix}$ ($m\,s^{-1}$)', '$F$']):
        if key == 'age':
            log = True
        else:
            log = False
        if key == 'age':
            factor = 1e-6
        else:
            factor = 1
        if key == 'zs':
            ax[a].hist(model_data[key] * factor,  color='gray',
                       edgecolor='black', log=log, bins=10)
        else:
            ax[a].hist(model_data[key] * factor, color='gray',
                       edgecolor='black', log=log)
        if key == 'alphas':
            ax[a].set_xticks([1.8, 2.0, 2.2, 2.4])
        if key == 'hes':
            ax[a].set_xticks([0.204, 0.224])

        ax[a].set_xlabel(name)

    ax["Q"].axis("off")
    ax["R"].axis("off")
    for a in ["Q", "R"]:
        ax[a].set_xlim(-1,1)
        ax[a].set_ylim(-1,1)
        ax[a].fill_between([-0.95, 0.95], y1= [0.95,0.95], y2= [-0.95,-0.95], ec = "k", fc = "grey")
    ax["Q"].text( 0, 0, "Run Automatic Echelle Diagram", va = "center", ha = "center")
    ax["R"].text( 0, 0, "Run Asteroseismic Modelling", va = "center", ha = "center")

        # ax[a].set_ylabel('# of models')
    plt.tight_layout()
    plt.subplots_adjust(wspace=1.0, hspace=1.0)
    # plt.show()


    def onclick(event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))
        if event.dblclick:
            # print(event.x, event.y, event.xdata, event.ydata, closest_point_id(ckdtree, event.xdata/x_scale, event.ydata/y_scale), closest_point_coords(ckdtree, event.xdata/x_scale, event.ydata/y_scale), closest_point_distance(ckdtree, event.xdata/x_scale, event.ydata/y_scale))

            # print(model_data["Teff"].iloc[closest_point], model_id)
            if event.inaxes.axes==ax["A"].axes:
                closest_point = closest_point_id(ckdtree, event.xdata / x_scale, event.ydata / y_scale)
                model_id = int(model_data["parent_model_numbers"].iloc[closest_point])
                echell_fig, [track_ax, echelle_ax] = plt.subplots(1, 2, figsize=(8, 6))
                if model_id > 1500:
                    folder = "10_dim_grid_extension_tracks"
                elif model_id > 500:
                    folder = "10_dim_grid_tracks"
                else:
                    compare = [model_data["e_fold"].iloc[closest_point], model_data["betas"].iloc[closest_point], model_data["mlins"].iloc[closest_point], model_data["mdots"].iloc[closest_point]]
                    if compare == [0.5, 0.1, 0.1, 5e-06]:
                        folder = "6_dim_grid_tracks"
                    else:
                        folder = "10_dim_grid_tracks"

                evol_track_data = np.loadtxt(f"pre-ms-models_database/{folder}/model_{model_id}.txt")
                # print(evol_track_data[0])
                track_ax.plot(10 ** evol_track_data[0], evol_track_data[1], 'k-')
                track_ax.plot(model_data["Teff"].iloc[closest_point], model_data["log_g"].iloc[closest_point], 'rx',
                              ms=10)

                if observed_star != None:
                    track_ax.plot(errbox_Teff, errbox_logg, color='red', marker='', lw=2)
                track_ax.set_ylabel('$\\log \\,g$')
                track_ax.set_xlabel('r$T_{eff}$')

                track_ax.invert_xaxis()
                track_ax.invert_yaxis()
                echelle_df = model_data[model_data["model_id"] == model_data["model_id"].iloc[closest_point]]
                r1 = echelle_df[echelle_df["l_val"] == 0]
                r2 = echelle_df[echelle_df["l_val"] == 1]
                r3 = echelle_df[echelle_df["l_val"] == 2]
                # print(r1)
                r1 = r1[['F', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7']]
                r1 = r1.to_dict('records')[0]
                r2 = r2[['F', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7']]
                r2 = r2.to_dict('records')[0]
                r3 = r3[['F', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7']]

                # print(r1, r2, r3)
                r3 = r3.to_dict('records')[0]
                r1 = modelling_ridge(r1, 0)
                r2 = modelling_ridge(r2, 1)
                r3 = modelling_ridge(r3, 2)

                plot_echelle(echelle_ax, [r1, r2, r3], observed_fs)
                plt.show()
            elif event.inaxes.axes==ax["Q"].axes:
                print("Run Automatic Echelle Diagram")
                if observed_star != None:
                    aed = automatic_echelle_diagramm(model_data, observed_star.frequencies, np.ones(len(observed_star.frequencies)),
                                                     observed_star.name, observed_star.rl, observed_star.Teff,
                                                     observed_star.errTeff, observed_star.logg, observed_star.errlogg)

            elif event.inaxes.axes==ax["R"].axes:
                # print("Run Asteroseismic modelling")
                # print(active_ridges)
                print("Run Asteroseismic Modelling with the ridges:")
                if observed_star != None:
                    for r in active_ridges:
                        r.print()
                    r1, k1 = active_ridges[0].to_modelling_ridge()
                    r2, k2 = active_ridges[1].to_modelling_ridge()
                    # print(r1, r2)
                    modelling(model_data, [r1, r2],
                              [k1, k2], Teff=observed_star.Teff, errTeff=observed_star.errTeff,
                              logg=observed_star.logg, errlogg=observed_star.errlogg,
                              rl=observed_star.rl, output_dir=observed_star.name + "/dirty_asteroseismic_modelling", mahalanobis_cutoff=1)
                    

        # elif event.inaxes.axes==ax["Q"].axes:
        #     if observed_star != None:
        #         aed = automatic_echelle_diagramm(observed_star.frequencies,  np.ones(len(observed_star.frequencies)), observed_star.name, observed_star.rl, observed_star.Teff, observed_star.errTeff, observed_star.logg, observed_star.errlogg)




    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def load_data(grids, min_Teff = 5999, max_Teff = 11001, min_logg = 3.09, max_logg= 4.71, hes = None, zs = None, other_constraints = None):
    # constraints_dict = {
    #     "mass_low": 0,
    #     "mass_high": 100,
    #     "radius_low": 0,
    #     "radius_high": 1000,
    #     "L_low": 0,
    #     "L_high": 10000000,
    #     "age_low": 0,
    #     "age_high": 1e12,
    # }
    constraints_dict_low = {}
    constraints_dict_high = {}
    if other_constraints != None:
        for const in other_constraints:
            if "low" in const[0]:
                constraints_dict_low[const[0]] = const[1]
            else:
                constraints_dict_high[const[0]] = const[1]


    new = pd.DataFrame(columns=['model_id', 'mass', 'Teff', 'log_g', 'radius', 'L', 'age', 'alphas', 'zs',
                                       'e_fold',
                                       'betas', 'mlins', 'mdots', 'Dmixs', 'f1s', 'hes', 'l_val', 'F', 'O1', 'O2',
                                       'O3',
                                       'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'F_drot', 'O1_drot', 'O2_drot',
                                       'O3_drot',
                                       'O4_drot', 'O5_drot', 'O6_drot', 'O7_drot', 'O8_drot', 'O9_drot', "parent_model_numbers"])

    full_zs = np.linspace(0.006, 0.026, 11)
    full_hes = np.array(["0.204", "0.214", "0.224", "0.234", "0.244", ])
    full_Teffs = np.linspace(6000,11000,21)
    full_loggs = np.linspace(3.1,4.7,17)

    # print(full_Teffs)
    teffs = full_Teffs[full_Teffs >= min_Teff]
    teffs = teffs[teffs <= max_Teff]
    
    # print(teffs)
    loggs = full_loggs[full_loggs >= min_logg]
    loggs = loggs[loggs <= max_logg]
    if zs == None:
        zs = full_zs
    if hes == None:
        hes = full_hes


    deltat = 250
    deltalogg = 0.1
    count = 0
    num_lines = 0
    # print(grids)

    for grid in grids:
        # print(grid)
        for teff in tqdm(teffs):
            # teff_frame = model_data[abs(model_data["Teff"]- teff+deltat/2)<= deltat/2]
            # print(len(teff_frame.index))
            # teff_frame = teff_frame[teff_frame["Teff"] < teff]
            # print(len(teff_frame.index))
            # if len(teff_frame.index) == 0:
            #     continue

            for logg in loggs:
                # log_g_frame = teff_frame[abs(teff_frame["log_g"] - logg+deltalogg/2) < deltalogg/2]
                # log_g_frame = log_g_frame[log_g_frame["log_g"] <logg]
                # if len(log_g_frame.index) == 0:
                #     continue

                for z in zs:
                    # z_frame =  log_g_frame[abs(log_g_frame["zs"]-z) <= 0.0001]
                    # if len(z_frame.index) == 0:
                    #     continue

                    for he in hes:
                        # save =  z_frame[abs(z_frame["hes"]-float(he)) <= 0.001]
                        # if len (save.index) == 0:
                        #     continue
                        # save.reset_index().to_feather("pre-ms-models_database/10_dim_grid_extension/Teff_{:.0f}".format(teff) + "-{:.0f}".format(teff+deltat) + "_logg_{:.1f}".format(logg)+ "-{:.1f}".format(logg+deltalogg) + f"_z_{z}_he_{he}.feather")

                        try:
                            # print(f"pre-ms-models_database/{grid}/" + "Teff_{:.0f}".format(teff) + "-{:.0f}".format(teff+deltat) + "_logg_{:.1f}".format(logg)+ "-{:.1f}".format(logg+deltalogg) + f"_z_{z}_he_{he}.feather")
                            load = pd.read_feather(f"pre-ms-models_database/{grid}/" + "Teff_{:.0f}".format(teff) + "-{:.0f}".format(teff+deltat) + "_logg_{:.1f}".format(logg)+ "-{:.1f}".format(logg+deltalogg) + f"_z_{z}_he_{he}.feather")
                            new = pd.concat([new,load])# print(count)
                            # print(new)
                            count += 1
                        except FileNotFoundError:
                            pass

    # if other_constraints != None:

    for key in constraints_dict_low:
        new = new[new[key.split("_")[0]] > constraints_dict_low[key]]
    for key in constraints_dict_high:
        new = new[new[key.split("_")[0]] < constraints_dict_high[key]]
    # print(new)
    return new

class frequency:
    def __init__(self, f, a):
        self.current_ratio = None
        self.f = f
        self.amp = a
        self.p = 1/self.f

        self.is_tested = [[None, None, None],[None, None, None],[None, None, None],[None, None, None],[None, None, None]]

def def_set(model):
    arr = []
    for key in ['mass', 'Teff', 'log_g', 'L', 'age']:
        arr.append(model.model[key])
    return np.array(arr)


def mode_from_key(key):
    return key[-5:-3]

def split(a, b):
    a = a
    b = b
    a_new = None
    b_new = None
    for i in range(1, len(b)):
        if b[i] == b[i-1]:
            a_new = a.copy()
            b_new = b.copy()
            a.pop(i)
            b.pop(i)
            a_new.pop(i-1)
            b_new.pop(i-1)
            break


    return a, b, a_new, b_new


def empty_ridge():
    return {'F':None, 'O1':None, 'O2':None, 'O3':None, 'O4':None, 'O5':None, 'O6':None, 'O7':None}

def difference(key1, key2):
    if key1 == key2:
        return 0

    elif key1 == 'F':
        return float(key2[-1])
    else:
        return float(key2[-1]) - float(key1[-1])


def write_f(frequency):
    try:
        return frequency.f
    except:
        return 0.0

def frequencies(ridge):
    return np.array([ridge[a] for a in ['F', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7'] if ridge[a] != 0.0])


class ridge:
    def __init__(self, parent, frequencies, modes, l, id):
        self.num_modes = len(modes)
        self.fvals = [f.f for f in frequencies]
        self.frequencies = frequencies
        self.parent = parent
        self.input_mode = modes[0]
        self.modes = empty_ridge()
        self.l = l
        self.xs = None
        self.teffs = None
        self.log_gs = None
        self.best_models = []
        if self.l == 0:
            bound = self.parent.boundaries_l0
        elif self.l == 1:
            bound = self.parent.boundaries_l1
        else:
            bound = self.parent.boundaries_l2
        self.boundaries = bound
        for f, m in zip(frequencies, modes):
            self.modes[m] = f

        self.delta_nu_vals = []
        for i in range(len(modes)):
            for j in range(i+1, len(modes)):
                dif = difference(modes[i], modes[j])
                if dif != 0:
                    self.delta_nu_vals.append((frequencies[j].f - frequencies[i].f)/dif)
        self.delta_nu = np.median(self.delta_nu_vals)

        self.verify_value = None
        self.log_g_xi1 = (None, None)
        self.log_g_xi2 = (None, None)
        self.log_g_xi5 = (None, None)
        self.log_g_xi10 = (None, None)
        self.Teff_xi1 = (None, None)
        self.Teff_xi2 = (None, None)
        self.Teff_xi5 = (None, None)
        self.Teff_xi10 = (None, None)
        self.id = id



    def print(self):
        print('Degree l=', self.l, 'Verfication Value = ', self.verify_value*100, 'Delta Nu = ', self.delta_nu)
        for key in self.modes.keys():
            # print(key , dir(self.modes[key]), self.modes[key])
            try:
                print(key , self.modes[key].f)
            except AttributeError:
                print(key , 'None')

    def plot(self):
        colors = ['forestgreen', 'aquamarine', 'teal', 'darkviolet']
        mosaic = """
                ABF
                EEF
                EEF
                CDF
                """
        fig = plt.figure(constrained_layout=True, figsize=(10,8), dpi = 300)
        ax = fig.subplot_mosaic(mosaic)

        cmap= ax["E"].scatter(self.teffs, self.log_gs, c=self.xs, cmap='viridis', vmin=0.1, vmax=100, s=5, norm=cl.LogNorm())
        ax["E"].plot(self.parent.errbox_x,self.parent.errbox_y, 'k-')
        for i, c in zip(range(min(4, len(self.best_models))), colors):
            ax["E"].plot(self.best_models[i].model['Teff'], self.best_models[i].model['log_g'], marker='X', color=c, ms=10)
        axes = "ABCD"
        for i in range(4):
            a = ax[axes[i]]
            a.plot(np.array(self.parent.fvals)%self.delta_nu,self.parent.fvals, 'kx')
            a.plot(np.array(self.fvals) % self.delta_nu, self.fvals, 'ro', ms=5)
            try:
                mode_fs = frequencies(self.best_models[i].model)
            except IndexError:
                continue
            a.plot(mode_fs % self.delta_nu, mode_fs, marker = '^', color = colors[i], ls = '--', lw = 0)
            a.text(0.05, 0.9, "$\\chi^2 = ${:6.3f}".format(self.best_models[i].xisq), ha='left', va='center', transform=a.transAxes)
            a.tick_params(axis='x', labelsize= 10)
            a.tick_params(axis='y', labelsize=10)
            a.set_xlim(0, self.delta_nu)
        ax['E'].tick_params(axis='x', labelsize=10)
        ax['E'].tick_params(axis='y', labelsize=10)
        axins1 = inset_axes(ax['E'],  # create an inset axis to plot the colormap on such that we can nicely place it
                            width="40%",  # width = 50% of parent_bbox width
                            height="3%",  # height : 5%
                            loc='upper left', bbox_to_anchor=(0.05, 0, 1, 1), bbox_transform=ax['E'].transAxes)

        # plot the colormap and set the label
        cbar = fig.colorbar(cmap, cax=axins1, orientation="horizontal", ax=ax)
        cbar.set_label(r'$\chi^2$', fontsize = 10)
        cbar.ax.tick_params(labelsize=10)

        ax["E"].set_ylim( 4.45, 3.55)
        ax["E"].set_xlim(11250, 4750)
        ax["F"].axis("off")

        ax['B'].yaxis.set_label_position("right")
        ax['D'].yaxis.set_label_position("right")
        ax['B'].yaxis.tick_right()
        ax['D'].yaxis.tick_right()
        ax['A'].xaxis.set_label_position("top")
        ax['B'].xaxis.set_label_position("top")
        ax['A'].xaxis.tick_top()
        ax['B'].xaxis.tick_top()



        for key in ['A', 'B', 'C', 'D']:
            ax[key].set_ylabel(fr'f d$^{-1}$', fontsize = 10)
            ax[key].set_xlabel(r'f \mathrm{mod} {:6.3f}'.format(self.delta_nu) +  ' d$^{-1}$', fontsize = 10)
        ax['E'].set_xlabel("r$T_{eff}$ K", fontsize = 10)
        ax['E'].set_ylabel("log(g)", fontsize = 10)

        ax["F"].text(0.2, 0.9, "Star: " + self.parent.file_name, ha='left', va='center',
               transform=ax["F"].transAxes)
        ax["F"].text(0.2, 0.8, "r$T_{eff}$=" + "{:5} $\\pm$ {:4} K".format(self.parent.Teff, self.parent.errTeff), ha='left', va='center',
               transform=ax["F"].transAxes)
        ax["F"].text(0.2, 0.8, "r$T_{eff}$=" +  "{:5} $\\pm$ {:4} K".format(self.parent.Teff, self.parent.errTeff) , ha='left', va='center',
               transform=ax["F"].transAxes)
        ax["F"].text(0.2, 0.75, f"mode degree: l = {str(self.l)}" , ha='left', va='center',
               transform=ax["F"].transAxes)


        plt.tight_layout()
        plt.savefig(star["Star Name"] + '/automatic_echelle_diagramm/'  f'{self.parent.ridge_file_name + 1}.png' )
        plt.close()


    def write_to_file(self):
        line = '{}      {}	    {}	    {:6.3f}      {:6.3f}	{:6.3f}	{:6.3f}	{:6.3f}	{:6.3f}	{:6.3f}   {:6.3f}     {:6.3f}      {:6.3f}   {:6.3f}\n'.format(self.parent.ridge_file_name, self.id, self.l, self.delta_nu, write_f(self.modes['F']), write_f(self.modes['O1']),
                                                                                 write_f(self.modes['O2']), write_f(self.modes['O3']),
                                                                                 write_f(self.modes['O4']), write_f(self.modes['O5']),
                                                                                 write_f(self.modes['O6']), write_f(self.modes['O7']),
                                                                                                                   self.verify_value, self.best_models[0].xisq )
        self.parent.results_file.write(line)
        # print("Done wriritng", line)




    def verify(self):
        # fig, ax = plt.subplots(self.num_modes, self.num_modes)
        self.verify_value = 0.
        count = 0.
        for key1 in ['F', 'O1', 'O2', 'O3', 'O4']:
            for key2 in self.modes.keys():
                string = key1 + '_' + key2 + '_l' + str(self.l)
                try:
                    poly = self.boundaries[string].polygon
                except KeyError:
                    continue
                try:
                    p = Point(self.modes[key1].p,self.modes[key2].p/self.modes[key1].p )
                except AttributeError:
                    continue
                if poly.contains(p):
                    self.verify_value += 1.
                count +=1.
        self.verify_value = self.verify_value/count

    def search_models(self):
        self.best_models = []
        data = self.parent.model_data[self.parent.model_data['l_val'] == self.l]

        self.xs = []
        # freqs = frequencies(ridge)# self.fvals
        self.teffs = []
        self.log_gs = []
        for i in range(len(data))[::5]:
            models = data.iloc[i]
            xsq = chi_square(self.modes, models, self.parent.rl)
            self.teffs.append(models["Teff"])
            self.log_gs.append(models["log_g"])
            self.xs.append(xsq)
            if xsq < 1:
                self.best_models.append(model(models, xsq))

    def to_modelling_ridge(self):
        r = empty_ridge()
        keys = []
            # print('Degree l=', self.l, 'Verfication Value = ', self.verify_value * 100, 'Delta Nu = ',
            #       self.delta_nu)
        for key in self.modes.keys():
            # print(key , dir(self.modes[key]), self.modes[key])
            try:
                # print(key, self.modes[key].f, self.modes[key].amp)
                r[key] = self.modes[key].f
                keys.append(key)
            except AttributeError:
                # print(key, 'None', 'None')
                pass
        # print(r, keys)
        return modelling_ridge(r, self.l), keys






class model:
    def __init__(self, model, xisq):
        self.model = model
        self.xisq = xisq


def chi_square(ridge, data, rl):
    xisq = 0.0
    count = -1.0
    for key in ['F', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7']:
        if ridge[key] != None and data[key] != 0:
            xisq += (ridge[key].f-data[key])**2
            count += 1
    return xisq/count/rl


def split_fitting(a, b):
    arr_a = [a]
    arr_b = [b]
    stop = False
    while not stop:
        stop = True
        arr_1 = []
        arr_2 = []
        for ca, cb in zip(arr_a, arr_b):
            a1, b1, a2, b2 = split(ca, cb)
            arr_1.append(a1)
            arr_2.append(b1)
            if a2 != None:
                arr_1.append(a2)
                arr_2.append(b2)
                stop = False

        arr_a = arr_1
        arr_b = arr_2

    stop = False
    while not stop:
        stop = True
        for i in range(len(arr_a)):
            for j in range(i +1, len(arr_a)):
                if arr_a[i] == arr_a[j]:
                    arr_a.pop(j)
                    arr_b.pop(j)
                    stop = False
                    break
    return arr_a, arr_b


class automatic_echelle_diagramm:
    def __init__(self, model_data, freqs, amplitudes, file_name, rl = 1.0/24, Teff =  0, errTeff = 0, logg= 0., errlogg = 0., check_l2 = False):
        self.model_data = model_data
        self.freqs = [frequency(f, a) for f, a in zip(freqs, amplitudes)]
        self.fvals = np.array(freqs)
        self.ampvals = amplitudes
        self.file_name = file_name
        self.check_l2 = check_l2
        self.Teff = Teff
        self.errTeff = errTeff
        self.logg = logg
        self.errlogg = errlogg
        self.errbox_x = [Teff - errTeff, Teff + errTeff, Teff + errTeff, Teff - errTeff, Teff - errTeff]
        self.errbox_y = [logg - errlogg, logg - errlogg, logg + errlogg, logg + errlogg, logg - errlogg]
        self.rl = rl
        # self.model_data = np.loadtxt('results_per_model.txt')
        # self.model_data = pd.DataFrame(self.model_data,
        #                           columns=['model_id', 'mass', 'Teff', 'log_g', 'radius', 'L', 'age', 'alphas', 'zs', 'e_fold', 'betas', 'mlins', 'mdots', 'Dmixs', 'f1s', 'hes', 'l_val', 'F', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'F_drot', 'O1_drot', 'O2_drot', 'O3_drot', 'O4_drot', 'O5_drot', 'O6_drot', 'O7_drot', 'O8_drot', 'O9_drot'])
        #
        # self.model_data = self.model_data[self.model_data['Teff'] < 10300]
        self.ridge_id = 1

        made_dir = False
        while not made_dir:
            # try:
            try:
                os.mkdir(star["Star Name"] + '/automatic_echelle_diagramm/' )
            except FileExistsError:
                pass
            made_dir = True
            # except:
            #     self.file_name += '+'

        self.results_file = open(star["Star Name"] + '/automatic_echelle_diagramm/' +  "results.txt", "w")

        line = 'num     ID         l        dnu         F       O1      O2      O3     O4      O5       O6         O7           vF       xisq\n'
        self.results_file.write(line)



        self.amps = np.array(amplitudes)
        self.boundaries_l0 = {}
        self.boundaries_l1 = {}
        self.boundaries_l2 = {}
        for o in ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7']:
            b = bd.boundary('F', o, 0)
            self.boundaries_l0[b.string] = b
            b = bd.boundary('F', o, 1)
            self.boundaries_l1[b.string] = b
            b = bd.boundary('F', o, 2)
            self.boundaries_l2[b.string] = b

        for o in [ 'O2', 'O3', 'O4', 'O5', 'O6', 'O7']:
            b = bd.boundary('O1', o, 0)
            self.boundaries_l0[b.string] = b
            b = bd.boundary('O1', o, 1)
            self.boundaries_l1[b.string] = b
            b = bd.boundary('O1', o, 2)
            self.boundaries_l2[b.string] = b

        for o in [ 'O3', 'O4', 'O5', 'O6', 'O7']:
            b = bd.boundary('O2', o, 0)
            self.boundaries_l0[b.string] = b
            b = bd.boundary('O2', o, 1)
            self.boundaries_l1[b.string] = b
            b = bd.boundary('O2', o, 2)
            self.boundaries_l2[b.string] = b

        for o in [ 'O4', 'O5', 'O6', 'O7']:
            b = bd.boundary('O3', o, 0)
            self.boundaries_l0[b.string] = b
            b = bd.boundary('O3', o, 1)
            self.boundaries_l1[b.string] = b
            b = bd.boundary('O3', o, 2)
            self.boundaries_l2[b.string] = b

        for o in [ 'O5', 'O6', 'O7']:
            b = bd.boundary('O4', o, 0)
            self.boundaries_l0[b.string] = b
            b = bd.boundary('O4', o, 1)
            self.boundaries_l1[b.string] = b
            b = bd.boundary('O4', o, 2)
            self.boundaries_l2[b.string] = b

        self.ridges = []
        self.ridge_file_name = 0
        print('search radial')
        self.search_ridge(0)
        print('search dipole')
        self.search_ridge(1)
        if check_l2:
            print('search quadrupole')
            self.search_ridge(2)

        self.ridges.sort(key=lambda x: x.num_modes, reverse=True)

        to_pop = []
        print("verify ridges\n")
        for i in tqdm(range(len(self.ridges))):

            self.ridges[i].verify()
            if self.ridges[i].verify_value < 0.85 and self.ridges[i].l == 2:
                # print(self.ridges[i].verify_value)
                to_pop.append(i)
            elif self.ridges[i].verify_value < 0.7 and self.ridges[i].l < 2:
                # print(self.ridges[i].verify_value)
                to_pop.append(i)
        for n in reversed(to_pop):
            self.ridges.pop(n)


        to_pop = []

        print("search ridges in models\n")

        for i in tqdm(range(len(self.ridges))):
            self.ridges[i].search_models()
            if not self.ridges[i].best_models:
                to_pop.append(i)

        for n in reversed(to_pop):
            self.ridges.pop(n)

        self.ridges.sort(key=lambda x: x.best_models[0].xisq, reverse=False)


        print("save results")
        for i in tqdm(range(len(self.ridges))):
            # self.ridges[i].plot()
            self.ridge_file_name += 1
            self.ridges[i].write_to_file()


        self.results_file.close()

        # self.plot_best()

        self.ridge_file_name = 0
        print("combine ridges")
        self.combine_ridges()
        print("Automatic Echelle diagram done")
        plt.show()


    def combine_ridges(self):
        for i in tqdm(range(len(self.ridges))):
            for j in range(i, len(self.ridges)):
                if self.ridges[j].l != self.ridges[i].l:
                    # print(len(self.ridges[i].best_models), len(self.ridges[j].best_models))
                    self.combined_search(self.ridges[i], self.ridges[j])


    def combined_search(self, ridge1, ridge2):
        number_plotted = 0
        for m in ridge1.best_models:
            for n in ridge2.best_models:
                if np.max(abs(def_set(m) - def_set(n))) < 0.01:
                    combined_xi = (m.xisq * len(ridge1.fvals) + n.xisq * len(ridge2.fvals))/(len(ridge1.fvals) + len(ridge2.fvals))
                    # print(combined_xi)
                    if combined_xi< 1 and number_plotted < 5:
                        # print(m.model)
                        self.plot_combination( ridge1, ridge2, m, n, combined_xi)
                        number_plotted += 1
                        return

    def markernstyle(self, lval):
        if lval == 0:
            return 'ro'
        elif lval == 1:
            return 'g^'
        else:
            return 'bP'

    def markerblack(self, lval):
        if lval == 0:
            return 'ko'
        elif lval == 1:
            return 'k^'
        else:
            return 'kP'


    def plot_combination(self, r1, r2, m1, m2, combined_xi):
        fig, ax = plt.subplots(figsize = (6,6), dpi = 150)
        dnu = (r1.delta_nu + r2.delta_nu)/2
        dnuplus = 1.2* dnu
        pall,  = ax.plot(self.fvals % dnu, self.fvals, 'kx', alpha = 0.5)
        ax.plot(self.fvals % dnu + dnu, self.fvals, 'kx', alpha = 0.5)

        marker = self.markerblack(int(r1.l))
        pobs1,  = ax.plot(r1.fvals % dnu, r1.fvals, marker, ms = 10, markerfacecolor = 'none')
        ax.plot(r1.fvals % dnu + dnu, r1.fvals, marker, ms = 10, markerfacecolor = 'none')
        marker = self.markerblack(int(r2.l))
        pobs2,  = ax.plot(r2.fvals % dnu, r2.fvals, marker, ms = 10, markerfacecolor = 'none')
        ax.plot(r2.fvals % dnu + dnu, r2.fvals, marker, ms = 10, markerfacecolor = 'none')

        mode_fs = frequencies(m1.model)
        marker = self.markernstyle(int(m1.model['l_val']))
        pmod1, = ax.plot(mode_fs % dnu, mode_fs, marker, ms = 7.5)
        ax.plot(mode_fs % dnu + dnu, mode_fs, marker, ms = 7.5)
        mode_fs = frequencies(m2.model)
        marker = self.markernstyle(int(m2.model['l_val']))
        pmod2, = ax.plot(mode_fs % dnu, mode_fs, marker, ms = 7.5)
        ax.plot(mode_fs % dnu + dnu, mode_fs, marker, ms = 7.5)
        ax.axvline(dnu, color= 'black', ls = '--', lw = 2)
        pemmpt =  Line2D([0],[0],color="w")
        ax.legend((pall,pemmpt, pobs1, pobs2, pmod1, pmod2 ),
                  ('all fs', '', f" observed l = {int(r1.l)}", f" observed l = {int(r2.l)}",
                   f" model l = {int(m1.model['l_val'])}", f" model l = {int(m2.model['l_val'])}"),
                  fontsize = 'small',  bbox_to_anchor=(0., 1.15), loc='upper left', ncol = 3)
        ax.text(0.05, 0.95, "$\\chi^2 = ${:6.3f}".format(combined_xi), ha='left', va='center', transform=ax.transAxes)
        ax.text(0.05, 0.9, f"Ridge IDs {r1.id} {r2.id}", ha='left', va='center', transform=ax.transAxes)
        ax.text(0.05, 0.85, f"Model ID {int(m1.model['model_id'])}", ha='left', va='center', transform=ax.transAxes, fontsize = 'xx-small')
        ax.text(0.99, 0.95, r"$\left(\nu \mathrm{mod} \Delta\nu\right) + \Delta\nu$", ha='right', va='center', transform=ax.transAxes, fontsize='xx-small')
        ax.text(0.8, 0.95, r"$\left(\nu \mathrm{mod} \Delta\nu\right) + \Delta\nu$", ha='right', va='center', transform=ax.transAxes, fontsize='xx-small')

        ax.set_ylabel('Frequency $(d^{-1})$')
        ax.set_xlabel('Frequency modulo ' + '{:.3f}'.format(dnu) + '$ (d^{-1})$')
        ax.set_xlim(0, dnuplus)
        plt.tight_layout()
        plt.savefig(star["Star Name"] + '/automatic_echelle_diagramm/' +  f"combination_ridge{self.ridge_file_name}.pdf")
        # plt.close()
        self.ridge_file_name += 1

        axins = inset_axes(ax, width="100%", height="100%", loc='upper left',
                           bbox_to_anchor=(0.65, 1.05, 0.35, 0.1), bbox_transform=ax.transAxes)

        axins.set_xlim(-1, 1)
        axins.set_ylim(-1, 1)
        axins.fill_between([-0.95, 0.95], y1=[0.95, 0.95], y2=[-0.95, -0.95], ec="k", fc="grey")
        axins.text(0, 0, "Use ID", va="center", ha="center")

        # axins.set_facecolor('grey')
        axins.xaxis.set_visible(False)
        axins.yaxis.set_visible(False)

        # plt.show()
        # plt.close()

        def onclick_combination(event):
            if event.dblclick:
                if event.inaxes.axes == axins.axes:
                    global active_ridges
                    active_ridges = [r1, r2]
                    print("set active ridges to ", active_ridges)
                    return

        cid = fig.canvas.mpl_connect('button_press_event', onclick_combination)
        return fig



    def plot_best(self):
        done = [False, False, False]
        dnu = []
        l0_vals = []
        l1_vals = []
        l2_vals = []
        for i in range(len(self.ridges)):
            r = self.ridges[i]
            if r.l == 0 and  not done[0]:
                l0_vals = r.fvals
                dnu.append(r.delta_nu)
                done[0] = True
            if r.l == 1 and  not done[1]:
                l1_vals = r.fvals
                dnu.append(r.delta_nu)
                done[1] = True

        for i in range(len(self.ridges)):
            r = self.ridges[i]
            if r.l == 2 and  not done[2]:
                skip = False
                for f in r.fvals:
                    if f in l0_vals or f in l1_vals:
                        skip = True
                if not skip:
                    l2_vals = r.fvals
                    dnu.append(r.delta_nu)
                    done[2] = True

        num = 0
        # l2_vals = [25.831,	31.347,	38.053,	44.904,	52.117]
        for deltanu in dnu:
            fig, ax = plt.subplots(figsize = (10,6))
            ax.plot(np.array(self.fvals) % deltanu, self.fvals, 'kx', ms = 10)
            ax.plot(np.array(l2_vals) % deltanu, l2_vals, 'ys', label = 'l=2', ms = 10)
            ax.plot(np.array(l0_vals) % deltanu, l0_vals, 'ro', label = 'l=0', ms = 10)
            ax.plot(np.array(l1_vals) % deltanu, l1_vals, 'gv', label = 'l=1', ms = 10)
            ax.legend()

            ax.set_ylabel('Frequency $(d^{-1})$')
            ax.set_xlabel('Frequency modulo' +  '{:3f}'.format(deltanu) +'$(d^{-1})$')
            plt.tight_layout()
            plt.savefig(star["Star Name"] + '/automatic_echelle_diagramm/' +  f'best_echelle{num}.pdf')
            plt.close()
            num = num + 1

    def search_ridge(self, l):
        if l == 0:
            self.search_radial()
        elif l ==1 :
            self.search_dipole()
        else:
            self.search_quadrupole()

    def search_radial(self):
        fundamental ={key: value for key, value in self.boundaries_l0.items() if key[0] == 'F'}
        self.search_ratios(0.0425, 0.18, fundamental, 0, 'F')
        fovertone ={key: value for key, value in self.boundaries_l0.items() if key[1] == '1'}
        self.search_ratios(0.031, 0.1375, fovertone, 0, 'O1')
        sovertone ={key: value for key, value in self.boundaries_l0.items() if key[1] == '2'}
        self.search_ratios(0.028, 0.11, sovertone, 0, 'O2')
        tovertone ={key: value for key, value in self.boundaries_l0.items() if key[1] == '3'}
        self.search_ratios(0.0235, 0.091, tovertone, 0, 'O3')
        fourthovertone ={key: value for key, value in self.boundaries_l0.items() if key[1] == '4'}
        self.search_ratios(0.02, 0.078, fourthovertone, 0, 'O4')


    def search_dipole(self):
        fundamental ={key: value for key, value in self.boundaries_l1.items() if key[0] == 'F'}
        self.search_ratios(0.0415, 0.175, fundamental, 1, 'F')

        fovertone ={key: value for key, value in self.boundaries_l1.items() if key[1] == '1'}
        self.search_ratios(0.0288, 0.133, fovertone, 1, 'O1')
        sovertone ={key: value for key, value in self.boundaries_l1.items() if key[1] == '2'}
        self.search_ratios(0.0255, 0.105, sovertone, 1, 'O2')
        tovertone ={key: value for key, value in self.boundaries_l1.items() if key[1] == '3'}
        self.search_ratios(0.0215, 0.087, tovertone, 1, 'O3')
        fourthovertone ={key: value for key, value in self.boundaries_l1.items() if key[1] == '4'}
        self.search_ratios(0.0185, 0.75, fourthovertone, 1, 'O4')

    def search_quadrupole(self):
        fundamental ={key: value for key, value in self.boundaries_l2.items() if key[0] == 'F'}
        self.search_ratios(0.035, 0.115, fundamental , 2, 'F')
        fovertone ={key: value for key, value in self.boundaries_l2.items() if key[1] == '1'}
        self.search_ratios(0.0275, 0.101, fovertone , 2, 'O1')
        sovertone ={key: value for key, value in self.boundaries_l2.items() if key[1] == '2'}
        self.search_ratios(0.024, 0.0925, sovertone , 2, 'O2')
        tovertone ={key: value for key, value in self.boundaries_l2.items() if key[1] == '3'}
        self.search_ratios(0.022, 0.8, tovertone, 2, 'O3')
        fourthovertone ={key: value for key, value in self.boundaries_l2.items() if key[1] == '4'}
        self.search_ratios(0.0175, 0.07, fourthovertone, 2, 'O4')

    def search_ratios(self, freq_min, freq_max, boundaries, l, mode):
        if mode == 'F':
            pos1 = 0
        elif mode == 'O1':
            pos1 = 1
        elif mode == 'O2':
            pos1 = 2
        elif mode == 'O3':
            pos1 = 3
        else:
            pos1 = 4

        if l == 0:
            pos2 = 0
        elif l == 1:
            pos2 = 1
        else:
            pos2 = 2
        keep_going = True
        while keep_going:
            possible_freqs = [x for x in self.freqs
                   if x.p > freq_min and x.p < freq_max and x.is_tested[pos1][pos2] != True]

            possible_freqs.sort(key=lambda x: x.amp, reverse=True)
            # for x in possible_freqs:
            #     print(x.f, x.p, x.amp)
            try:
                test_f = possible_freqs[0]
            except IndexError:
                keep_going = False
                continue
            for f in self.freqs:
                f.current_ratio = f.p/test_f.p
            #
            # fig, ax = plt.subplots()
            # for key in boundaries.keys():
            #     boundaries[key].plot(ax, ec = 'green')
            #
            # ax.plot(test_f.p*np.ones(len(self.freqs)), [x.current_ratio for x in self.freqs], 'k*', ms = 3)
            # ax.set_title(test_f.f)
            # plt.show()

            fitting_freqs = [test_f]
            fitting_modes = [mode]
            for f in self.freqs:
                point = Point(test_f.p, f.current_ratio)
                for key in boundaries.keys():
                    if boundaries[key].polygon.contains(point):
                        fitting_freqs.append(f)
                        fitting_modes.append(mode_from_key(key))
            # print(fitting_modes)
            # print(fitting_modes)
            if len(fitting_modes) > 1:
                sort = np.argsort(fitting_modes)
                # print(sort)
                fitting_modes = list(np.array(fitting_modes)[sort])
                fitting_freqs = list(np.array(fitting_freqs)[sort])
                # print('start_splitting')
                # print(fitting_modes)
                all_fitting_f, all_fitting_modes =  split_fitting(fitting_freqs, fitting_modes)

                for fs, ms in zip(all_fitting_f, all_fitting_modes):
                    if len(fs) > 3:
                        self.ridges.append(ridge(self, fs, ms, l, self.ridge_id))
                        self.ridge_id += 1




            # print(fitting)

            # ax.plot(test_f.p*np.ones(len(fitting_freqs[1:])), [x.current_ratio for x in fitting_freqs[1:]], 'c*', ms = 10)
            # ax.set_title(test_f.f)
            #
            #
            # # plt.show()
            # plt.close()
            test_f.is_tested[pos1][pos2] = True

def chi_square_multiple_ridges(ridges,ridges_keys, data_full, rl):
    seismic_xisq = 0.0
    count = 0.0
    for ridge, keys in zip(ridges, ridges_keys):
        data = data_full[data_full['l_val'] == ridge.l]
        try:
            vals =  [(ridge.data[key]-data[key].values[0])**2 for key in keys]
        except IndexError:
            print(data_full[['l_val', 'F', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7']])
        count += len(vals)
        seismic_xisq += np.sum(vals)
    return seismic_xisq/rl, count

class result_dict:
    def __init__(self, results, xi_cutoff = 0, classic_box = 'None'):
        self.data = {}
        if xi_cutoff != 0:
            results = np.array([x for x in results if x.xisq < xi_cutoff])
        if classic_box == '1sigma':
            results = np.array([x for x in results if x.inbox1])
        elif classic_box == '3sigma':
            results = np.array([x for x in results if x.inbox3])

        for key in ['mass', 'Teff', 'log_g','radius', 'L', 'age', 'alphas', 'zs', 'e_fold', 'betas', 'mlins', 'mdots', 'Dmixs', 'f1s', 'hes']:
            self.data[key] = np.array([x.model[key].values[0] for x in results])
        self.data['xisq'] = np.array([x.xisq for x in results])
        self.data['spec_xisq'] = np.array([x.spec_xisq for x in results])


class modelling_model:
    def __init__(self, model, seismic_xisq,spec_xisq, inbox1, inbox3):
        self.model = model
        self.xisq = seismic_xisq
        self.spec_xisq = spec_xisq
        self.inbox1 = inbox1
        self.inbox3 = inbox3
        self.mahalanobis_distance = None
        self.mahalanobis_vector = None

class modelling:
    def __init__(self, model_data, ridges, ridges_keys, rl = 1.0/24, Teff =  0, errTeff = 0, logg= 0., errlogg = 0., data = None, output_dir = '', mahalanobis_cutoff = 1.0, mahalanobis_keys = None, mahalanobis_vals = None, mahalanobis_uncertainty_vals = None):
        self.ridges = ridges
        self.ridges_keys = ridges_keys
        self.rl = rl
        self.Teff = Teff
        self.errTeff = errTeff
        self.logg = logg
        self.errlogg = errlogg
        self.errbox_Teff = [Teff - errTeff, Teff + errTeff, Teff + errTeff, Teff - errTeff, Teff - errTeff]
        self.errbox_logg = [logg - errlogg, logg - errlogg, logg + errlogg, logg + errlogg, logg - errlogg]
        self.results = []
        self.data = data
        self.mahalanobis_matrix = None
        self.mahalanobis_cutoff = mahalanobis_cutoff
        self.mahalanobis_mean = None
        self.mahalanobis_vals = mahalanobis_vals
        self.mahalanobis_uncertainty_vals = mahalanobis_uncertainty_vals

        self.observed_vector = []
        self.uncertainties_vector = []
        for r, k in zip(self.ridges, self.ridges_keys):
            for key in k:
                self.observed_vector.append(r.data[key])
                self.uncertainties_vector.append(self.rl)
        if mahalanobis_keys != None:
            self.mahalanobis_keys = mahalanobis_keys
            for key in range(len(self.mahalanobis_vals)):
                self.observed_vector.append(self.mahalanobis_vals[key])
                self.uncertainties_vector.append(self.mahalanobis_uncertainty_vals[key])
        else:
            self.mahalanobis_keys = []
        self.observed_vector = np.array(self.observed_vector)
        self.uncertainties_vector = np.array(self.uncertainties_vector)





        self.model_data = model_data
        # self.model_data = np.loadtxt('results_per_model.txt')
        # self.model_data = np.loadtxt('results_6_dim_model.txt')
        # # self.model_data = np.loadtxt('results_per_model_incl_murphy.txt')
        # self.model_data = pd.DataFrame(self.model_data,
        #                           columns=['model_id', 'mass', 'Teff', 'log_g', 'radius', 'L', 'age', 'alphas', 'zs', 'e_fold', 'betas', 'mlins', 'mdots', 'Dmixs', 'f1s', 'hes', 'l_val', 'F', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'F_drot', 'O1_drot', 'O2_drot', 'O3_drot', 'O4_drot', 'O5_drot', 'O6_drot', 'O7_drot', 'O8_drot', 'O9_drot'])

        self.model_data = self.model_data[self.model_data['Teff'] < 10300]
        self.search_models()


        self.results.sort(key=lambda x: x.xisq, reverse=False)



        # self.calculate_mahalanobis_inverse(self.mahalanobis_cutoff)
        # self.calculate_mahalanobis_distance()



        if output_dir != '':
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            self.output_dir = output_dir + '/'
            
        else:
            self.output_dir = output_dir

        self.plot_results_one()
        self.plot_results_one( xisq='spec_xisq')
        
        # try:
        #     self.plot_results_two()
        # except AssertionError:
        #     pass
        # except ValueError:
        #     pass
        # try:
        #     self.plot_results_two(classic_box='3sigma')
        # except AssertionError:
        #     pass
        # except ValueError:
        #     pass
        # try:
        #     self.plot_results_two(classic_box='1sigma')
        # except AssertionError:
        #     pass
        # except ValueError:
        #     pass

        for i in range(5):
            self.plot_echelle(self.ridges, self.ridges_keys, self.results[i], mode = 'xisq')


        self.results.sort(key=lambda x: x.spec_xisq, reverse=False)
        for i in range(5):
            self.plot_echelle(self.ridges, self.ridges_keys, self.results[i], mode = 'spec_xisq')


        del self.results

        print("Done Modelling")
        
        # print(ridges_keys)
        get_expection_values(self.output_dir, len(ridges_keys[0]) + len(ridges_keys[1]),Teff, errTeff,
                              logg, errlogg,)
        # for m in self.results:
        #     print(m.xisq, m.mahalabonis_distance)

        # self.results.sort(key=lambda x: x.mahalabonis_distance, reverse=False)
        # for i in range(3):
        #     self.plot_echelle(self.ridges, self.ridges_keys, self.results[i], type = 'mahalanobis')


    def calculate_mahalanobis_inverse(self, xi_cutoff):
        mean = []
        for r, k in zip(self.ridges, self.ridges_keys):
            for key in k:
                mean.append(0.0)
        for key in self.mahalanobis_keys:
            mean.append(0.0)

        mean = np.array(mean)

        num = 0
        for m in self.results:
            vector = []
            for r, k in zip(self.ridges, self.ridges_keys):
                data_l = m.model[m.model['l_val'] == r.l]
                for key in k:
                    vector.append(data_l[key].values[0])
            for key in self.mahalanobis_keys:
                vector.append(data_l[key].values[0][key])
            m.mahalabonis_vector = np.array(vector)

            if m.xisq <= xi_cutoff and m.inbox1:
                num += 1
                mean += m.mahalabonis_vector
        mean = mean / num

        # print(mean)
        count = 0
        mod1 = self.results[0]
        dif = np.array([mod1.mahalabonis_vector - mean])
        matrix = np.matmul(dif.T, dif)

        for m in self.results[1:]:
            if m.xisq > xi_cutoff:
                continue
            if not m.inbox1:
                continue
            count += 1
            dif = np.array([m.mahalabonis_vector - mean])
            matrix += np.matmul(dif.T, dif)
        matrix = matrix/count
        # print(matrix)
        self.mahalanobis_matrix = np.linalg.inv(matrix + self.uncertainties_vector)


        # fig, ax = plt.subplots()
        # ax.imshow(matrix + self.uncertainties_vector)
        # plt.show()

    def calculate_mahalanobis_distance(self):
        for m in self.results:
            m.mahalabonis_distance = np.matmul(np.array([m.mahalabonis_vector - self.observed_vector]), np.matmul(self.mahalanobis_matrix,np.array([m.mahalabonis_vector - self.observed_vector]).T ))[0][0]


    #

    def search_models(self):
        for i in tqdm(range(1, int(len(self.model_data['model_id'])/3))):
        # for i in tqdm(range(0, 20000)):
            models = self.model_data.iloc[3 * i: 3 * i + 3]
            seismic_xsq, count = chi_square_multiple_ridges(self.ridges, self.ridges_keys, models, self.rl)
            inbox1 = False
            inbox3 = False
            if abs(models["log_g"].values[0] - self.logg) < self.errlogg and abs(models["Teff"].values[0] - self.Teff) < self.errTeff:
                inbox1 = True
            if abs(models["log_g"].values[0] - self.logg) < 3*self.errlogg and abs(models["Teff"].values[0] - self.Teff) < 3*self.errTeff:
                inbox3 = True
            spec_xsq = (seismic_xsq + (self.Teff - models["Teff"].values[0])**2/self.errTeff + (self.logg - models["log_g"].values[0])**2/self.errlogg) /(count + 2)
            self.results.append(modelling_model(models, seismic_xsq/count, spec_xsq,  inbox1, inbox3))


    def plot_results_one(self, classic_box = 'None', xisq  = 'xisq'):
        mosaic = """
                    AAABCD
                    AAAEFG
                    AAAHIJ
                    KLMNOP
                    """
        fig = plt.figure(constrained_layout=True, figsize=(13,10), dpi = 300)
        ax = fig.subplot_mosaic(mosaic)


        results_dicts = result_dict(self.results, classic_box = classic_box)
        if classic_box == 'None':
            a_file = open(self.output_dir + "results_dict_cl.pkl", "wb")
            pickle.dump(results_dicts.data, a_file)
            a_file.close()
            gridsize = 80
        elif classic_box == '3sigma':
            gridsize = 50
        else:
            gridsize = 25

        ax["A"].hexbin(results_dicts.data['Teff'], results_dicts.data['log_g'], C= results_dicts.data[xisq], vmin = 0, vmax = 10,  reduce_C_function = np.amin, gridsize = gridsize)
        ax["A"].plot(self.errbox_Teff, self.errbox_logg, 'k-')
        for a, key in zip(["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
                          ['mass', 'Teff', 'log_g','radius', 'L', 'age', 'alphas', 'zs', 'e_fold', 'betas',
                           'mlins', 'mdots', 'Dmixs', 'f1s', 'hes']):
            try:
                ax[a].axvline(self.data[key], color = 'red')
            except:
                pass
            ax[a].axhline(1, color = 'blue')
            ax[a].plot(results_dicts.data[key],results_dicts.data[xisq], 'ko', ms = 0.5 )
            ax[a].set_yscale('log')
            ax[a].set_xlabel(key)
        ax["A"].invert_yaxis()
        plt.savefig(self.output_dir +f'overview_plot_{classic_box}_{xisq}.png')
        plt.close()


    def plot_results_two(self, classic_box = 'None'):
        result = result_dict(self.results, xi_cutoff=1, classic_box= classic_box)
        corner.corner(result.data, show_titles=True, quantiles=(0.16, 0.84), levels=(1-np.exp(-0.5),))
        plt.savefig(self.output_dir +f'Corner_plot{classic_box}.png')
        names = ['mass', 'Teff', 'log_g', 'radius',  'L', 'age', 'alphas', 'zs', 'e_fold', 'betas',
                           'mlins', 'mdots', 'Dmixs', 'f1s', 'hes']
        f = open(self.output_dir +f"Resuting_parameters_{classic_box}.txt", "w")
        for i in range(14):  # must be done once per variable
            q_16, q_50, q_84 = corner.quantile(result.data[names[i]], [0.16, 0.5, 0.84])  # your x is q_50
            dx_down, dx_up = q_50 - q_16, q_84 - q_50
            f.write(names[i] +  '{0:.3E}'.format(self.data[names[i]]) +  'result: {0:.3E}'.format(q_50) + '    - {0:.3E}'.format(dx_down) + '+     {0:.3E}\n'.format(dx_up))
        f.close()
        plt.close()

    def plot_echelle(self, ridges, keys, model, type = 'xisq', mode = 'xisq'):
        fig, ax = plt.subplots(figsize=(6, 6))
        dnu = 0
        for r in ridges:
            dnu += r.delta_nu
        dnu = dnu/len(ridges)
        dnuplus = 1.2 * dnu
        for r, k in zip(ridges, keys):
            marker = self.markerblack(int(r.l))
            ax.plot(r.fvals % dnu, r.fvals, marker, ms=10, markerfacecolor='none')
            ax.plot(r.fvals % dnu + dnu, r.fvals, marker, ms=10, markerfacecolor='none')
            model_part = model.model[model.model['l_val'] == r.l]
            marker = self.markernstyle(int(r.l))
            fvals = []
            for key in k:
                if model_part[key].values[0] != 0:
                    fvals.append(model_part[key].values[0])
            fvals = np.array(fvals)
            ax.plot(fvals % dnu, fvals, marker, ms=7.5)
            ax.plot(fvals % dnu + dnu, fvals, marker, ms=7.5)

        ax.axvline(dnu, color='black', ls='--', lw=2)
        pemmpt = Line2D([0], [0], color="w")


        ax.text(0.99, 0.95, r"$(\nu \mathrm{mod} \Delta\nu) + \Delta\nu$", 
                ha='right', va='center', transform=ax.transAxes)

        ax.text(0.8, 0.95, r"$(\nu \mathrm{mod\Delta\nu) + \Delta\nu$", 
                ha='right', va='center', transform=ax.transAxes, fontsize='xx-small')


        ax.set_ylabel('Frequency $(d^{-1})$')
        ax.set_xlabel('Frequency modulo ' + '{:.3f}'.format(dnu) + '$ (d^{-1})$')
        ax.set_xlim(0, dnuplus)
        plt.tight_layout()
        if type == 'xisq':
            plt.savefig(self.output_dir + f'best_echelle_diagramm' + str(int(model.model['model_id'].values[0])) + f'{mode}.png')
        elif type == 'mahalanobis':

            plt.savefig(self.output_dir + f'best_echelle_diagramm' + str(int(model.model['model_id'].values[0])) + 'mahalanobis.png')
        # plt.show()
        plt.close()

    def markernstyle(self, lval):
        if lval == 0:
            return 'ro'
        elif lval == 1:
            return 'g^'
        else:
            return 'bP'

    def markerblack(self, lval):
        if lval == 0:
            return 'ko'
        elif lval == 1:
            return 'k^'
        else:
            return 'kP'
            

def get_expection_values(filename, DoF, Teff, errTeff, logg, errlogg):
# file = open('model_10001_rl_54_num_random_56.96707348622678_0.06838001542591794/results_dict_cl.pkl', 'rb')
    file = open(filename+ '/results_dict_cl.pkl', 'rb')
    data = pickle.load(file)
    data = pd.DataFrame.from_dict(data)


    p_values_005 = {"1": 3.841, "2": 5.991, "3": 7.815, "4": 9.488, "5": 11.070, "6": 12.592, "7": 14.067, "8": 15.507, "9": 16.919, "10": 18.307, "11": 19.675, "12": 21.026, "13": 22.362, "14": 23.685, "15": 24.996, "16": 26.296, "17": 27.587, "18": 28.869, "19": 30.144, "20": 31.410, }


    sigma_box3 = data[abs(data['Teff'] - Teff) < 3*errTeff]
    sigma_box3 = sigma_box3[abs(sigma_box3['log_g'] - logg) < 3*errlogg]

    # sigma_box1 = sigma_box3[abs(sigma_box3['Teff'] - Teff) < errTeff]
    # sigma_box1 = sigma_box1[abs(sigma_box1['log_g'] - logg) < errlogg]

    # spec_sigma1 = data[data['spec_xisq'] <= 1]
    spec_sigmap005 = data[data['spec_xisq'] <= p_values_005[str(DoF)]]

    # np.savetxt('approach_5_betas_mlins_mdots.txt', np.array([spec_sigmap005['betas'].values, spec_sigmap005['mlins'].values, spec_sigmap005['mdots'].values]).T)
    # print(spec_sigmap005['betas'].values)
    # data = data[data['xisq'] <= 1]
    sigma_box3 = sigma_box3[sigma_box3['xisq'] <= 1]
    # sigma_box1 = sigma_box1[sigma_box1['xisq'] <= 1]
    # real = model_data(10001)
    for key in ['mass', 'Teff', 'log_g', 'radius', 'L', 'age', 'zs', ]:
        fig, ax = plt.subplots(1,2, sharex=False, figsize = (10,6), dpi = 200)
        print('Value:', key) #, 'real:', real[key])
        count = 0
        for set, name in zip([sigma_box3, spec_sigmap005], ["sigma_box3", "spec_sigmap005"]):
            if key != 'zs' :
                print(name, np.mean(set[key]), np.std(set[key]))
                
                ax[count].hist(set[key], density = True, fc = "yellowgreen", ec= "grey")

                lims = ax[count].get_xlim()
                # ax[count].axvline(real[key], color='red')
                x = np.linspace(lims[0], lims[1], 1000)
                # print(x)
                # print( np.exp(-0.5*(x-np.mean(set[key])**2)/np.std(set[key])**2))
                ax[count].plot(x, 1/(np.std(set[key])*np.sqrt(2*np.pi))*np.exp(-0.5*(x-np.mean(set[key]))**2/np.std(set[key])**2), "r-")
                ax[count].axvline(np.mean(set[key]), color='black')
                ax[count].axvline(np.mean(set[key]) - np.std(set[key]), color='black', ls = '--')
                ax[count].axvline(np.mean(set[key]) + np.std(set[key]), color='black', ls = '--')
                ax[count].set_xlim(lims)
                ax[count].set_xlabel(key)
                ax[count].set_ylabel("density")
                ax[count].set_title(name)
                
                count += 1
            else:
                print(name, np.median(set[key]), np.std(set[key]))
                ax[count].hist(set[key], density = True, fc = "yellowgreen", ec= "grey")

                
                lims = ax[count].get_xlim()
                # ax[count].axvline(real[key], color='red')
                x = np.linspace(lims[0], lims[1], 1000)
                
                ax[count].plot(x, 1/(np.std(set[key])*np.sqrt(2*np.pi))*np.exp(-0.5*(x-np.median(set[key]))**2/np.std(set[key])**2), "r-")
                # ax[count].axvline(real[key], color='red')
                
                ax[count].axvline(np.median(set[key]), color='black')
                ax[count].axvline(np.median(set[key]) - np.std(set[key]), color='black', ls = '--')
                ax[count].axvline(np.median(set[key]) + np.std(set[key]), color='black', ls = '--')
                
                ax[count].set_xlim(lims)
                ax[count].set_xlabel(key)
                ax[count].set_ylabel("density")
                ax[count].set_title(name)
                count += 1
            
        plt.tight_layout()
        plt.savefig(star["Star Name"] + "/dirty_asteroseismic_modelling/" + key + ".png")
        plt.close()
        # plt.show()
        # if key == 'age':
        #     figs, acs = plt.subplots()
        #     acs.plot(sigma_box3[key], sigma_box3['xisq'], 'ko')
        #     plt.show()
        # plt.close()


def analyse(star, frequencies, rayleigh_limit, model_grids, min_Teff = 5999, max_Teff = 11001, min_logg = 3.09, max_logg= 4.71, hes = None, zs = None, other_constraints = None):
    try:
        os.mkdir(star["Star Name"])
    except:
        pass
    star_object = observed_star(star["logTeff"], star["errlogTeff"], star["logg"], star["errlogg"], frequencies, star["Star Name"], rayleigh_limit)
    model_data = load_data(model_grids, min_Teff= min_Teff, max_Teff=max_Teff, min_logg=min_logg, max_logg=max_logg, hes = hes, zs = zs,  other_constraints = other_constraints)
    parameter_overview_plot(model_data ,  star_object)
    

# fs = [8.9451,11.3175,13.1139,15.7096,18.6209,18.9390,19.2725,19.9133,20.3081,20.5458,20.8734,21.3234,21.6618,22.5726,23.8711,24.1423,24.9067,24.9847,25.2163,25.2835,25.5571,26.0588,26.6828,27.0188,27.8288,28.1720,28.2860,28.2956,29.5497,30.1113,31.7397,32.2569,32.5485,32.9529,35.1658,35.6122,36.2495,37.8695,42.7297]
# fs = [9.185,	10.881,19.198,	22.286 ,  24.910   ,  27.577, 18.245,	20.883  , 23.651   ,  26.221]

fs = [
    59.962,
    60.172,
    60.248,
    63.699,
    63.756,
    63.852,
    63.865,
    67.274,
    67.504,
    67.644,
    70.930,
    71.091,
    131.030,
    138.362
]

# tic_354824205 = observed_star(np.log10(7150), np.log10(7150+500) - np.log10(7150), 3.75, 0.3, fs, "TYC 4470-591-1", 1/24)

# hd_139614 = observed_star(np.log10(7650), np.log10(7650+200) - np.log10(7650), 4.31, 0.12, [21.365, 27.953, 34.776, 41.537, 48.38, 55.257, 61.978, 26.732, 32.553, 38.406, 58.461], "hd_139614", 1/24)

other_constraints = None
# other_constraints = [("age_low", 5e6), ("age_high", 6e6), ("L_low", 0), ("L_high", 50)]
# # model_data = load_data(["10_dim_grid"], min_Teff= 6000, max_Teff=8000, min_logg=3.5, max_logg=4.1, other_constraints = other_constraints)
# model_data = load_data(["10_dim_grid"],)
# parameter_overview_plot(model_data ,  tic_354824205)

# get_expection_values("TYC 4470-591-1", 10, 7150, 500, 3.75, 0.3)

star = {"Star Name": "HD144277",
    "logTeff": np.log10(9200), 
    "errlogTeff": np.log10(9200) - np.log10(7200), 
    "logg": 4,
    "errlogg": 0.3}
    


analyse(star, fs, 1/10, ["10_dim_grid"], min_Teff = 6500, max_Teff = 10000, min_logg = 3.7, max_logg= 4.9, hes = None, zs = None, other_constraints = other_constraints)
# analyse(star, fs, 1/24, ["10_dim_grid"])