import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from PIL import Image
from pims_nd2 import ND2_Reader as nd2_opener
from os import listdir
from os.path import isfile, join
import warnings
import sys

def Num_to_Str(_i):

    if _i < 10:
        out = '000' + str(_i)
    if _i >= 10 and _i < 100:
        out = '00' + str(_i)
    if _i >= 100 and _i < 1000:
        out = '0' + str(_i)
    if _i > 1000:
        out = str(_i)

    return out

def Get_Image_Size(_path_name, verbose=False):

    onlyfiles = [f for f in listdir(_path_name) if isfile(join(_path_name, f)) and join(_path_name, f).endswith('.nd2')]

    _h = nd2_opener(join(_path_name, onlyfiles[0])).metadata['height']
    _w = nd2_opener(join(_path_name, onlyfiles[0])).metadata['width']
    # _c = nd2_opener(join(_path_name, onlyfiles[0])).metadata['channels']
    # _z = nd2_opener(join(_path_name, onlyfiles[0])).metadata['z_levels']
    _shape = np.array(nd2_opener(join(_path_name, onlyfiles[0]))).shape

    if verbose:
        print('Height:', _h, 'Width:', _w)
        print('Image shape:', _shape)

    return _h, _w

def Local_to_Global(_P , _M, _Size):
    #if isinstance(_M, pd.DataFrame):
    #    _M = _M.to_numpy()

    _P = np.array(_P, dtype=int)

    _P_out = np.empty([len(_P), 2], dtype=int)

    for p in range(len(_P)):

        M_i, M_j = np.where(_M == _P[p, 0])
        M_i = np.squeeze(M_i)
        M_j = np.squeeze(M_j)

        _P_out[p, 0] = int(_Size[0]*M_i + _P[p, 1])
        _P_out[p, 1] = int(_Size[1]*M_j + _P[p, 2])

    return _P_out

def Global_to_Local(_P, _M, _Size):
    #if isinstance(_M, pd.DataFrame):
    #    _M = _M.to_numpy()

    _P = np.array(_P, dtype=int)

    _P_out = np.empty([len(_P), 3], dtype=int)

    for p in range(len(_P)):

        M_i = int(_P[p, 0] / _Size[0])
        M_j = int(_P[p, 1] / _Size[1])

        _P_out[p, 0] = int(_M[M_i, M_j])

        _P_out[p, 1] = int(_P[p, 0] - M_i * _Size[0])
        _P_out[p, 2] = int(_P[p, 1] - M_j * _Size[1])

    return _P_out

def Arrange_Tiles(_path_name, plot_location=False):
    print('Arranging tiles from directory')

    warnings.filterwarnings('ignore')

    onlyfiles = [f for f in listdir(_path_name) if isfile(join(_path_name, f)) and join(_path_name, f).endswith('.nd2')]

    assert len(onlyfiles)!=0, 'No nd2 files found'

    Coord = np.empty([len(onlyfiles), 5], dtype=object)
    for t, tile_name in enumerate(tqdm(onlyfiles)):

        img = nd2_opener(join(_path_name,tile_name))

        Coord[t, 0] = tile_name
        Coord[t, 1] = img[0].metadata['t_ms']
        Coord[t, 2] = img[0].metadata['x_um']
        Coord[t, 3] = img[0].metadata['y_um']
        Coord[t, 4] = img[0].metadata['mpp']

    _Tile_List = pd.DataFrame(Coord, columns=['name', 't', 'x', 'y', 'mpp'])
    _Tile_List = _Tile_List.sort_values(by=['t'])
    _Tile_List.insert(loc=1, column='position', value=range(len(_Tile_List)))
    Coord = _Tile_List.to_numpy()

    h, w = np.array(img[0]).shape

    X = (Coord[:, 3] - Coord[:, 3].min())/(Coord[:, 5]*w)
    Y = (Coord[:, 4] - Coord[:, 4].min())/(Coord[:, 5]*h)

    _M = -1 * np.ones([int(Y.max()+1), int(X.max()+1)], dtype=int)
    for i in range(len(X)):
        _M[int(round(Y[i])), int(round(X[i]))] = i
    # _M = pd.DataFrame(_M)

    if plot_location:
        plt.scatter(X, Y, c='blue', s=4)
        for p in range(len(onlyfiles)):
            plt.text(X[p], Y[p], str(p), c='blue')
        plt.show()

    return _M, _Tile_List

def model_TRS(_P_10X_, dof, angle='degree'):

    _P_10X_ = np.array(_P_10X_)

    if angle == 'degree':
        a = np.pi * dof[2] / 180
    else:
        a = dof[2]

    X = dof[3] * (np.cos(a) * _P_10X_[:, 0] - np.sin(a) * _P_10X_[:, 1] + dof[0])
    Y = dof[4] * (np.sin(a) * _P_10X_[:, 0] + np.cos(a) * _P_10X_[:, 1] + dof[1])

    _P_out = np.concatenate((X.reshape(len(X),1), Y.reshape(len(Y),1)), axis=1)

    return _P_out

def fun_min(x, _P_10X_, _P_40X_, _angle='degree'):

    _P_10X_ = np.array(_P_10X_)
    _P_40X_ = np.array(_P_40X_)

    assert _P_10X_.shape == _P_40X_.shape, 'Number of points must be the same in both magnifications'

    _P = model_TRS(_P_10X_, x, angle=_angle)

    obj = np.sum((_P[:, 0] - _P_40X_[:, 0]) ** 2 + (_P[:, 1] - _P_40X_[:, 1]) ** 2)

    return np.sum(obj)

def Fit_By_Points(_P_10X, _P_40X, verbose=False):

    _P_10X = np.array(_P_10X)
    _P_40X = np.array(_P_40X)

    x0 = (0, 0, 0, 0, 0)

    res = minimize(fun_min, x0, args=(_P_10X, _P_40X), method='SLSQP')

    _P = model_TRS(_P_10X, res.x)

    if verbose:
        print(res)
        print('Pixel Errors (X,Y)')
        for p in range(len(_P)):
            print('Point ', p, [round(_P_40X[p, 0] - _P[p, 0]), round(_P_40X[p, 1] - _P[p, 1])])

    return res.x

def Check_Tile_Configuration(_path_name, _M, file_type='nd2', plot_align = False, print_config = False, plot_tiles = False):
    print('Verifying tiles align with tile-map')

    def get_last2d(data):
        if data.ndim <= 2:
            return data
        slc = [0] * (data.ndim - 2)
        slc += [slice(None), slice(None)]
        return data[slc]

    warnings.filterwarnings('ignore')

    if file_type == 'nd2':
        onlyfiles = [f for f in listdir(_path_name) if isfile(join(_path_name, f)) and join(_path_name, f).endswith('.nd2')]
        assert len(onlyfiles)!=0, 'No nd2 files found'

    if file_type == 'tif':
        onlyfiles = [f for f in listdir(_path_name) if isfile(join(_path_name, f)) and join(_path_name, f).endswith('.tif')]
        assert len(onlyfiles)!=0, 'No tif files found'

    transformation_list = ['None', 'transpose', 'fliplr', 'flipud', 'flip', 'transpose -> flipud', 'transpose -> fliplr', 'transpose -> flip']

    obj_list = np.empty([8])
    # _M = _M.to_numpy()
    M_I, M_J = _M.shape
    M_list = np.empty([8, M_I, M_J])

    if M_I > M_J:
        E = -1*np.ones([M_I, int(M_I-M_J)])
        _M = np.concatenate((_M, E), axis=1)
        M_list = np.empty([8, M_I, M_I])

    if M_I < M_J:
        E = -1*np.ones([int(M_J-M_I), M_J])
        _M = np.concatenate((_M, E), axis=0)
        M_list = np.empty([8, M_J, M_J])

    MC_I = int(M_I / 2 - 1)
    MC_J = int(M_J / 2 - 1)
    if MC_I < 0:
        MC_I = 0
    if MC_J < 0:
        MC_J = 0

    M_list[0] = _M
    M_list[1] = np.transpose(_M)
    M_list[2] = np.fliplr(_M)
    M_list[3] = np.flipud(_M)
    M_list[4] = np.flip(_M)
    M_list[5] = np.fliplr(np.transpose(_M))
    M_list[6] = np.flipud(np.transpose(_M))
    M_list[7] = np.flip(np.transpose(_M))

    for m in range(8): # tqdm(range(8)):

        if file_type == 'nd2':
            img_0 = get_last2d(np.array(nd2_opener(join(_path_name, onlyfiles[int(M_list[m][MC_I, MC_J])])), dtype=np.float64))
            img_1 = get_last2d(np.array(nd2_opener(join(_path_name, onlyfiles[int(M_list[m][MC_I + 1, MC_J])])), dtype=np.float64))
            img_2 = get_last2d(np.array(nd2_opener(join(_path_name, onlyfiles[int(M_list[m][MC_I, MC_J + 1])])), dtype=np.float64))
            img_3 = get_last2d(np.array(nd2_opener(join(_path_name, onlyfiles[int(M_list[m][MC_I + 1, MC_J + 1])])), dtype=np.float64))

        if file_type == 'tif':
            img_0 = get_last2d(np.array(Image.open(join(_path_name, onlyfiles[int(M_list[m][MC_I, MC_J])])), dtype=np.float64))
            img_1 = get_last2d(np.array(Image.open(join(_path_name, onlyfiles[int(M_list[m][MC_I + 1, MC_J])])), dtype=np.float64))
            img_2 = get_last2d(np.array(Image.open(join(_path_name, onlyfiles[int(M_list[m][MC_I, MC_J + 1])])), dtype=np.float64))
            img_3 = get_last2d(np.array(Image.open(join(_path_name, onlyfiles[int(M_list[m][MC_I + 1, MC_J + 1])])), dtype=np.float64))

        R_i, R_j = img_0.shape

        if plot_align:
            plt.suptitle(transformation_list[m])

            plt.subplot(2,2,1)
            plt.plot(range(R_i), img_0[:, -1])
            plt.plot(range(R_i), img_2[:, 0])

            plt.subplot(2,2,2)
            plt.plot(range(R_j), img_0[-1, :])
            plt.plot(range(R_j), img_1[0, :])

            plt.subplot(2,2,3)
            plt.plot(range(R_i), img_1[:, -1])
            plt.plot(range(R_i), img_3[:, 0])

            plt.subplot(2,2,4)
            plt.plot(range(R_j), img_2[-1, :])
            plt.plot(range(R_j), img_3[0, :])

            plt.show()

        bnd_0 = np.sum(np.abs(img_0[:, -1] - img_2[:, 0]))
        bnd_1 = np.sum(np.abs(img_0[-1, :] - img_1[0, :]))
        bnd_2 = np.sum(np.abs(img_1[:, -1] - img_3[:, 0]))
        bnd_3 = np.sum(np.abs(img_2[-1, :] - img_3[0, :]))
        obj = bnd_0 + bnd_1 + bnd_2 + bnd_3

        obj_list[m] = np.log10(obj)

    if print_config:
        for k in range(8):
            print(transformation_list[k], obj_list[k])

        print('Correct configuration:', transformation_list[np.argmin(obj_list)])

    _M_crt = np.array(M_list[np.argmin(obj_list)], dtype=int)

    if plot_tiles:

        M_I, M_J = _M_crt.shape
        MC_I = int(M_I / 2 - 1)
        MC_J = int(M_J / 2 - 1)
        if MC_I < 0:
            MC_I = 0
        if MC_J < 0:
            MC_J = 0

        if file_type == 'nd2':
            img_0 = get_last2d(np.array(nd2_opener(join(_path_name, onlyfiles[int(_M_crt[MC_I, MC_J])])), dtype=np.float64))
            img_1 = get_last2d(np.array(nd2_opener(join(_path_name, onlyfiles[int(_M_crt[MC_I + 1, MC_J])])), dtype=np.float64))
            img_2 = get_last2d(np.array(nd2_opener(join(_path_name, onlyfiles[int(_M_crt[MC_I, MC_J + 1])])), dtype=np.float64))
            img_3 = get_last2d(np.array(nd2_opener(join(_path_name, onlyfiles[int(_M_crt[MC_I + 1, MC_J + 1])])), dtype=np.float64))

        if file_type == 'tif':
            img_0 = get_last2d(np.array(Image.open(join(_path_name, onlyfiles[int(_M_crt[MC_I, MC_J])])), dtype=np.float64))
            img_1 = get_last2d(np.array(Image.open(join(_path_name, onlyfiles[int(_M_crt[MC_I + 1, MC_J])])), dtype=np.float64))
            img_2 = get_last2d(np.array(Image.open(join(_path_name, onlyfiles[int(_M_crt[MC_I, MC_J + 1])])), dtype=np.float64))
            img_3 = get_last2d(np.array(Image.open(join(_path_name, onlyfiles[int(_M_crt[MC_I + 1, MC_J + 1])])), dtype=np.float64))

        master = np.zeros([2 * R_i, 2 * R_j])

        master[0:R_i, 0:R_j] = img_0
        master[R_i:2 * R_i, 0:R_j] = img_1
        master[0:R_i, R_j:2 * R_j] = img_2
        master[R_i:2 * R_i, R_j:2 * R_j] = img_3

        plt.imshow(master)
        plt.show()

    return _M_crt # pd.DataFrame(_M_crt)

    onlyfiles = [f for f in listdir(_path_name) if isfile(join(_path_name, f)) and join(_path_name, f).endswith('.nd2')]


    _shape = np.array(nd2_opener(join(_path_name, onlyfiles[0])))

def Plot_Point_Mapping(_p_10X, _p_40X, _Tile_List_10X, _Tile_List_40X, _path_10X, _path_40X, _channel=0):

    def get_last3d(data):
        if data.ndim <= 3:
            return data
        slc = [0] * (data.ndim - 3)
        slc += [slice(None), slice(None)]
        return data[slc]

    _p_10X = np.array(_p_10X)
    _p_40X = np.array(_p_40X)

    for p in tqdm(range(len(_p_10X))):

        _q_10X = _p_10X[p, :]
        _q_40X = _p_40X[p, :]

        tile_name_10X = _Tile_List_10X['name'].iloc[_q_10X[0]]
        tile_name_40X = _Tile_List_40X['name'].iloc[_q_40X[0]]

        img_10X = get_last3d(np.array(nd2_opener(join(_path_10X, tile_name_10X))))[_channel]
        img_40X = get_last3d(np.array(nd2_opener(join(_path_40X, tile_name_40X))))[_channel]

        plt.subplot(1, 2, 1)
        plt.imshow(img_10X)
        plt.scatter(_q_10X[2], _q_10X[1], s=4, c='red')

        plt.subplot(1, 2, 2)
        plt.imshow(img_40X)
        plt.scatter(_q_40X[2], _q_40X[1], s=4, c='red')

        plt.show()