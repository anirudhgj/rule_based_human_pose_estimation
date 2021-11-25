#17j

import multiprocessing
from contextlib import closing
from multiprocessing import Pool
import numpy as np

from scipy.io import loadmat
from transforms3d.euler import EulerFuncs

from . import prior_sk_data as psk_data

sk_data = psk_data

# default_coordinate_transform = 'euler'
default_coordinate_transform = 'azimuth'

euler_fx = EulerFuncs('rxyz')

eps = 0.00000005


def set_skeleton_data(skeleton_data):
    global sk_data
    sk_data = skeleton_data


def set_angle_convention(angles):
    global default_coordinate_transform
    default_coordinate_transform = angles


def unit_norm(v, axis=0):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + 1e-9)


def mat2euler(T):
    return euler_fx.mat2euler(T)


def euler2mat(alpha, beta, gamma):
    return euler_fx.euler2mat(alpha, beta, gamma)


def get_Rx(theta):
    return np.matrix([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])


def get_Ry(theta):
    return np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])


def get_Rz(theta):
    return np.matrix([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def getT(alpha, beta, gamma):
    T_a = get_Rx(alpha)
    T_b = get_Ry(beta)
    T_g = get_Rz(gamma)

    return T_a * T_b * T_g


def gram_schmidt(A):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise Exception('GramSchmidt: not valid shape %d' % A.shape)
    n = A.shape[0]

    B = np.zeros((n, n))

    B[0] = unit_norm(A[0])

    for i in range(1, n):
        v = A[i]
        U = B[:i]
        pc = np.dot(U, v)
        p = np.dot(U.T, pc)
        v = v - p
        v = unit_norm(v)
        if np.linalg.norm(v) < eps:
            raise Exception('GramSchmidt: 0 norm exception : %s' % str(v))
        B[i] = v
    return B


def get_eulerian_local_coordinates(skeleton, sk_data=sk_data):
    right_hip = sk_data.get_joint_index('right_hip')
    left_hip = sk_data.get_joint_index('left_hip')
    neck = sk_data.get_joint_index('neck')

    r = skeleton[right_hip]
    l = skeleton[left_hip]
    n = skeleton[neck]

    m = 0.5 * (r + l)
    z_ = unit_norm(n - m)
    y_ = unit_norm(np.cross(l - r, n - r))
    x_ = np.cross(y_, z_)

    return x_, y_, z_


def get_transform_matrices(skeleton, sk_data=sk_data):
    right_hip = sk_data.get_joint_index('right_hip')
    left_hip = sk_data.get_joint_index('left_hip')
    neck = sk_data.get_joint_index('neck')

    r = skeleton[:, right_hip:right_hip + 1]
    l = skeleton[:, left_hip:left_hip + 1]
    n = skeleton[:, neck:neck + 1]

    m = 0.5 * (r + l)
    z_ = unit_norm(n - m, axis=-1)
    y_ = unit_norm(np.cross(l - r, n - r), axis=-1)
    x_ = np.cross(y_, z_)

    return np.concatenate([x_, y_, z_], axis=1)


def get_azimuthal_local_coordinates(skeleton, sk_data=sk_data):
#     right_hip = sk_data.get_joint_index('right_hip')
#     left_hip = sk_data.get_joint_index('left_hip')
    right_hip=9
    left_hip=12

    r = skeleton[right_hip]
    l = skeleton[left_hip]

    z_ = unit_norm(np.array([0., 0., 1.]))
    x_ = unit_norm((r - l) * np.array([1., 1., 0.]))
    y_ = np.cross(z_, x_)
    return x_, y_, z_


def get_local_coordinates(skeleton, sk_data=sk_data, system=default_coordinate_transform):
    if system == 'euler':
        return get_eulerian_local_coordinates(skeleton, sk_data)
    if system == 'azimuth':
        return get_azimuthal_local_coordinates(skeleton, sk_data)
    raise NotImplementedError('Invalid system: %s' % system)


def get_global_angles_and_transform(skeleton, sk_data=sk_data):
    x_, y_, z_ = get_local_coordinates(skeleton, sk_data, default_coordinate_transform)

    B = np.matrix([x_, y_, z_])

    alpha, beta, gamma = mat2euler(B)

    return alpha, beta, gamma, np.array(skeleton * B.T)


def get_global_angles_and_transform_batch(skeleton_batch, sk_data=sk_data):
    # [B, 3, 3]
    transform_mats = get_transform_matrices(skeleton_batch, sk_data)

    # [B, 17, 3]
    view_norm = np.matmul(skeleton_batch, transform_mats.transpose([0, 2, 1]))

    angles = np.array([mat2euler(mat) for mat in transform_mats])

    return angles, view_norm


def get_global_joints(local_joints, alpha, beta, gamma):
    B = euler2mat(alpha, beta, gamma)

    global_joints = np.matrix(local_joints) * B

    return global_joints


def get_global_joints_batch(local_joints_batch, angles):
    # [B, 3, 3]
    transform_mats = np.array([euler2mat(*a) for a in angles])

    global_joints_batch = np.matmul(local_joints_batch, transform_mats)

    return global_joints_batch


def prior_get_normal(x1, a, x):
    if np.linalg.norm(x - a) < eps or np.linalg.norm(x + a) < eps:
        n = np.cross(x, x1)
        flag = True
    else:
        n = np.cross(a, x)
        flag = True
    return unit_norm(n), flag


def prior_global2local(skeleton):
    a = sk_data.arbitary_vec
    D = sk_data.D
    parents = sk_data.limb_parents

    dS = skeleton[parents] - skeleton

    shldr = dS[sk_data.joint_map['left_shoulder']] - dS[sk_data.joint_map['right_shoulder']]
    hip = dS[sk_data.joint_map['left_hip']] - dS[sk_data.joint_map['right_hip']]

    dSl = dS.copy()

    for i in sk_data.child:

        if i in sk_data.upper_limbs:
            u = shldr if i in sk_data.neck_joints else hip
            u = unit_norm(u)
            v = unit_norm(dS[1])
        else:
            u = dS[parents[i]]
            u = unit_norm(u)

            x1 = np.dot(R.T, D[parents[i]])
            x2 = np.dot(R.T, a)
            x3 = u
            v, _ = prior_get_normal(x1, x2, x3)

        w = np.cross(u, v)
        w = unit_norm(w)
        R = gram_schmidt(np.array([u, v, w]))
        dSl[i] = np.dot(R, dS[i])

    return dSl


def prior_estimate_absolute_positions(dS):
    S = np.zeros(dS.shape)
    limb_order = sk_data.limb_order
    parent = sk_data.limb_parents
    S[limb_order[0]] = np.zeros((3,))

    for i in limb_order[1:]:
        S[i] = S[parent[i]] - dS[i]
    return S


def prior_local2global(dS_local):
    a = sk_data.arbitary_vec
    D = sk_data.D
    parents = sk_data.limb_parents

    shldr = dS_local[sk_data.joint_map['left_shoulder']] - dS_local[sk_data.joint_map['right_shoulder']]
    hip = dS_local[sk_data.joint_map['left_hip']] - dS_local[sk_data.joint_map['right_hip']]

    dS = np.ones(dS_local.shape) * np.nan
    dS[sk_data.torso_joints] = dS_local[sk_data.torso_joints]

    for i in sk_data.child:

        if i in sk_data.upper_limbs:
            u = shldr if i in sk_data.neck_joints else hip
            u = unit_norm(u)
            v = unit_norm(dS[1])

        else:
            u = dS[parents[i]]
            u = unit_norm(u)

            x1 = np.dot(R.T, D[parents[i]])
            x2 = np.dot(R.T, a)
            x3 = u
            v, _ = prior_get_normal(x1, x2, x3)

        w = np.cross(u, v)
        w = unit_norm(w)
        try:

            R = gram_schmidt(np.array([u, v, w]))
            dS[i] = np.dot(R.T, dS_local[i])
        except:
            raise Exception('Error in %d joint: %s %s' % (i, u, v))

    S = prior_estimate_absolute_positions(dS)
    return S


def root_relative_to_local_skeleton(skeleton):
    try:
        if np.all(skeleton == 0.):
            return np.array([0., 0., 0.]), skeleton.copy(), skeleton.copy()
        alpha, beta, gamma, view_norm_skeleton = get_global_angles_and_transform(skeleton)
        local_skeleton = prior_global2local(view_norm_skeleton)
        return np.array([alpha, beta, gamma]), view_norm_skeleton, local_skeleton
    except Exception as ex:
        print ex, 'Error while processing: '
        print skeleton


def root_relative_to_view_norm_skeleton(skeleton):
    try:
        if np.all(skeleton == 0.):
            return np.array([0., 0., 0.]), skeleton.copy(), skeleton.copy()
        alpha, beta, gamma, view_norm_skeleton = get_global_angles_and_transform(skeleton)
        return np.array([alpha, beta, gamma]), view_norm_skeleton
    except Exception as ex:
        print ex, 'Error while processing: '
        print skeleton

def local_to_root_relative_for_norm_ske(skeleton_batch):
    limb_lengths = psk_data.limb_ratios*2
    skeleton_batch = skeleton_batch * limb_lengths[:, np.newaxis]
    sk_batch_view_norm = prior_local2global(skeleton_batch)
    return sk_batch_view_norm


def local_to_root_relative_skeleton(global_angles, local_skeleton):
    if np.all(local_skeleton == 0.):
        return local_skeleton.copy()
    view_norm_skeleton = prior_local2global(local_skeleton)
    root_relative_skeleton = get_global_joints(view_norm_skeleton, *global_angles)
    return root_relative_skeleton


def view_norm_to_root_relative_skeleton(global_angles, view_norm_skeleton):
    if np.all(view_norm_skeleton == 0.):
        return view_norm_skeleton.copy()
    root_relative_skeleton = get_global_joints(view_norm_skeleton, *global_angles)
    return root_relative_skeleton


###########################################################
####### Root Relative to Local Conversion Functions #######

def __root_relative_to_local_skeleton_batch_basic_impl(skeleton_batch):
    # [d1, d2, d3..., num_joints, 3]
    # only operates on last two dimensions, rest will be considered batch
    orig_shape = skeleton_batch.shape
    skeleton_batch = skeleton_batch.reshape(-1, *orig_shape[-2:])
    global_angles_batch = np.zeros((skeleton_batch.shape[0], 3))
    view_norm_skeleton_batch = np.zeros(skeleton_batch.shape)
    local_skeleton_batch = np.zeros(skeleton_batch.shape)

    for i, skeleton in enumerate(skeleton_batch):
        global_angles, view_norm_skeleton, local_skeleton = root_relative_to_local_skeleton(skeleton)
        global_angles_batch[i] = global_angles
        view_norm_skeleton_batch[i] = view_norm_skeleton
        local_skeleton_batch[i] = local_skeleton

    global_angles_batch = global_angles_batch.reshape(*[orig_shape[:-2] + (3,)])
    view_norm_skeleton_batch = view_norm_skeleton_batch.reshape(*orig_shape)
    local_skeleton_batch = local_skeleton_batch.reshape(*orig_shape)

    return global_angles_batch, view_norm_skeleton_batch, local_skeleton_batch


def __root_relative_to_local_skeleton_batch_fast_impl(skeleton_batch):
    # [d1, d2, d3..., num_joints, 3]
    # only operates on last two dimensions, rest will be considered batch
    orig_shape = skeleton_batch.shape
    skeleton_batch = skeleton_batch.reshape(-1, *orig_shape[-2:])
    global_angles_batch = np.zeros((skeleton_batch.shape[0], 3))
    view_norm_skeleton_batch = np.zeros(skeleton_batch.shape)
    local_skeleton_batch = np.zeros(skeleton_batch.shape)

    try:
        num_processes = multiprocessing.cpu_count()
    except:
        num_processes = 6

    with closing(Pool(num_processes)) as pool:
        outputs = pool.map(root_relative_to_local_skeleton, skeleton_batch)
        pool.terminate()

    for i, output in enumerate(outputs):
        global_angles_batch[i] = output[0]
        view_norm_skeleton_batch[i] = output[1]
        local_skeleton_batch[i] = output[2]

    global_angles_batch = global_angles_batch.reshape(*[orig_shape[:-2] + (3,)])
    view_norm_skeleton_batch = view_norm_skeleton_batch.reshape(*orig_shape)
    local_skeleton_batch = local_skeleton_batch.reshape(*orig_shape)

    return global_angles_batch, view_norm_skeleton_batch, local_skeleton_batch


###########################################################
####### Root Relative to View Norm Conversion Functions #######


def __root_relative_to_view_norm_skeleton_batch_fast_impl(skeleton_batch):
    # [d1, d2, d3..., num_joints, 3]
    # only operates on last two dimensions, rest will be considered batch
    orig_shape = skeleton_batch.shape
    skeleton_batch = skeleton_batch.reshape(-1, *orig_shape[-2:])
    global_angles_batch = np.zeros((skeleton_batch.shape[0], 3))
    view_norm_skeleton_batch = np.zeros(skeleton_batch.shape)

    try:
        num_processes = multiprocessing.cpu_count()
    except:
        num_processes = 6

    with closing(Pool(num_processes)) as pool:
        outputs = pool.map(root_relative_to_view_norm_skeleton, skeleton_batch)
        pool.terminate()

    for i, output in enumerate(outputs):
        global_angles_batch[i] = output[0]
        view_norm_skeleton_batch[i] = output[1]

    global_angles_batch = global_angles_batch.reshape(*[orig_shape[:-2] + (3,)])
    view_norm_skeleton_batch = view_norm_skeleton_batch.reshape(*orig_shape)

    return global_angles_batch, view_norm_skeleton_batch


def __root_relative_to_view_norm_skeleton_batch_basic_impl(skeleton_batch):
    # [d1, d2, d3..., num_joints, 3]
    # only operates on last two dimensions, rest will be considered batch
    orig_shape = skeleton_batch.shape
    skeleton_batch = skeleton_batch.reshape(-1, *orig_shape[-2:])

    global_angles_batch, view_norm_skeleton_batch = get_global_angles_and_transform_batch(skeleton_batch)

    global_angles_batch = global_angles_batch.reshape(*[orig_shape[:-2] + (3,)])
    view_norm_skeleton_batch = view_norm_skeleton_batch.reshape(*orig_shape)

    return global_angles_batch, view_norm_skeleton_batch


###########################################################
######## Local to View Norm Conversion Functions ##########

def __local_to_view_norm_batch_fast_impl(local_skeleton_batch):
    orig_shape = local_skeleton_batch.shape
    local_skeleton_batch = local_skeleton_batch.reshape(-1, *orig_shape[-2:])

    view_norm_skeleton_batch = np.zeros(local_skeleton_batch.shape)

    try:
        num_processes = multiprocessing.cpu_count()
    except:
        num_processes = 6

    with closing(Pool(num_processes)) as pool:
        outputs = pool.map(prior_local2global, local_skeleton_batch)
        pool.terminate()

    for i, output in enumerate(outputs):
        view_norm_skeleton_batch[i] = output

    view_norm_skeleton_batch = view_norm_skeleton_batch.reshape(*orig_shape)

    return view_norm_skeleton_batch


###########################################################
######## Local to View Norm Conversion Functions ##########

def local_to_view_norm_batch_fast_impl_norm(local_skeleton_batch):
    orig_shape = local_skeleton_batch.shape
    
    limb_lengths = psk_data.limb_ratios*2
    local_skeleton_batch = local_skeleton_batch * limb_lengths[:, np.newaxis]
    
    local_skeleton_batch = local_skeleton_batch.reshape(-1, *orig_shape[-2:])
    
    
    
    view_norm_skeleton_batch = np.zeros(local_skeleton_batch.shape)

    try:
        num_processes = multiprocessing.cpu_count()
    except:
        num_processes = 6

    with closing(Pool(num_processes)) as pool:
        outputs = pool.map(prior_local2global, local_skeleton_batch)
        pool.terminate()

    for i, output in enumerate(outputs):
        view_norm_skeleton_batch[i] = output

    view_norm_skeleton_batch = view_norm_skeleton_batch.reshape(*orig_shape)

    return view_norm_skeleton_batch


###########################################################
######## View Norm to Local Conversion Functions ##########

def __view_norm_to_local_batch_fast_impl(view_norm_skeleton_batch):
    orig_shape = view_norm_skeleton_batch.shape
    view_norm_skeleton_batch = view_norm_skeleton_batch.reshape(-1, *orig_shape[-2:])

    local_skeleton_batch = np.zeros(view_norm_skeleton_batch.shape)

    try:
        num_processes = multiprocessing.cpu_count()
    except:
        num_processes = 6

    with closing(Pool(num_processes)) as pool:
        outputs = pool.map(prior_global2local, view_norm_skeleton_batch)
        pool.terminate()

    for i, output in enumerate(outputs):
        local_skeleton_batch[i] = output

    local_skeleton_batch = local_skeleton_batch.reshape(*orig_shape)

    return local_skeleton_batch


###########################################################
######## View Norm to Root Relative Conversion Functions ##########
def __get_root_relative_from_view_norm_skeleton_fast(args):
    global_angles, view_norm_skeleton = args

    if np.all(view_norm_skeleton == 0.):
        return view_norm_skeleton.copy()

    root_relative_skeleton = get_global_joints(view_norm_skeleton, *global_angles)
    return root_relative_skeleton


def __view_norm_to_root_relative_skeleton_batch_fast_impl(global_angles_batch, view_norm_skeleton_batch):
    global_angles_orig_shape = global_angles_batch.shape
    skeleton_batch_orig_shape = view_norm_skeleton_batch.shape

    global_angles_batch = global_angles_batch.reshape(-1, *global_angles_orig_shape[-1:])
    view_norm_skeleton_batch = view_norm_skeleton_batch.reshape(-1, *skeleton_batch_orig_shape[-2:])

    global_skeleton_batch = np.zeros(view_norm_skeleton_batch.shape)

    try:
        num_processes = multiprocessing.cpu_count()
    except:
        num_processes = 6

    with closing(Pool(num_processes)) as pool:
        outputs = pool.map(__get_root_relative_from_view_norm_skeleton_fast, zip(global_angles_batch, view_norm_skeleton_batch))
        pool.terminate()

    for i, output in enumerate(outputs):
        global_skeleton_batch[i] = output

    global_skeleton_batch = global_skeleton_batch.reshape(*skeleton_batch_orig_shape)

    return global_skeleton_batch


def __view_norm_to_root_relative_skeleton_batch_basic_impl(global_angles_batch, view_norm_skeleton_batch):
    global_angles_orig_shape = global_angles_batch.shape
    skeleton_batch_orig_shape = view_norm_skeleton_batch.shape

    global_angles_batch = global_angles_batch.reshape(-1, *global_angles_orig_shape[-1:])
    view_norm_skeleton_batch = view_norm_skeleton_batch.reshape(-1, *skeleton_batch_orig_shape[-2:])

    global_skeleton_batch = get_global_joints_batch(view_norm_skeleton_batch, global_angles_batch)

    global_skeleton_batch = global_skeleton_batch.reshape(*skeleton_batch_orig_shape)

    return global_skeleton_batch


###########################################################
####### Local to Root Relative Conversion Functions #######

def __get_root_relative_global_skeleton_fast(args):
    global_angles, local_skeleton = args

    if np.all(local_skeleton == 0.):
        return local_skeleton.copy()

    view_norm_skeleton = prior_local2global(local_skeleton)
    root_relative_skeleton = get_global_joints(view_norm_skeleton, *global_angles)
    return root_relative_skeleton


def __local_to_root_relative_skeleton_batch_basic_impl(global_angles_batch, local_skeleton_batch):
    global_angles_orig_shape = global_angles_batch.shape
    skeleton_batch_orig_shape = local_skeleton_batch.shape

    global_angles_batch = global_angles_batch.reshape(-1, *global_angles_orig_shape[-1:])
    local_skeleton_batch = local_skeleton_batch.reshape(-1, *skeleton_batch_orig_shape[-2:])

    global_skeleton_batch = np.zeros(local_skeleton_batch.shape)

    for i, global_angles in enumerate(global_angles_batch):
        global_skeleton_batch[i] = local_to_root_relative_skeleton(global_angles, local_skeleton_batch[i])

    global_skeleton_batch = global_skeleton_batch.reshape(*skeleton_batch_orig_shape)

    return global_skeleton_batch


def __local_to_root_relative_skeleton_batch_fast_impl(global_angles_batch, local_skeleton_batch):
    global_angles_orig_shape = global_angles_batch.shape
    skeleton_batch_orig_shape = local_skeleton_batch.shape

    global_angles_batch = global_angles_batch.reshape(-1, *global_angles_orig_shape[-1:])
    local_skeleton_batch = local_skeleton_batch.reshape(-1, *skeleton_batch_orig_shape[-2:])

    global_skeleton_batch = np.zeros(local_skeleton_batch.shape)

    try:
        num_processes = multiprocessing.cpu_count()
    except:
        num_processes = 6

    with closing(Pool(num_processes)) as pool:
        outputs = pool.map(__get_root_relative_global_skeleton_fast, zip(global_angles_batch, local_skeleton_batch))
        pool.terminate()

    for i, output in enumerate(outputs):
        global_skeleton_batch[i] = output

    global_skeleton_batch = global_skeleton_batch.reshape(*skeleton_batch_orig_shape)

    return global_skeleton_batch


def local_to_view_norm_skeleton_batch(local_skeleton_batch, fast=True):
    return __local_to_view_norm_batch_fast_impl(local_skeleton_batch)


def view_norm_to_local_skeleton_batch(view_norm_skeleton_batch, fast=True):
    return __view_norm_to_local_batch_fast_impl(view_norm_skeleton_batch)


def root_relative_to_view_norm_skeleton_batch(root_relative_skeleton_batch, fast=False):
    if fast:
        __func = __root_relative_to_view_norm_skeleton_batch_fast_impl
    else:
        __func = __root_relative_to_view_norm_skeleton_batch_basic_impl
    return __func(root_relative_skeleton_batch)


def view_norm_to_root_relative_skeleton_batch(global_angles_batch, view_norm_skeleton_batch, fast=False):
    if fast:
        __func = __view_norm_to_root_relative_skeleton_batch_fast_impl
    else:
        __func = __view_norm_to_root_relative_skeleton_batch_basic_impl
    return __func(global_angles_batch, view_norm_skeleton_batch)


def root_relative_to_local_skeleton_batch(skeleton_batch, fast=True):
    if fast:
        __func = __root_relative_to_local_skeleton_batch_fast_impl
    else:
        __func = __root_relative_to_local_skeleton_batch_basic_impl
    return __func(skeleton_batch)


def local_to_root_relative_skeleton_batch(global_angles_batch, local_skeleton_batch, fast=True):
    if fast:
        __func = __local_to_root_relative_skeleton_batch_fast_impl
    else:
        __func = __local_to_root_relative_skeleton_batch_basic_impl
    return __func(global_angles_batch, local_skeleton_batch)


def joint_angle_crossover(root_relative_batch_aa, root_relative_batch_bb, fast=False):
    global_angles_batch_a, view_norm_skeleton_batch_a = root_relative_to_view_norm_skeleton_batch(root_relative_batch_aa, fast)
    global_angles_batch_b, view_norm_skeleton_batch_b = root_relative_to_view_norm_skeleton_batch(root_relative_batch_bb, fast)

    root_relative_batch_ab = view_norm_to_root_relative_skeleton_batch(global_angles_batch_b, view_norm_skeleton_batch_a, fast)
    root_relative_batch_ba = view_norm_to_root_relative_skeleton_batch(global_angles_batch_a, view_norm_skeleton_batch_b, fast)

    return root_relative_batch_ab, root_relative_batch_ba
