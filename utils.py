import os
from datetime import datetime

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.profiler import record_function
from inspect import isfunction
from copy import deepcopy

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def match_shape(values, broadcast_array, tensor_format="pt"):
    values = values.flatten()

    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)

    return values


def clip(tensor, min_value=None, max_value=None):
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, min_value, max_value)
    elif isinstance(tensor, torch.Tensor):
        return torch.clamp(tensor, min_value, max_value)

    raise ValueError("Tensor format is not valid is not valid - " \
        f"should be numpy array or torch tensor. Got {type(tensor)}.")

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def ohe_to_categories(ohe, K):
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i]:indices[i+1]].argmax(dim=1))
    return torch.stack(res, dim=1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def log_categorical(log_x_start, log_prob, mask):
    out = (log_x_start.exp() * log_prob) * mask
    return out.sum(dim=1)

def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, int(i)], num_classes[i]))
 
    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot

def log_sum_exp_by_classes(x, slices):
    device = x.device
    res = torch.zeros_like(x)
    for ixs in slices:
        res[:, ixs] = torch.logsumexp(x[:, ixs], dim=1, keepdim=True)

    assert x.size() == res.size()

    return res

@torch.jit.script
def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m

@torch.jit.script
def sliced_logsumexp(x, slices):
    lse = torch.logcumsumexp(
        torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float('inf')),
        dim=-1)

    slice_starts = slices[:-1]
    slice_ends = slices[1:]

    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    slice_lse_repeated = torch.repeat_interleave(
        slice_lse,
        slice_ends - slice_starts, 
        dim=-1
    )
    return slice_lse_repeated

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

class FoundNANsError(BaseException):
    """Found NANs during sampling"""
    def __init__(self, message='Found NANs during sampling.'):
        super(FoundNANsError, self).__init__(message)

def m(v):
    nonnull = v[np.isnan(v) == False]
    return np.mean(nonnull)

def std(v):
    nonnull = v[np.isnan(v) == False]
    return np.std(nonnull)

def infill_null(v):
    v[np.isnan(v)] = 0
    return v

def remove_outliers(lists):
    b = np.ones(lists[0].shape)

    for l in lists:
        q1 = np.nanquantile(l,0.25)
        q3 = np.nanquantile(l,0.75)

        iqr = q3 - q1
        lower = q1 - 1.5*iqr
        upper = q3 + 1.5*iqr

        b = np.logical_and(b, np.logical_or(l > lower , np.isnan(l)))
        b = np.logical_and(b, np.logical_or(l < upper , np.isnan(l)))
    
    return b

def norm(v):
    nonnull = v[np.isnan(v) == False]
    max = np.nanmax(nonnull)
    min = np.nanmin(nonnull)

    return 2*((v - min) / (max - min))-1

def get_local_gaussian(ys, numbins=50):
    max = np.nanmax(ys)
    min = np.nanmin(ys)
    bins = np.linspace(min, max, num=numbins)
    s_ys = np.array(sorted(ys, reverse=True))

    d, m, s = [], [], []
    for i in range(numbins-1):
        low = bins[i]
        high = bins[i+1]
        tbool = np.logical_and(low<=s_ys, s_ys<=high)
        data = s_ys[tbool]
        d.append(len(data))
        m.append(data.mean() if not np.isnan(data.mean()) else low)
        s.append(data.std() if not np.isnan(data.std()) else 0.01)
    d = np.array(d) / sum(d)

    return d, np.array(m), np.array(s)

def convert_categorical(ys, numbins=50):
    max = np.nanmax(ys)
    min = np.nanmin(ys)
    bins = np.linspace(min, max, num=numbins)
    s = np.array(sorted(enumerate(ys), key=lambda x:x[1]))
    s_inds = s[:,0].astype(int)
    s_ys = s[:,1]
    
    
    nys = deepcopy(ys)
    for i in range(numbins-1):
        low = bins[i]
        high = bins[i+1]
        tbool = np.logical_and(low<=s_ys, s_ys<=high)
        
        if sum(tbool) > 1:
            nys[np.array(s_inds[tbool])] = i+1

    print(i+1)
    nys[np.isnan(nys)] = numbins
    return nys

def sample_local_gaussian(v, numbins=50, infilling=None):
    d,m,s = get_local_gaussian(v, numbins=numbins)
    num = sum(np.isnan(v))

    samples = np.random.choice(numbins-1, num, p=d)
    rand_n = np.random.randn(num)

    adjust = m[samples] + 1.2 * rand_n *s[samples]
    
    # override 
    #adjust = np.zeros(adjust.shape)
    
    if infilling == "zeros":
        adjust = np.zeros(adjust.shape)

    if infilling == "uniform":
        adjust = np.random.uniform(low=-1.0, high=1.0, size=adjust.shape)

    if infilling == "gaussian":
        adjust = np.random.normal(size=adjust.shape)

    v[np.isnan(v)] = adjust
    return v, (d,m,s)


def sample_noise(size, dmss, numbins=50, noise=None):
    if noise == "uniform":
        return np.random.uniform(low=-1.0, high=1.0, size=size)
    
    if noise == "gaussian":
        return np.random.normal(size=size)

    vs = []
    for d,m,s in dmss:
        samples = np.random.choice(numbins-1, size[0], p=d)
        rand_n = np.random.randn(size[0])
        vs.append(m[samples] + 1.2 * rand_n *s[samples])
    return np.stack(vs, axis=-1)

def unison_shuffled_copies(a, b):
    #assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return np.array(a)[p], np.array(b)[p]

def categorical_norm(ys, numbins=50):
    max = np.nanmax(ys)
    min = np.nanmin(ys)
    bins = np.linspace(min, max, num=numbins)
    s = np.array(sorted(enumerate(ys), key=lambda x:x[1]))
    s_inds = s[:,0].astype(int)
    s_ys = s[:,1]
    
    
    nys = deepcopy(ys)
    nnys = deepcopy(ys)
    for i in range(numbins-1):
        low = bins[i]
        high = bins[i+1]

        tbool = np.logical_and(low<=s_ys, s_ys<=high)
        
        if sum(tbool) > 1:
            nys[np.array(s_inds[tbool])] = i+1

            print(low, high)
            normalized = (nnys[np.array(s_inds[tbool])] - low) / (high-low)
            print(normalized)
            nnys[np.array(s_inds[tbool])] = normalized * 2 - 1

    nys[np.isnan(nys)] = numbins
    nnys[np.isnan(nnys)] = np.random.uniform(low=-1.0, high=1.0, size=sum(np.isnan(nnys)))
    return nys, nnys