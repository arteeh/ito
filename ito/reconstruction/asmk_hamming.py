"""ASMK's `hamming` module in NumPy (upstream ships it as Cython, asmk/cython/hamming.pyx).

Same API and results: rows binarised by `value > threshold`, packed most significant bit first
into uint32 words (a last partial word padded with zero bits), and normalised Hamming distances.
Installed as `asmk.hamming` before the vendored asmk is imported (mast3r_runtime.prepare_full).
"""

import math

import numpy as np


def binarize_and_pack_2D(arr, threshold=0):
    arr = np.asarray(arr, np.float32)
    words = math.ceil(arr.shape[1] / 32)
    bits = np.zeros((arr.shape[0], words * 32), np.uint8)
    bits[:, : arr.shape[1]] = arr > threshold
    return np.packbits(bits, axis=1, bitorder="big").view(">u4").astype(np.uint32)


def binarize_and_pack(arr, threshold=0):
    return binarize_and_pack_2D(np.asarray(arr, np.float32)[None], threshold)[0]


def hamming_cdist_packed(arr1, arr2, normalization=0):
    arr1, arr2 = np.asarray(arr1, np.uint32), np.asarray(arr2, np.uint32)
    assert arr1.shape[1] == arr2.shape[1]
    normalization = normalization or arr1.shape[1] * 32
    counts = np.bitwise_count(arr1[:, None, :] ^ arr2[None, :, :]).sum(-1, dtype=np.int64)
    return (counts / np.float32(normalization)).astype(np.float32)


def hamming_dist_packed(n1, n2, normalization=0):
    return float(
        hamming_cdist_packed(np.asarray(n1)[None], np.asarray(n2)[None], normalization)[0, 0]
    )
