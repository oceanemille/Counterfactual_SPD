import torch
import pymanopt
import numpy as np
from dataset_config import DATASET_CONFIG
from pyriemann.classification import MDM, TSClassifier
from pyriemann.utils.mean import mean_riemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from moabb.paradigms import FilterBankMotorImagery
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from sklearn.model_selection import train_test_split

from pyriemann.tangentspace import TangentSpace
from functools import partial
import math
import pickle
import matplotlib.pyplot as plt

np.random.seed(3)

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys()),
        required=True,
        help="Dataset à utiliser : BNCI2014_001, Dreyer2023C, Beetl2021_A"
    )
    parser.add_argument(
        "--clf",
        type=str,
        choices=["MDM", "TSClassifier"],
        required=True,
        help="Classifieur à utiliser : MDM, TSClassifier"
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=10,
        help="Nombre de splits (défaut: 10)"
    )

    return parser.parse_args()


def proj_on_spd(M, eps=1e-10):
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T


def margin(matrix_mod, clf, baseline_class):
    d = clf.transform(matrix_mod[None, :, :])[0]      # distances aux 2 moyennes de classe
    classes = list(clf.classes_)
    idx_own = classes.index(baseline_class)
    idx_opp = 1 - idx_own
    return d[idx_own] - d[idx_opp]   # < 0 : encore classe d'origine ; >= 0 : bascule



def first_flip(hist):
    idx = np.where(np.asarray(hist) >= 0)[0]
    return int(idx[0]) if len(idx) else np.nan   # indice = nombre de canaux modifiés


def compute_margins_covs_vars(matrix, clf, var_left, var_right,
                              max_features=None, criterion="variance_gain", stop_on_flip=False):
    """
    Trois trajectoires, appliquées aux mêmes canaux sélectionnés :
      - decorr      : décorrélation seule (covariances à zéro)
      - var         : variance seule (covariances intactes)
      - both        : décorrélation + variance poussée vers la cible

    criterion : "variance_gain" -> canal qui maximise l'effet marginal de la variance
                                   (en présence de décorrélation)
                "margin"        -> canal qui maximise la marge de 'both'
    """
    assert criterion in ("variance_gain", "margin")

    n_feat = matrix.shape[0]
    max_features = max_features or n_feat

    baseline_class = clf.predict(matrix[None, :, :])[0]
    target_var = var_left if baseline_class == "right_hand" else var_right

    matrix_decorr = matrix.copy()
    matrix_var = matrix.copy()
    matrix_both = matrix.copy()

    selected = []
    remaining = list(range(n_feat))

    m0 = margin(matrix, clf, baseline_class)
    m_decorr = m_var = m_both = m0
    gap = 0.0   # écart cumulé both - decorr (0 au départ)

    margin_history = [m_both]                 # décorrélation + variance
    reference_margin_history = [m_decorr]     # décorrélation seule
    variance_only_margin_history = [m_var]    # variance seule
    cumulative_gap_history = [gap]            # both - decorr
    variance_contrib_history = []             # gain marginal de la variance, par étape
    interaction_history = [0.0]               # interaction = both - decorr - var + m0

    while remaining and len(selected) < max_features and (not stop_on_flip or m_both < 0):
        best, best_score = None, -np.inf

        for feat in remaining:
            cand_decorr = matrix_decorr.copy()
            cand_decorr[feat, :] = 0
            cand_decorr[:, feat] = 0
            cand_decorr[feat, feat] = matrix[feat, feat]
            c_decorr = margin(cand_decorr, clf, baseline_class)

            cand_var = matrix_var.copy()
            cand_var[feat, feat] = target_var[feat]
            c_var = proj_on_spd(cand_var)
            c_var = margin(c_var, clf, baseline_class)

            cand_both = matrix_both.copy()
            cand_both[feat, :] = 0
            cand_both[:, feat] = 0
            cand_both[feat, feat] = target_var[feat]
            c_both = margin(cand_both, clf, baseline_class)

            cum_gap = c_both - c_decorr          # écart cumulé après ajout du canal
            marginal_gain = cum_gap - gap        # apport de ce seul canal

            score = marginal_gain if criterion == "variance_gain" else c_both

            if score > best_score:
                best_score = score
                best = dict(feat=feat,
                            decorr=cand_decorr, var=cand_var, both=cand_both,
                            m_decorr=c_decorr, m_var=c_var, m_both=c_both,
                            cum_gap=cum_gap, marginal_gain=marginal_gain)

        selected.append(best["feat"])
        remaining.remove(best["feat"])

        matrix_decorr, matrix_var, matrix_both = best["decorr"], best["var"], best["both"]
        m_decorr, m_var, m_both = best["m_decorr"], best["m_var"], best["m_both"]
        gap = best["cum_gap"]

        margin_history.append(m_both)
        reference_margin_history.append(m_decorr)
        variance_only_margin_history.append(m_var)
        cumulative_gap_history.append(gap)
        variance_contrib_history.append(best["marginal_gain"])
        interaction_history.append(m_both - m_decorr - m_var + m0)

    flipped = m_both >= 0
    n_both = first_flip(margin_history)
    return {
        "selected_features": selected,
        "flipped": not np.isnan(n_both),
        "n_features_needed": n_both,
        "n_features_needed_decorr": first_flip(reference_margin_history),
        "n_features_needed_var": first_flip(variance_only_margin_history),
        "margin_history": margin_history,
        "reference_margin_history": reference_margin_history,
        "variance_only_margin_history": variance_only_margin_history,
        "cumulative_gap_history": cumulative_gap_history,
        "variance_contrib_history": variance_contrib_history,
        "interaction_history": interaction_history,
    }


def isolate_var_cov_effects(matrix, clf, var_left, var_right):
    """
    Décompose l'effet d'une modification de matrice de covariance
    en :
        - effet covariance
        - effet variance
        - interaction variance/covariance

    Les quatre matrices comparées :

    A : matrice originale
    B : covariance supprimée, variance originale
    C : covariance originale, variance modifiée
    D : covariance supprimée, variance modifiée
    
    On effectue ces changements sur tous les capteurs à la fois
    """

    baseline_class = clf.predict(matrix[None, :, :])[0]

    target_var = var_left if baseline_class == "right_hand" else var_right

    A = matrix.copy()

    B = matrix.copy()

    B = np.diag(np.diag(B))

    C = matrix.copy()

    np.fill_diagonal(C, target_var)

    C = proj_on_spd(C)

    D = B.copy()

    np.fill_diagonal(D, target_var)

    mA = margin(A, clf, baseline_class)
    mB = margin(B, clf, baseline_class)
    mC = margin(C, clf, baseline_class)
    mD = margin(D, clf, baseline_class)

    covariance_effect = mB - mA
    variance_effect = mC - mA
    interaction_effect = mD - mB - mC + mA


    total_effect = mD - mA

    if total_effect != 0:
        covariance_ratio = covariance_effect / total_effect
        variance_ratio = variance_effect / total_effect
        interaction_ratio = interaction_effect / total_effect
    else:
        covariance_ratio = np.nan
        variance_ratio = np.nan
        interaction_ratio = np.nan


    return {
        "baseline_class": baseline_class,

        "margins": {
            "original": mA,
            "covariance_only": mB,
            "variance_only": mC,
            "covariance_plus_variance": mD,
        },

        "effects": {
            "covariance": covariance_effect,
            "variance": variance_effect,
            "interaction": interaction_effect,
            "total": total_effect,
        },

        "ratios": {
            "covariance_fraction": covariance_ratio,
            "variance_fraction": variance_ratio,
            "interaction_fraction": interaction_ratio,
        },

        "flipped": {
            "original": mA >= 0,
            "covariance_only": mB >= 0,
            "variance_only": mC >= 0,
            "combined": mD >= 0,
        }
    }


def isolate_variance_covariance_flipped_effects(matrix, clf, var_left, var_right, matrix_left, matrix_right):
    """
    Décompose l'effet d'une modification de matrice de covariance
    en :
        - effet covariance
        - effet variance
        - interaction variance/covariance

    Les quatre matrices comparées :

    A : matrice originale
    B : covariance opposée, variance originale
    C : covariance originale, variance modifiée
    D : covariance opposée, variance modifiée
    """

    baseline_class = clf.predict(matrix[None, :, :])[0]

    target_var = var_left if baseline_class == "right_hand" else var_right
    target_matrix = matrix_left if baseline_class == "right_hand" else matrix_right

    A = matrix.copy()

    B = matrix.copy()

    B = target_matrix.copy() - np.diag(np.diag(target_matrix.copy())) + np.diag(np.diag(matrix.copy()))


    C = matrix.copy()

    np.fill_diagonal(C, target_var)

    D = target_matrix.copy()

    mA = margin(A, clf, baseline_class)
    mB = margin(B, clf, baseline_class)
    mC = margin(C, clf, baseline_class)
    mD = margin(D, clf, baseline_class)

    covariance_effect = mB - mA
    variance_effect = mC - mA
    interaction_effect = mD - mB - mC + mA


    total_effect = mD - mA

    if total_effect != 0:
        covariance_ratio = covariance_effect / total_effect
        variance_ratio = variance_effect / total_effect
        interaction_ratio = interaction_effect / total_effect
    else:
        covariance_ratio = np.nan
        variance_ratio = np.nan
        interaction_ratio = np.nan


    return {
        "baseline_class": baseline_class,

        "margins": {
            "original": mA,
            "covariance_only": mB,
            "variance_only": mC,
            "covariance_plus_variance": mD,
        },

        "effects": {
            "covariance": covariance_effect,
            "variance": variance_effect,
            "interaction": interaction_effect,
            "total": total_effect,
        },

        "ratios": {
            "covariance_fraction": covariance_ratio,
            "variance_fraction": variance_ratio,
            "interaction_fraction": interaction_ratio,
        },

        "flipped": {
            "original": mA >= 0,
            "covariance_only": mB >= 0,
            "variance_only": mC >= 0,
            "combined": mD >= 0,
        }
    }
