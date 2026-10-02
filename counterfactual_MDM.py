import torch
import pymanopt
import numpy as np
from src.data.dataset_config import DATASET_CONFIG
from pyriemann.classification import MDM, TSClassifier
from pyriemann.utils.mean import mean_riemann
from pyriemann.estimation import Covariances
from pyriemann.utils.geodesic import geodesic
from pyriemann.tangentspace import TangentSpace
from moabb.paradigms import FilterBankMotorImagery
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from sklearn.model_selection import train_test_split

from src.Visualization.topomap import plot_topomap, plot_pannel
from src.Permutation.FeaturePermutationSPD import (proj_on_spd,percent_spd, spd_values, eigenval_pos)
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
        default=1,
        help="Nombre de splits (défaut: 10)"
    )

    
    parser.add_argument(
        "--eigenval_info",
        type=bool,
        default=False,
        help="Si True, le programme affichera des informations sur les valeurs propres des matrices avant projection."
    )

    return parser.parse_args()

def compute_counterfactual(X, y, clf, MA, MB, tol=1e-5, max_iter=50):
    results = []
    colors = []
    for i, x in enumerate(X):
        initial_pred = clf.predict(x[np.newaxis, :])
        target_class = 'right_hand' if initial_pred != 'right_hand' else 'left_hand'
        M = MB if initial_pred != 'right_hand' else MA 
        if initial_pred == y[i]:
            colors.append('blue')
        else:
            colors.append('red')

        lo, hi = 0.0, 1.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            pred = clf.predict(geodesic(x, M, mid)[np.newaxis, :])[0]
            if pred == initial_pred:
                lo = mid  
            else:
                hi = mid  
            if hi - lo < tol:
                break
        if hi < tol :
            print(pred,initial_pred)
            results.append(-1)
        else:
            results.append(hi)  
    return results, colors

if __name__ == "__main__":
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    dataset = cfg["dataset"]
    SENSORS = cfg["sensors"]

    OUT_DIR = f'../../Results/Permutation/{args.clf}/{args.dataset}/With_cue/Modif_covs/Heuristic'

    N_SPLITS = args.n_splits
    SCORE_THRESHOLD = 0.75

    CLASSIFIERS = {
    "MDM": MDM,
    "TSClassifier": TSClassifier,
}

    clf = CLASSIFIERS[args.clf]()

    visu_only = True
    results_per_subjects = []


    if not visu_only:
        paradigm = FilterBankMotorImagery(filters=[(7, 35)], events ={"left_hand": 1, "right_hand": 2})
        X, y, meta = paradigm.get_data(dataset=dataset)
        results_per_subject = []
        colors_per_subjects = []
        results_comp_per_subject = []
        for subject in dataset.subject_list:
            subject_mask = meta.subject == subject
            data = X[subject_mask]
            y_subj = y[subject_mask]

            results_per_split = []
            colors_per_split = []
            results_comp_per_split = []

            for i in range(N_SPLITS):
                X_train, X_test, y_train, y_test = train_test_split(
                    data, y_subj, stratify = y_subj, test_size=0.2, random_state=i
                )
                covs = Covariances()
                C_train = covs.fit_transform(X_train)
                C_test = covs.transform(X_test)
                clf.fit(C_train, y_train)
                MA, MB = clf.covmeans_
                results, colors = compute_counterfactual(C_test, y_test, clf, MA, MB)
                results_per_split.append(results)
                colors_per_split.append(colors)
            results_per_subjects.append(results_per_split)
            colors_per_subjects.append(colors_per_split)

        np.save(f"{OUT_DIR}/counterfactual_results_{args.dataset}_{args.clf}.npy", results_per_subjects)
        np.save(f"{OUT_DIR}/counterfactual_colors_{args.dataset}_{args.clf}.npy", colors_per_subjects)

    results = np.load(f"../../Results/Permutation/{args.clf}/{args.dataset}/With_cue/Modif_covs/Heuristic/counterfactual_results_{args.dataset}_{args.clf}.npy")
    colors = np.load(f"../../Results/Permutation/{args.clf}/{args.dataset}/With_cue/Modif_covs/Heuristic/counterfactual_colors_{args.dataset}_{args.clf}.npy")
    print(results.shape)
    for idx, subject_data in enumerate(results):
        split_data = subject_data[0]
        color_data = colors[idx][0]
        filtered_data = split_data[split_data != -1]
        print((len(split_data) - len(filtered_data)) / len(split_data))
        # 2. Split the data based on your color array
        blue_data = filtered_data[color_data == 'blue']
        red_data = filtered_data[color_data == 'red']

        # 3. Define your custom bins
        custom_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        # 4. Plot both groups together (stacked=True keeps them clean and visible)
        plt.hist(
            [blue_data, red_data], 
            bins=custom_bins, 
            stacked=True, 
            color=['blue', 'red'],
            label=[f"Exemples bien classés", f"Exemples mal classés"]
        )

        plt.legend()
        plt.title(f"Histogram of alpha values where the example changes class for Subject {dataset.subject_list[idx]}")
        plt.xlabel("Alpha")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/Histogram_{args.dataset}_{args.clf}_Subject_{dataset.subject_list[idx]}.pdf", bbox_inches='tight')
        plt.show()