from counterfactual_var_covs_functions import compute_margins_covs_vars

from dataset_config import DATASET_CONFIG
import numpy as np 
import moabb
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from moabb.paradigms import FilterBankMotorImagery

import mne
import math
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.pipeline import Pipeline
from pyriemann.classification import MDM, TSClassifier
from pyriemann.estimation import Covariances
from sklearn.model_selection import train_test_split

import os
import pickle

import argparse
np.random.seed(3)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys()),
        required=True,
        help="Dataset à utiliser : BNCI2014_001, Dreyer2023C, Beetl2021_A ou pollution"
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
        default = 1,
        help = "Nombre de splits. Warning, si le nombre de splits est supérieur à 1, des valeurs locales de différents traisl seront moyennés ensemble. Risque de perte d'interprétabilité locale."
    )

    parser.add_argument(
        "--savefile",
        type=str,
        default = None,
        help = "Le chemin où tu veux sauvegarder ton fichier"
    )


    return parser.parse_args()


def class_diag_mean(C, y, label):
    return np.diagonal(C[y == label], axis1=1, axis2=2).mean(axis=0)



def summarize_subjects(all_subjects_results):
    keys = {
        "both":   "n_features_needed",
        "decorr": "n_features_needed_decorr",
        "var":    "n_features_needed_var",
    }
    rows = []
    for subject, splits in all_subjects_results.items():
        row = {"subject": subject}
        for name, key in keys.items():
            # tous les essais de tous les splits, mis à plat
            n = np.array([res[key] for split in splits for res in split], dtype=float)
            flipped = ~np.isnan(n)
            row[f"pct_flipped_{name}"] = 100 * flipped.mean()
            row[f"mean_n_{name}"] = np.nanmean(n) if flipped.any() else np.nan
            row[f"median_n_{name}"] = np.nanmedian(n) if flipped.any() else np.nan
            row[f"std_n_{name}"] = np.nanstd(n) if flipped.any() else np.nan
        row["n_trials"] = len(n)
        rows.append(row)
    return pd.DataFrame(rows).set_index("subject")



if __name__ == "__main__":
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    dataset = cfg["dataset"]
    paradigm = FilterBankMotorImagery(
        filters=[[7, 35]], events={"left_hand": 1, "right_hand": 2}
    )
    SENSORS = cfg["sensors"]

    OUT_DIR = args.savefile or f"results/exp1/{args.dataset}/{args.clf}/"
    os.makedirs(OUT_DIR, exist_ok=True)

    CLASSIFIERS = {"MDM": MDM, "TSClassifier": TSClassifier}
    N_SPLITS = args.n_splits
    visualize_only=False

    if visualize_only == True: 

        all_subjects_results = {}
        for subject in dataset.subject_list:
            all_subjects_results[subject] = []
            X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])

            for _ in range(N_SPLITS):
                classifier = CLASSIFIERS[args.clf]()  
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=None
                )
                cov = Covariances()
                C_train = cov.fit_transform(X_train)
                C_test = cov.transform(X_test)

                classifier.fit(C_train, y_train)
                score = classifier.score(C_test, y_test)
                print(f"Subject {subject} - Score: {score:.2f}")

                # Cibles estimées sur le train, vecteurs (n_ch,)
                var_left = class_diag_mean(C_train, y_train, "left_hand")
                var_right = class_diag_mean(C_train, y_train, "right_hand")

                split_results = []
                for matrix in C_test:       
                    res = compute_margins_covs_vars(
                        matrix, classifier, var_left, var_right, criterion="margin"
                    )
                    split_results.append(res)
                all_subjects_results[subject].append(split_results)

        with open(os.path.join(OUT_DIR, f"all_subjects_results_{args.dataset}_{args.clf}.pkl"), "wb") as f:
            pickle.dump(all_subjects_results, f)

    all_subjects_results = pickle.load(open(os.path.join(OUT_DIR, f"all_subjects_results_{args.dataset}_{args.clf}.pkl"), "rb"))

    df = summarize_subjects(all_subjects_results)
    print(df.round(2))
    print(df.mean().round(2))   # moyenne sur les sujets
    df.to_csv("statistics.xlsx")

    for subject in dataset.subject_list:
        res = all_subjects_results[subject][0][2]   # premier split, premier essai
        print(f"Subject {subject}: {res}")
        plt.plot(res["margin_history"], label="décorrélation + variance")
        plt.plot(res["reference_margin_history"], label="décorrélation seule")
        plt.plot(res["variance_only_margin_history"], label="variance seule")
        plt.plot(res["interaction_history"], label="interaction variance/covariance")
        plt.axhline(0, color="k", ls="--", lw=0.8)
        plt.xlabel("Nombre de canaux modifiés"); plt.ylabel("Marge"); plt.legend()
        plt.savefig(os.path.join(OUT_DIR, f"subject_{subject}_trial{2}_margins.png"))
        plt.show()
    
