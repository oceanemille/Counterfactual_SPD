## Installation

Linux and Python 3.12 are recommended

```bash
git clone https://github.com/oceanemille/Counterfactual_SPD.git
cd Counterfactual_SPD

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Explanations

`counterfactual_MDM.py` computes an alpha for each trial that corresponds to the minimum proportion of the geodesic from the trial to the mean of the opposite class to has to be reached in order for the class to change. It then produces a histogram per subject that shows the distribution of alphas depending on the accuracy score.

It is best called this way :

```bash
python -m counterfactual_MDM --clf "MDM" --dataset "BNCI2014_001"
```

By default, TSClassifier is allowed in place of MDM but it will not work as this function is only working on the basis of the MDM classifier. The dataset can be switched.

`counterfactual_var_covs_functions.py` has three main functions. `compute_margins_covs_vars` computes margins for which the example switches classes when the covariances are set to zero, the variances are switched to the other class mean's variances or both are done at the same time. The margin are computed feature by feature, starting with the highest margin feature.

`isolate_var_cov_effects.py` computes if a class switch happens when all variances are switched, all covariances are set to zero or both are done at the same time.

`isolate_var_cov_flipped_effects.py` computes if a class switch happens when all variances are switched, all covariances are switched or both are done at the same time.

`computing_scores.py` is a script made to call `compute_margins_covs_vars` but it can easily be modified to call the other two functions described. This script produces a .csv file containing statistics on the computed margins as well as plots per subject for a given trial. 

It is best called this way :

```bash
python -m compute_margins_covs_vars --clf "MDM" --dataset "BNCI2014_001"
```

