# SHL Grammar Scoring Engine
## Overview
This project explores automated grammar scoring from spoken English audio. The objective is to predict a continuous grammar score on a 0–5 scale from short WAV recordings.
The project was developed for a Kaggle challenge using audio representations, handcrafted acoustic and prosodic features, and regression models.
## Approach
- **Audio representations:** WavLM embeddings.
- **Handcrafted features:** Acoustic and prosodic descriptors.
- **Models evaluated:** Histogram-based Gradient Boosting and CatBoost.
- **Validation:** Five-fold cross-validation.
- **Ensembling:** Blending predictions from multiple regressors.
- **Calibration:** Investigated to address regression toward the mean.
## Results
| Evaluation | Result |
|---|---:|
| Best recorded public leaderboard score | 0.4897 |
| CatBoost cross-validation RMSE | 0.6301 |
| CatBoost cross-validation MAE | 0.4903 |
| CatBoost cross-validation Pearson correlation | 0.8649 |
The cross-validation results are internal estimates, not independent test-set performance. Model selection, blending, and calibration were explored using the available validation predictions, so the best cross-validation result may be optimistic. The public leaderboard score is reported separately and should not be interpreted as proof of generalization to all unseen recordings.
## Repository Contents
- `notebooks/01_baseline_github.ipynb` — notebook copy with saved cell outputs removed.
- `requirements-github.txt` — selected Python dependencies for the project.
- `requirements.txt` — the larger environment dependency list.
- `outputs/figures/` and selected experiment summaries — visualizations and model-comparison artifacts, where included.
The notebook records an experimental workflow rather than a guaranteed end-to-end reproduction. Some cells refer to local project directories or require cached features and competition files.
## Data and Reproducibility
Competition audio, labels, cached features, trained model binaries, and private submission files are not included. Obtain the dataset through the authorized competition source and follow its rules before using or redistributing it.
The notebook was developed in a local Windows environment. A clean installation and full end-to-end run on another machine have not been verified. WavLM feature extraction can be computationally expensive; existing cached features were used during later experiments.
## Limitations
- The training and test audio showed distribution differences in exploratory analysis.
- Validation-based model selection can overestimate performance.
- The public leaderboard is only one evaluation sample and may not reflect private leaderboard performance.
- Further validation on independent recordings would be needed before practical deployment.
## Technologies
Python, NumPy, pandas, scikit-learn, PyTorch, Transformers, librosa, CatBoost, and Jupyter Notebook.
