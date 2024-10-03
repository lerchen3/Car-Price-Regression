My first machine learning project.

Solution that placed 5th on the public leaderboard here: https://www.kaggle.com/competitions/playground-series-s4e9/leaderboard?tab=public

For preprocessing, I experimented with bigrams and OpenFE-esque transformations, but neither produced significant improvements. Ultimately, I settled on ordered target encoding, which proved to be more compatible with the numerical models I developed.

In addition to using standard models like LGBM, CatBoost, and neural networks, I experimented with several novel approaches:

1. **Ensembles with Rotations**: I built an ensemble of LGBMs, each trained on data rotated by a random orthogonal matrix. This heuristic aimed to make the model more robust by avoiding axis-aligned decision boundaries. However, in practice, this approach performed worse than standard LGBMs, likely due to the noisiness of the dataset.
2. **Attention-Based NNs**: I implemented neural networks with attention mechanisms applied to features. Feature attention weights were optimized using Optuna, but this did not significantly outperform other models.
3. **Custom 'Power' Loss Functions**: I experimented with NNs and LGBMs using a custom loss function of the form abs(prediction-actual)^power, with power ranging from 1 to 2. This was inspired by combining MAE and MSE objectives to optimize different error distributions. Again, Optuna was used for hyperparameter tuning.


Ultimately, my final solution fed the predictions from an NN with the power loss function, along with external predictions, into a meta-model built using AutoGluon. This combination produced my final output.
