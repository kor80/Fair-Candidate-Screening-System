3. Training and Testing
========================

This section describes the training and evaluation process for the classification models, including both standard (no mitigation) and fairness-aware (with mitigation) approaches. All models are validated using a 5-fold cross-validation strategy.

Evaluation Setup
----------------

- **Validation strategy:** 5-fold cross-validation
- **Metrics:**
  - **Performance metrics:** Accuracy, Precision, Recall, F1-score, AUC
  - **Fairness metrics:** Demographic Parity, Equalized Odds Ratio
- **Comparison:** Results are shown both **with** and **without mitigation techniques**

Visualization
-------------

The following plots are generated to visualize performance and fairness:

- **Performance column bar plot:**  
  Displays mean ± standard deviation across folds for each performance metric  
  (two subplots: *without* mitigation, *with* mitigation)

- **Fairness column bar plot:**  
  Displays mean ± standard deviation across folds for each fairness metric  
  (two subplots: *without* mitigation, *with* mitigation)

3.1 Pre-processing Mitigation
-----------------------------

Pre-processing mitigation techniques are applied **before training** to reduce bias in the data itself.

- A comparison is made between the **original dataset** and the **transformed dataset** using a coordinate plot
- Techniques used may include:
  - Reweighting
  - Resampling (e.g., oversampling minority group)
  - Fair representation learning (e.g., learning latent fair embeddings)

These methods aim to remove correlations between sensitive features and the target variable before any model sees the data.

3.2 In-processing Mitigation
----------------------------

In-processing methods are applied **during model training**, modifying the learning process itself to enforce fairness.

- Fairness-aware algorithms are used (e.g., adversarial debiasing, fair boosting, or fairness-constrained optimization)
- These methods explicitly **optimize both accuracy and fairness** during learning
- Metrics from in-processing models are compared against baseline (unmitigated) models to evaluate trade-offs

