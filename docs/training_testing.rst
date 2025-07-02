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

In-processing methods are applied **during model training**, modifying the learning process itself to enforce fairness constraints directly within the optimization loop.

We experimented with three main fairness-aware algorithms:

- **Adversarial Debiasing**
- **FaUCI (Fair Uncertainty-aware Classification Index)**
- **Prejudice Removal**

These methods aim to **jointly optimize predictive performance and fairness**, using different fairness regularization strategies.

Among these, **Adversarial Debiasing consistently achieved the best trade-off** between fairness and accuracy. It works by introducing an adversarial component that learns to predict the sensitive attribute, while the main model is trained to minimize its predictive power. This leads to internal representations that are less biased and more equitable across groups.

In contrast, **FaUCI and Prejudice Removal** showed limited improvement in fairness, even when tuning regularization strengths (`lambda`, `eta`). The main bottleneck was the **lack of sufficient examples from minority sensitive groups**, which prevented the algorithms from learning generalizable fairness constraints. Data augmentation partially mitigated this issue, improving fairness outcomes for underrepresented subgroups.

Furthermore, we observed that **fairness metrics degraded when using standard k-fold cross-validation**, due to the absence of stratification over sensitive attributes. Some folds lacked representation for certain groups, making fairness evaluation unreliable. When using a **stratified split based on the sensitive feature**, both fairness training and evaluation improved significantly.

For this reason, we strongly recommend using **stratified cross-validation over sensitive attributes** when training and evaluating fairness-aware models.


3.3 Analysis of In-processing Techniques
========================================

This section focuses on the evaluation of in-processing mitigation strategies used during model training. The primary goal was to reduce bias while maintaining model performance.

Adversarial Debiasing
----------------------

Among the methods tested, **Adversarial Debiasing** showed the most promising results across both fairness and performance metrics.

- This technique trains a secondary adversarial model that tries to **predict the sensitive attribute** from the main model's internal representation.
- Simultaneously, the main model is optimized to **prevent the adversary from succeeding**, thereby learning representations that are **informative but not biased**.
- In our experiments, Adversarial Debiasing consistently **reduced Disparity Metrics** such as Demographic Parity and Equalized Odds, while maintaining high performance (e.g., F1-score, AUC).
- It also proved to be **robust with minimal hyperparameter tuning**, making it the most effective of the evaluated in-processing techniques.

FaUCI and Prejudice Removal
----------------------------

The other two in-processing methods—**FaUCI** and **Prejudice Removal**—showed limited effectiveness, even when increasing regularization parameters such as `lambda` and `eta`.

- These methods **did not significantly improve fairness metrics** in our initial tests.
- A key issue was the **lack of representative samples** for combinations of sensitive groups and target classes.
- Without enough examples in underrepresented subgroups, these methods struggled to learn fair decision boundaries.
- We addressed this limitation by applying **targeted data augmentation**, which improved fairness results and highlighted the importance of **balanced subgroup representation**.

Effect of Cross-Validation
---------------------------

An unexpected observation was that fairness metrics were generally **better when using a simple train/test split** compared to k-fold cross-validation.

- This is due to the fact that **standard cross-validation does not guarantee balanced representation of sensitive subgroups** across folds.
- Some folds may contain very few or no examples from certain sensitive groups, causing instability in fairness metrics and poor generalization.

To overcome this, we recommend using **stratified cross-validation based on the sensitive attribute**. This ensures each fold contains a proportional representation of all sensitive subgroups and allows for more **reliable fairness evaluation**.

Key Takeaways
--------------

- **Adversarial Debiasing** is the most effective in-processing mitigation strategy in this study.
- **Fairness performance is highly sensitive to subgroup representation** in the dataset.
- **Data augmentation** can help improve fairness when subgroup data is sparse.
- Always use **stratified cross-validation over sensitive features** for fairness-aware model evaluation.


