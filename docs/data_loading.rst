2. Data Loading & Understanding
===============================

This section describes the loading of the dataset and the exploratory analysis required to understand its main characteristics, potential sensitive variables, and the presence of indirect proxies.

2.1 Feature Selection
---------------------

In this phase, all dataset features are statistically analyzed.

For each variable, the following indicators are computed:

- Missing values (count and percentage)
- Minimum, maximum, mean, standard deviation
- 1st percentile, 2nd percentile (median), 3rd percentile
- Data type (numeric, categorical, boolean, etc.)
- Number of distinct values
- Variable distribution (via histograms or density plots)

Additionally, the following have been identified:

- **Sensitive features** (e.g., gender, ethnicity, age, marital status)
- The **target feature(s)** (the variable to be predicted, such as hiring, promotion, etc.)

2.2 Proxy Identification
------------------------

This step aims to identify **proxy features**, i.e., variables that are not directly sensitive but are **correlated** with one or more sensitive attributes and may introduce indirect bias.

- A **correlation matrix** was generated across all features, with visual representation (e.g., heatmap)
- Significant correlation thresholds were defined (e.g., Pearson > 0.7) to highlight potential proxies
- Variables highly correlated with sensitive features were reviewed to assess their impact on the model

2.3 Bias Detection
------------------

Before applying any mitigation strategies, an initial bias assessment was conducted on the dataset.

For each **(sensitive feature, target feature)** pair and each **(sensitive group, target class)** combination, the following fairness metrics were calculated:

- **Statistical Parity Difference (SPD):** measures the difference in positive outcome probability across groups
- **Disparate Impact (DI):** measures the ratio of outcome probabilities between groups

These metrics allow us to assess **the dataset's inherent bias** before applying any fairness interventions.

