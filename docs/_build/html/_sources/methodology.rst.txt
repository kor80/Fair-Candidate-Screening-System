Methodology
===========

1. Data Cleaning (Selection)
----------------------------

In this initial phase, the raw dataset is subjected to a cleaning and selection process to ensure a consistent,
noise-free foundation suitable for the classification phase.

2. Data Loading & Understanding
-------------------------------

2.1 Feature selection
~~~~~~~~~~~~~~~~~~~~~

In this phase, all dataset features are statistically analyzed.

For each variable, the following indicators are computed:

- Missing values (count and percentage)
- Minimum, maximum, mean, standard deviation
- 1st percentile, 2nd percentile (median), 3rd percentile
- Data type (numeric, categorical, boolean, etc.)
- Number of distinct values
- Variable distribution (via histograms or density plots)

Additionally, the following have been identified:

- **Sensitive features** (e.g., gender, age, domicile_region)
- The **target feature(s)** (the variable to be predicted, such as **hiring**)

2.2 Proxy Identification
~~~~~~~~~~~~~~~~~~~~~~~~

This step aims to identify **proxy features**, i.e., variables that are not directly sensitive but are **correlated** with one or more sensitive attributes and may introduce indirect bias.

- A **correlation matrix** was generated across all features, with visual representation (e.g., heatmap)
- Significant correlation thresholds were defined (e.g., Pearson > 0.7) to highlight potential proxies
- Variables highly correlated with sensitive features were reviewed to assess their impact on the model

2.3 Bias Detection
~~~~~~~~~~~~~~~~~~

Before applying any mitigation strategies, an initial bias assessment was conducted on the dataset.

For each **(sensitive feature, target feature)** pair we computed the **Statistical Parity Difference** fairness metric,
which measures the difference in positive outcome probability across groups.
This metric allow us to assess **the dataset's inherent bias** before applying any fairness interventions.


Sensitive feature selection
----------------------------

The construction of this variable follows two steps:

- **Semantic definition**: We first defined the grouping schema based on domain knowledge and contextual relevance. Geographic regions were aggregated to capture meaningful socioeconomic and cultural distinctions, while gender was included as an established axis of potential bias in hiring.
- **Validation of assumptions**: We then examined the resulting distributions in the data to verify that our semantic definitions reflected actual patterns of variation and imbalance. This step ensured that the combined variable meaningfully captured intersectional disparities potentially relevant to fairness assessments.

By explicitly modeling *grouped_region_gender* as a sensitive attribute, we aimed to detect nuanced forms of bias that might arise at the intersection of location and gender, which could be overlooked in analyses considering these features independently.

