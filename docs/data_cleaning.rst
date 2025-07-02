1. Data Cleaning (Selection)
=============================

In this initial phase, the raw dataset is subjected to a cleaning and selection process to ensure a consistent, noise-free foundation suitable for the classification phase.

Objective
---------

- Prepare a dataset that can be used for **classification**
- Remove inconsistent data, duplicates, or records with excessive missing values
- Apply logical or statistical filters to exclude irrelevant entries

Actions Performed
-----------------

- Removed records with more than **30% missing values**
- Eliminated **irrelevant features** (e.g., unused IDs, timestamps, free-text comments)
- Converted categorical columns into readable or encoded formats
- Standardized data types (e.g., converted string-formatted floats into numeric types)

Outcome
-------

The resulting dataset was reduced from **X** initial features to **Y** selected features, with a total of **N** instances (records), ready for further analysis.

The dataset was validated for absence of duplicates, consistency of data types, and sufficient information density to proceed with classification tasks.
