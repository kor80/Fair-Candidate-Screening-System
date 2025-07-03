Dataset and use-case context
============================

The dataset used in our study originates from the Adecco Group and reflects a snapshot
of one day’s operation of their candidate recommendation system.
This system automatically matches job openings with potential candidates by analyzing
resume data and job requirements.

Key characteristics of the dataset:

- For each job position, the system outputs the *top 10* most compatible candidates.
- Candidate-job matches are ranked by match score (descending), with secondary sorting by candidate ID (ascending).
- The dataset is anonymized and includes demographic and contextual variables such as: age, gender, domicile region, education, job sector, etc.

This data offers a unique opportunity to examine the internal behavior of an existing
AI hiring tool and assess its fairness characteristics at scale.
