# AI-READI HbA1c digital phenotyping

Analysis code accompanying a study of the incremental information
provided by wearable, dietary, social, and basic clinical predictors
for estimating recorded laboratory HbA1c.

## Current contents

- aireadi_no_cgm.py: CGM-independent data preparation.
- aireadi_no_cgm_models.py: model development and evaluation.

## Study design

The analysis uses the AI-READI recommended participant splits.
Revised eligibility does not require continuous glucose monitoring
(CGM) availability.

Models were fitted using eligible training participants. Revised
validation and test evaluations are exploratory because results from
earlier analyses had already been examined.

The outcome is recorded visit-day HbA1c. This is cross-sectional
estimation, not prospective forecasting.

## Data access

Participant-level data are not included in this repository.
Researchers must obtain AI-READI data through its official access
process and comply with the applicable data-use conditions.

## Reproducibility status

This repository is being prepared for the manuscript release.
Execution instructions, configuration information, additional analysis
scripts, and reproducibility records are being assembled.
