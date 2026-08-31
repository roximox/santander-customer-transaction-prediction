# Data Processing, Validation Strategy and Start of Individual Analysis
 
- **Date:** 2026-08-09
- **Participants:** Member 01 — Yassine El Hari; Member 02 — Aya Olali; Member 03 — Ilias El Hamri; Member 04 — Chaymae Akouaouch.
- **Sprint goal:** Establish a reliable data-processing and validation workflow and enable members to start their individual analysis and Machine Learning tasks on a consistent basis.
- **Results from previous sprint:** The common project structure and data foundation progressed, and reproducible access to the project data was established. This gave the team a sufficiently common basis to move toward data splitting, validation, EDA, and initial model development.
## Discussion
 
We discussed how to prepare the data for Machine Learning and agreed on a consistent train/test and validation strategy. The dataset is split using a stratified 80/20 split into 160,000 development observations and 40,000 reserved final-test observations. Model development and comparison will use the development partition, with five-fold StratifiedKFold cross-validation and random_state=42 as the common validation basis.
 
The reserved final-test partition must remain separate from iterative model development, tuning, and model selection. With the shared foundation now in place, the individual work packages can progress more independently, marking a shift from common setup work toward parallel scientific work.
 
## Decisions
 
We agreed to use the same data split and validation principles across the project:
 
- stratified 80/20 development/test split;
- 160,000 observations for development;
- 40,000 observations reserved for final testing;
- five-fold StratifiedKFold for cross-validation;
- random_state=42 for reproducibility;
- no use of the reserved final-test partition during iterative model development or tuning.
All members will apply comparable evaluation principles, flag important methodological changes early, and increasingly progress their individual work packages in parallel.
 
## Task Assignment
 
- **Member 01** will continue the data-processing and validation foundation, initial baseline modeling, and the planned Logistic Regression responsibility.
- **Member 02** will begin exploratory data analysis on the common dataset — distributions, descriptive characteristics, variable relationships, and relevant visual analysis — while preparing the basis for later tree-based modeling.
- **Member 03** will begin preparing the feature-engineering and dimensionality-reduction workflow, reviewing the requirements for PCA and feature selection and preparing later reduced-feature modeling.
- **Member 04** will begin the Gradient Boosting / HistGradientBoosting modeling work using the common development data and validation protocol, with optimization and detailed diagnostic evaluation planned for later stages.
## Risks and Blockers
 
Inconsistent train/test or validation procedures could make model results difficult to compare. Preprocessing decisions must not introduce data leakage, and the reserved final-test partition must remain isolated from iterative development. Members therefore need to stay synchronized on shared data and evaluation conventions.
 
No concrete technical error has been identified.
 
## Changes from Initial Plan
 
No major change to the responsibility distribution. The project is moving from a setup-heavy phase toward parallel individual analysis and modeling work now that the common data and validation foundation is available.
 
## Deadlines
 
No exact deadlines were set. Before the next meeting, Member 01 will continue baseline and validation work, Member 02 will progress EDA, Members 03 and 04 will begin their respective modeling work, and everyone will document initial progress and open questions.
 
## Next Meeting
 
Review the first concrete results and observations from the individual work packages: baseline modeling, EDA, feature engineering and PCA, Gradient Boosting, and any methodological issues or dependencies discovered during implementation.
 
## Retrospective
 
- **What went well:** The common project foundation and validation strategy allowed the team to move toward parallel analysis and model development.
- **What did not go well:** The individual modeling and analysis work was only beginning, so comparable results across the different work packages were not yet available.
- **Improvement action:** Apply the common validation protocol consistently and discuss methodological changes early so later model comparisons remain valid.
 
