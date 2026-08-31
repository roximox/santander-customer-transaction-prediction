# Project Structure and Common Data Foundation
 
- **Date:** 2026-08-02
- **Participants:** Member 01 — Yassine El Hari; Member 02 — Aya Olali; Member 03 — Ilias El Hamri; Member 04 — Chaymae Akouaouch.
- **Sprint goal:** Establish a common project structure, reproducible data foundation, and shared conventions that enable the later analysis and modeling work packages.
- **Results from previous sprint:** Santander was selected as the common project, and initial responsibilities were distributed. Member 01 started establishing the common technical and data foundation as the project moved from planning into initial implementation.
## Discussion
 
We reviewed the initial project setup and discussed how the common data foundation should support the individual work packages. We agreed on the need for a shared dataset source, reproducible data loading, consistent project organization, and avoiding incompatible data versions.
 
This foundation will underpin later validation and model comparison, and the other work packages partly depend on it before their full analysis and modeling workflows can begin.
 
## Decisions
 
We agreed to use one common, reproducible data foundation and shared project conventions. Member 01 will continue establishing the technical and data foundation, and individual ML work will follow the same data and evaluation principles.
 
The train/test and validation methodology will be refined in the next phase, with particular attention to preventing data leakage and keeping test data separate from iterative model development and tuning.
 
## Task Assignment
 
- **Member 01** will continue the project setup and common data foundation, including reproducible data loading and the basis for the later train/test and validation workflow.
- **Member 02** will prepare the EDA approach and identify the descriptive analyses and visualizations needed.
- **Member 03** will prepare the feature-engineering and dimensionality-reduction approach and identify the requirements for later PCA and feature-selection work.
- **Member 04** will prepare the Gradient Boosting modeling approach and identify the common data and evaluation inputs needed before model development begins.
All members will follow the common project conventions, document their individual work, and flag dependencies that affect other work packages.
 
## Risks and Blockers
 
Several later tasks depend on the shared data foundation, so inconsistent preprocessing or validation could make later model comparisons unreliable. The technical conventions still need refinement.
 
No technical failure has been identified so far.
 
## Changes from Initial Plan
 
No major change to the responsibility distribution. We confirmed the initial division of work and agreed that the common technical and data foundation should be stabilized before individual ML workflows become fully independent.
 
## Deadlines
 
No exact deadlines were set. Before the next meeting, the immediate priority is progressing the common data-handling and validation foundation so individual analysis and modeling tasks can move forward.
 
## Next Meeting
 
Review data processing, the train/test strategy, validation methodology, and initial progress on EDA and baseline modeling.
 
## Retrospective
 
- **What went well:** The project moved from planning into initial implementation, and the team established a clearer common technical direction.
- **What did not go well:** Some individual work packages still depended on completing the shared data and validation foundation.
- **Improvement action:** Finalize the essential shared data conventions and clarify the validation workflow so more tasks can proceed in parallel.
 