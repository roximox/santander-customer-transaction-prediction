# Project Structure and Common Data Foundation

- Date: 2026-08-02
- Participants: Member 01 — Yassine El Hari; Member 02 — Aya Olali; Member 03 — Iliase Lhamri; Member 04 — Chaymae Akouaouch.
- Sprint goal: Establish a common project structure, reproducible data foundation, and shared conventions that enable the later analysis and modeling work packages.
- Results from previous sprint: Santander was selected as the common project, and initial responsibilities were distributed. Member 01 started establishing the common technical and data foundation as the project moved from planning into initial implementation.
- Discussion: The team reviewed the initial project setup and discussed how the common data foundation should support the individual work packages. They emphasized a shared dataset source, reproducible data loading, consistent project organization, and avoiding incompatible data versions. The foundation will provide a common basis for later validation and model comparison. The other work packages partly depend on this shared foundation before their full analysis and modeling workflows can begin.
- Decisions: The team will use one common reproducible data foundation and shared project conventions. Member 01 will continue establishing the technical and data foundation. Individual ML work will use the same data and evaluation principles. Train/test and validation methodology will be refined in the next phase, and reserved final test data will not be used for iterative model development or tuning.
- Task assignment: Member 01 will continue the project setup and common data foundation, including reproducible data loading and the basis for the later train/test and validation workflow. Member 02 will prepare the EDA approach and identify required descriptive analyses and visualizations. Member 03 will prepare the feature-engineering and dimensionality-reduction approach and identify requirements for later PCA and feature-selection work. Member 04 will prepare the Gradient Boosting modeling approach and identify the common data and evaluation inputs required before model development begins. All members will follow common project conventions, document their individual work, and communicate dependencies affecting other work packages.
- Risks and blockers: Several later tasks depend on the shared data foundation. Inconsistent preprocessing or validation could make later model comparisons unreliable, and technical conventions still require refinement. No technical failure was identified.
- Changes from initial plan: No major change was made to the responsibility distribution. The team confirmed the initial division of work and recognized that the common technical and data foundation should be stabilized before individual ML workflows become fully independent.
- Deadlines: No exact deadlines were set. Before the next meeting, the immediate priority is to progress the common data-handling and validation foundation so that individual analysis and modeling tasks can proceed.
- Next meeting: Review data processing, train/test strategy, validation methodology, and initial progress of EDA and baseline modeling.

## Retrospective

- What went well: The project moved from planning into initial implementation, and the team established a clearer common technical direction.
- What did not go well: Some individual work packages still depended on completion of the shared data and validation foundation.
- Improvement action: Finalize essential shared data conventions and clarify the validation workflow so that more tasks can proceed in parallel.
