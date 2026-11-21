# Generative AI Use Log

**Course:** DATA 245, San Jose State University  
**Student:** Danish Waseem (Person 3)  
**Project:** Exoplanet Classification Using Machine Learning  
**My Role:** Decision Tree, Random Forest, pruning, overfitting analysis, feature importance, centroid sensitivity, shared data setup, and final integration and evaluation.

This log summarizes how Generative AI tools were used as supporting utilities during development and documentation. All research questions, experimental designs, data splits, leakage controls, and final evaluations were directed and verified directly by me.

Items marked **[CHECK]** are placeholders to confirm before submission.

---

## 1. Tools Used

| Tool | Period | Supporting Role |
|---|---|---|
| ChatGPT (OpenAI) | Oct 6 to Nov 20, 2026 | Reference for project planning, conceptual explanations, initial literature keywords, LaTeX formatting, and wording revisions |
| Cursor Agent (Cursor IDE) | Oct 31 to Nov 3, 2026 | IDE coding assistant for boilerplate setup, syntax debugging, formatting scripts, and organizing repository folders |
| Claude (Anthropic) | Nov 20, 2026 | Used for checking saved output tables, formatting integration checklists, and organizing slide outlines |

---

## 2. Work Completed Independently

* Defined my project scope, research questions, and core methodology choices with the team (grouped train/test splitting by `kepid`, leakage-aware feature selection, macro-F1 evaluation, and single final test evaluation).
* Designed and directed every stage of the modeling pipeline (data preprocessing, baseline trees, cost-complexity pruning, Random Forest tuning, permutation importance, and centroid sensitivity experiments).
* Reviewed and verified all intermediate metrics, figures, and validation outputs before advancing between stages.
* Coordinated with teammates by phone on November 20 to confirm final integration steps and approve the single test-set evaluation.
* Reviewed all pull requests, verified the final evaluation metrics against saved files, and wrote the core analysis for the reports and presentation.

**[CHECK]** Add any additional manual steps you want to highlight (e.g., specific code edits, manual checks, or report sections written entirely from scratch).

---

## 3. Mandatory Report Question

**Did you find or come across solutions to similar problems by using Generative AI or other sources?**

Yes. During initial background research, ChatGPT suggested general keywords and standard references related to exoplanet classification and machine learning methods. It did not provide ready-made solutions or code for this project, and no experimental results came from AI. I reviewed the literature to understand standard methodologies and cited the relevant works in the report:

| Source | Relevance to Project |
|---|---|
| McCauliff et al. (2015), automatic classification of Kepler transit candidates | Background on applying Random Forests to Kepler candidate data |
| Armstrong et al. (2021), exoplanet validation with machine learning | Context on machine learning approaches for Kepler validation |
| Thompson et al. (2018), Kepler DR25 catalog | Understanding Kepler vetting flags and potential label leakage |
| Breiman (2001), Random Forests | Foundational methodology for Random Forest ensembles |
| Kaufman et al. (2012), leakage in data mining | Principles for preventing data leakage in tabular pipelines |
| Altmann et al. (2010), permutation importance | Methodology for permutation-based feature importance |
| scikit-learn documentation | Reference for cost-complexity pruning, grouped CV, and model APIs |

**[CHECK]** Confirm that every cited paper above has been opened and verified for exact author names, publication year, title, and findings before final submission.

---

## 4. Summary of Assistance by Task

### Project Planning and Proposal (ChatGPT)
* **Support Provided:** Helped organize the initial proposal structure, suggested LaTeX formatting templates, and provided general reference notes on FAIR data management principles and grouped validation concepts.
* **Verification:** I selected my project role, defined the specific hypotheses, and wrote and edited the proposal text.

### Pipeline Scaffolding and Code Formatting (Cursor IDE)
* **Support Provided:** Assisted with initial repository folder organization (`src/`, `docs/`, `results/`), boilerplate data-loading templates, syntax debugging, and formatting scripts for cross-validation loops, pruning curves, and permutation importance routines.
* **Verification:** I specified all modeling rules (random seed 245, `kepid` grouping, exclusion of `koi_fwm_stat_sig` from the primary set, and strict isolation of the test partition). I ran all scripts locally, inspected unit tests (`pytest`), and verified every saved metric and figure directly.

### Final Integration and Presentation Prep (Claude and ChatGPT)
* **Support Provided:** Assisted with formatting the combined team comparison tables, structuring the final test checklist, organizing demo script boilerplate (`demo/predict_koi.py`), and drafting slide bullet outlines.
* **Verification:** All final test metrics were generated directly by running our evaluation scripts on the held-out test set after team approval. I cross-checked all reported numbers against the saved CSV outputs and edited all report and slide text myself.