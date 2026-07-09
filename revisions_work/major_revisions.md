# Changes that require additional research-work / analysis

## 1. Re-run main predictive results without `years_active`
Reviewer 2 asks for the main results without this variable because it uses information unavailable at early-career prediction time. This is not just rewording; it requires model retraining and updated performance reporting.

Required work:
- Remove `years_active` from the feature matrix.
- Re-run the main predictive models.
- Compare performance against the original model.
- Update results, feature importance, and interpretation.

## 2. Replace `years_active` with a time-valid control, if desired
Instead of simply removing `years_active`, the paper could introduce a valid time-of-prediction control. This would require defining and computing a new feature.

Possible replacements:
- Career age fixed at prediction time, likely constant if all artists are evaluated after five years.
- Debut cohort indicators.
- Calendar debut year.
- Years elapsed between the end of the five-year window and outcome measurement, though this still needs careful interpretation.

Required work:
- Define a valid control.
- Justify it methodologically.
- Recompute features.
- Re-run models and compare results.

## 3. Clarify and possibly recompute global network features and node2vec embeddings to avoid temporal leakage
Reviewer 2 raises a substantive concern: if all early-career collaborations are pooled into one graph across many calendar eras, then earlier artists’ network positions may be affected by later cohorts.

This requires research-work if the current graph construction allows future-cohort information to influence earlier artists’ features.

Required work:
- Audit how global network features were computed.
- Determine whether features for an artist use collaborations or nodes that occurred after that artist’s five-year window.
- Determine whether node2vec was trained on a pooled graph containing future cohorts.
- If leakage exists, recompute graph features using only information observable by the relevant prediction time.
- Update the methods and results accordingly.

## 4. Chronological or cohort-based validation split
Reviewer 2 recommends training on earlier debut cohorts and testing on later ones. This is a new validation design, not a wording change.

Required work:
- Define debut-year cohorts or chronological cutoffs.
- Split training/test data by debut year.
- Recompute or restrict features according to what would have been observable at the prediction time.
- Re-run the predictive models.
- Compare chronological results to the original evaluation.
- Update claims if performance changes materially.

## 5. Temporally valid ablation table
Reviewer 2 asks for an ablation table under a chronological or cohort-based split.

Required work:
- Run models for:
  - Metadata-only.
  - Metadata without `years_active`.
  - Graph-only.
  - Embeddings-only.
  - Combined model.
- Ideally run these under the temporally valid split, not just the original split.
- Report predictive metrics consistently.
- Update the interpretation of the incremental contribution of network features.

## 6. Debut-year distribution
Reviewer 1 asks to report the distribution of debut years. This requires at least a descriptive analysis, though it is relatively lightweight.

Required work:
- Compute debut-year summary statistics.
- Possibly add a histogram or table.
- Report the range, median, quartiles, and concentration by era.
- Use this to support the added limitation about era effects.

## 7. Sensitivity analysis for betweenness sampling
Reviewer 2 notes that betweenness was estimated using 256 sampled source nodes. Since this is stochastic or approximation-dependent, sensitivity should be reported.

Required work:
- Recompute sampled betweenness under multiple random seeds and/or different sample sizes.
- Quantify stability of the betweenness feature.
- Assess whether model performance or feature importance changes.
- Report results in the supplement or robustness section.

## 8. Sensitivity analysis for node2vec random walks
Reviewer 2 asks for sensitivity to random seeds or sampling choices because node2vec uses random walks.

Required work:
- Train node2vec embeddings under multiple random seeds.
- Possibly vary key hyperparameters if not already fixed, such as walk length, number of walks, window size, and dimensions.
- Re-run predictive models or at least assess embedding stability and downstream performance stability.
- Report whether conclusions are robust.

## 9. Reassess feature importance and mechanism claims after temporal controls
Both reviewers question whether feature importance and mechanism interpretations may conflate brokerage/network effects with era effects or longevity effects.

Required work:
- Recompute feature importance after removing or replacing `years_active`.
- Recompute feature importance under chronological/cohort-based validation if added.
- Check whether network features remain important.
- Revise RQ1/RQ3 interpretations based on the updated evidence.