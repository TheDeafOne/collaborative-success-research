# Changes that are mainly rewording, clarification, or limitations

## 1. Explicitly state that `years_active` is not derived from the first five years
Reviewer 1 notes an inconsistency between the paper’s framing and the feature definition.

Required text change:
- In the introduction and methods, state that most features are derived from the first five years, with `years_active` as an exception.
- Explain that it was intended as a longevity control.
- Avoid saying that all features come strictly from the first five years unless `years_active` is removed from the main model.

## 2. Qualify the early-career prediction framing
Because `years_active` uses information through 2025, the paper should avoid implying that the current full model is a purely time-of-prediction early-career model.

Required text change:
- Rephrase claims from “early-career prediction” to something like “association between early-career collaboration structure and later Spotify follower counts,” unless a temporally valid model is added.
- If the revised model removes `years_active`, the stronger early-career framing can be retained for that version.

## 3. Note that `years_active` has limited use in real early-career candidate prediction
Reviewer 1 specifically asks for this interpretation.

Required text change:
- Add that in practical applications involving artists currently early in their careers, `years_active` would have limited discriminative value or would not be available in the same form.

## 4. Add era effects as a limitation
Reviewer 1 asks for a distinct limitation about calendar-era effects.

Required text change:
- Explain that artists’ first five-year windows occur in different historical periods.
- Note that MusicBrainz coverage density may vary by era.
- Note that pre-streaming success may map imperfectly onto December 2025 Spotify follower counts.
- State that this complicates mechanism interpretation, especially for feature importance and brokerage effects.

## 5. Clarify that elapsed time is controlled by `years_active`, but era effects may remain
This is mostly interpretive clarification.

Required text change:
- Distinguish longevity from era.
- Say that controlling for elapsed career duration does not fully control for cohort-specific differences in data coverage, platform adoption, industry structure, or genre visibility.

## 6. Present network contribution as incremental rather than dominant
Reviewer 2 accepts that graph features and embeddings improve performance, but says metadata variables appear strongest.

Required text change:
- Soften claims that network structure is the central or dominant predictor.
- State that network features add measurable predictive value beyond metadata.
- Avoid overstating mechanism claims if metadata features dominate feature importance.

## 7. Replace “musician success” with a more precise outcome label
Reviewer 2 says the outcome is Spotify followers, not success broadly.

Required text change:
- Replace “musician success” with “Spotify popularity,” “Spotify follower count,” or “later Spotify follower count.”
- In places where “success” remains, define it narrowly as operationalized by Spotify followers.

## 8. Clarify how global graph features are computed
Some of this may become research-work if leakage exists, but the manuscript definitely needs clearer wording either way.

Required text change:
- State whether graph metrics are computed on each artist’s ego/early-career graph, the full pooled collaboration graph, cohort-restricted graphs, or time-restricted graphs.
- State what information is included and excluded.
- State how node2vec embeddings are trained and what graph they use.

## 9. Add caveats around feature importance as mechanism evidence
Reviewer 1 says era effects do not destroy predictive validity but complicate interpretation.

Required text change:
- Rephrase feature importance as predictive importance, not causal mechanism.
- Avoid claiming that brokerage causes popularity unless the design supports it.
- Note that feature importance may reflect cohort, coverage, or platform-era effects.

## 10. Add supplementary reporting language
Reviewer 1 frames most requested changes as suitable for supplementary reporting.

Required text change:
- Move descriptive debut-year distribution, sensitivity checks, and possibly ablation tables to supplement if space is limited.
- Reference them clearly in the main text.