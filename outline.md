## **1. Project Overview**

This project investigates how creative relationships shape success in the modern music industry. We model the industry as a large-scale network of songwriters, producers, performers, and composers who connect through collaboration on songs. By leveraging openly available data from MusicBrainz and Spotify, we construct a collaboration network that captures both artistic and social ties, and we analyze how its structural properties are associated with long-term career success.

Where prior research has focused on individual-level characteristics (talent, productivity, genre), this project shifts the emphasis toward **relational structure** — the idea that _who you collaborate with_ and _how your collaborations position you within the creative ecosystem_ may meaningfully influence commercial success. This perspective is inspired by **Granovetter’s Strength of Weak Ties**, which proposes that weak, cross-group connections act as bridges for information, opportunity, and innovation.

The project integrates tools from **network science**, **social theory**, and **machine learning**, combining theory-driven generative models (ERGMs) with predictive models of success. This dual approach allows us to ask both mechanism-oriented and outcome-oriented questions.

---
## **2. Goals**

### **Goal 1 — Identify Structural Mechanisms of Collaboration**
We aim to understand _how_ collaboration networks form by evaluating mechanisms such as:
- **Triadic closure** (strong ties)
- **Weak-tie formation** (bridging across communities)
- **Preferential attachment** (popularity-driven visibility)
- **Homophily** (genre, role, or stylistic similarity)
- **Exposure effects** (opportunity driven by shared activity windows)

We fit **Exponential Random Graph Models (ERGMs)** to quantify the extent to which each mechanism contributes to real-world collaboration patterns.

---
### **Goal 2 — Predict Artist Success from Network Structure**
We test whether structural position — measured through node-level features derived from the network and ERGM simulations — predicts artist success above and beyond non-network baselines.

We use spotify's follower count as our success metric

This is approached via a set of supervised learning models (regression, random forest, neural networks).

---

### **Goal 3 — Measure the Value of Weak Ties and Network Diversity**
We evaluate whether artists with:
- more **weak ties**,
- more **cross-genre collaborations**, or
- more **community-spanning** relationships

demonstrate higher long-term success. This provides an empirical test of Granovetter’s theory in a creative-industry context.

---

## **3. Hypotheses**

### **H1: Weak Tie Advantage**

Artists with **more weak ties** — low embeddedness, low overlap, cross-genre or cross-role connections — will achieve higher success than those embedded in dense, homogeneous neighborhoods.

### **H2: Strong-Tie Stability (Triadic Closure)**

Collaboration exhibits strong **triadic closure**, forming tightly knit groups of co-writers and producers. This helps explain stability and role specialization inside genres.

### **H3: Homophily and Creative Clustering**

Artists tend to collaborate within their own genres or role categories (writers with writers, etc.), producing genre-specific sub-networks.

### **H4: Network Position Predicts Success**

Measures of centrality, community bridging, and structural holes correlate positively with long-term success.  
We expect **weak-tie residuals** and **bridging scores** to be especially predictive.

---

## **4. Key Assumptions**
1. **Collaboration accurately reflects meaningful creative relationships.**  
    Co-writing and co-production capture real artistic interaction relevant to diffusion of ideas and opportunity.
    
2. **Early-career structure is predictive.**  
    We use each artist’s **first _n_ songs** to build the network, treating this structure as a foundation for future success.
    
3. **Success measured at a fixed endpoint is appropriate.**  
    Current Spotify metrics are treated as dependent variables summarizing long-term career outcomes.
    
4. **Open-source metadata is reliable enough** to identify roles, genres, and collaboration events.
    
5. **ERGMs can identify underlying generative mechanisms** in the collaboration network.
    

---

## **5. Methods**

### **5.1 Data Construction**

1. **Bipartite Artist–Song Network**  
    Nodes: artists  
    Nodes: songs  
    Edges: contributions (writing, production, performance)  
    Attributes: roles, release dates, genre tags, team sizes; derived from the first-N-song window defined in preprocessing.
    
2. **Projected Artist–Artist Collaboration Network**  
    Edge weight metrics:
    - raw count of co-credits
    - team-size adjusted weight
    - recency-weighted collaboration strength
    - weak-tie overlap indicator (`low_overlap`) exported for `edgecov`
        
3. **Node Metadata**  
    Includes `primary_genre`, `primary_label`, `primary_role`, `artist_country`, `artist_city`, release windows, standardized productivity counts (`num_songs_std`, `collab_count_std`), and time-in-network (`time_std`) so ERGM formulas can encode opportunity.
    

---

### **5.2 Structural Models (ERGMs)**

We build a ladder of ERGMs (`statnet`) so each conceptual mechanism is added transparently:

- **Model 0 — Baseline density.** `edges = -6.30` (`logit^{-1} ≈ 0.0018`), confirming the sparse null that later odds-ratio interpretations reference.
- **Model 1 — Core structure.** Adds `gwesp(0.5,fixed=TRUE)` and `gwdegree(0.8,fixed=TRUE)`; closure ≈ 4.5 (90× odds), while degree penalty (≈ -3.2) keeps hubs realistic.
- **Model 2 — Homophily.** Extends Model 1 with `nodematch` on `primary_genre`, `primary_label`, `primary_role`, `artist_country`, `artist_city`; all positive even after structural controls, evidencing creative clustering.
- **Model 3 — Exposure controls.** Adds `nodecov(num_songs_std)`, `nodecov(collab_count_std)`, `absdiff(time_std)` and switches to Contrastive Divergence. Productivity (≈ 0.17) and aligned tenure (≈ 0.15) are significant; closure strengthens (`gwesp ≈ 5.3`) and genre homophily ≈ 2.0.
- **Model 4 — Weak ties.** Introduces `gwdsp` derived from neighborhood overlap/bridging scores. Its positive coefficient tests Goal 3 directly, showing weak ties are overrepresented even after density, structure, homophily, and exposure controls.

Each rung supplies mechanism-specific coefficients, diagnostics, and simulated expectations that we later convert into residual node features and hypothesis tests.

---

### **5.3 Node-Level Feature Engineering**

We align node features with the ERGM terms while adding diagnostics for prediction:

- **Local structure:** clustering coefficients, triangle counts (overall plus within/across genre/label/role), open wedges, and a `gwesp`-style closure score per artist.
- **Centrality & core:** degree/log-degree, eigenvector, betweenness, closeness, k-core index, and minimum distance to the network’s hubs.
- **Opportunity windows:** component sizes, two-hop reach, open dyads, standardized productivity (`num_songs_std`, `collab_count_std`), and tenure (`time_std`).
- **Diversity & weak ties:** weak-tie edge fractions, neighborhood composition (same-genre/label/role shares), cross-community participation coefficients.
- **Model-informed residuals:** observed vs. ERGM-expected degrees, triangles, weak-tie scores, and closure-pressure residuals.

These outputs feed both hypothesis tests (Goal 3) and supervised models (Goal 2) by capturing how each artist deviates from the generative baseline.

---

### **5.4 Predictive Modeling**

We compare:

1. **Baseline models (no network features)**
    
    - regression
        
    - ridge
        
    - random forest
        
    - neural network
        
2. **Network-feature models**  
    Adds structural metrics, weak-tie scores, community bridging features.
    
3. **ERGM-informed models**  
    Includes expected/residual metrics from model simulations.
    

We measure performance improvements to quantify the predictive contribution of network structure.

---

### **5.5 Hypothesis Testing & Feature Pruning**

Using ERGMs:

- **Term stability.** Significant, well-mixed coefficients (closure, homophily, weak ties) justify keeping their related features in the predictive design matrix.
- **Exposure-aware pruning.** If an effect vanishes once `nodecov`/`absdiff` controls enter, we drop the associated features to avoid conflating opportunity with structure.
- **GOF-driven emphasis.** Terms that materially improve GOF or diagnose misfit guide which residuals to compute and highlight in the narrative.
- **Weak-tie confirmation.** Evidence that `gwdsp > 0` provides the empirical hook for Goal 3; the same edge covariate underlies the node-level weak-tie fractions.

This produces a **mechanism-driven feature selection process** linking ERGM diagnostics to predictive modeling.

---
# 1. Introduction
## 1.1 Motivation
Collaboration networks are central to creative industries. Early-career structural positioning may predict long-term success. We focus on deriving predictive value from collaboration patterns using network science + machine learning.

## 1.2 Key Questions
What mechanisms shape how collaboration networks form?  
Can early-career structural features predict future success?  
Which aspects of network position (weak ties, closure, homophily) matter most?

## 1.3 Contributions
Construct a large-scale, early-career collaboration network.  
Fit ERGMs to uncover generative mechanisms of tie formation.  
Translate ERGM terms → node-level predictive features.  
Add learned node embeddings (node2vec).  
Train XGBoost regression models to predict success (Spotify followers).  
Demonstrate gains from network-informed features.

# 2. Background and Related Work (Concise Outline)

## 2.1 Networks, Collaboration, and Creative Success
- Network position shapes opportunity (weak ties, structural holes).
- Small-world team structures improve creative outcomes.
- Early network embeddedness predicts long-run success across arts and sciences.
- Principle: success depends on both talent and collaboration structure.

## 2.2 Collaboration Networks in Music
- Music networks show large connected components, genre clusters, cumulative advantage.
- Collaboration-profile studies identify structural types tied to chart success; some transitions precede success gains.
- Centrality and mentorship/co-credit ties relate to popularity.
- High-status partnerships improve chart performance.

## 2.3 Predicting Success in Music
### 2.3.1 Song-level (Hit Song Science)
- Focus on predicting song popularity using audio/multimodal features.
- Gaps: limited modeling of artist careers or collaboration networks.

### 2.3.2 Artist-level
- Large-scale models use centrality + productivity but rely on full-career aggregates.
- Panel-data studies are small and sparse.
- Missing: systematic modeling of early-career network structure.

## 2.4 Adjacent Creative Fields
- Visual art and film show early network pathways predict career outcomes.
- Hot-streak research highlights clustered impact shaped by early opportunity.
- Early-career network signals are broadly predictive.

## 2.5 Network Models and Representation Learning
- ERGMs capture generative mechanisms but are rarely applied in music.
- Node2vec embeddings capture higher-order structure but are underused for artist prediction.
- Opportunity: integrate ERGM-informed features with learned representations.

## 2.6 Positioning of Present Work
- Gaps: few artist-level models, little early-career focus, minimal generative network modeling.
- Our approach:
  - Build early-career collaboration networks (MusicBrainz + Spotify).
  - Fit ERGMs to identify structural mechanisms.
  - Convert mechanisms into predictive features + node2vec embeddings.
  - Evaluate improvements in forecasting long-term artist success.
- Goal: emulate A&R/investor decision-making using only first-five-year collaborations.

# 3. Data & Network Construction
## 3.1 Data Sources
MusicBrainz (collaborations, roles, metadata).  
Spotify API (followers, popularity, genres).

## 3.2 Early-Career Window
Use first 5 years of each artist’s career.  
Success measured at present day (Spotify followers).

## 3.3 Artist–Song Bipartite Graph
Nodes: artists | songs.  
Edges: writing, producing, performing relationships.  
Attributes: roles, release dates, team sizes, genres.

## 3.4 Projected Artist Collaboration Network
Edge weights: raw co-writes, team-size adjusted, recency-weighted.  
Weak-tie overlap computed from neighbor overlap.

## 3.5 Cleaning & Filtering
Remove non-person entities, resolve duplicates, handle missing metadata.  
Restrict to artists with minimum releases + Spotify presence.

## 3.6 Descriptive Statistics
Degree distribution, components, genre breakdown, temporal coverage.

# 4. Network Formation Modeling (ERGMs)
## 4.1 Purpose
Identify mechanisms explaining tie formation.  
Provide generative baseline informing node-level features.

## 4.2 Model Ladder
Model 0: density.  
Model 1: structural terms (gwesp, gwdegree).  
Model 2: homophily (genre, role, label, geography).  
Model 3: opportunity controls (productivity, collaboration count, tenure).  
Model 4: weak-tie term (gwdsp).

## 4.3 Findings
Strong triadic closure.  
Clear homophily.  
Productivity & exposure effects.  
Weak ties overrepresented after controls.

## 4.4 Diagnostics
MCMC convergence, goodness-of-fit, simulation validation.

# 5. Node-Level Feature Engineering
## 5.1 Structural Features
Degree, log-degree, eigenvector, betweenness, closeness, k-core.  
Distance to hubs, component metrics.

## 5.2 Triadic / Closure Features
Clustering coefficient, triangle counts, open wedges.  
ERGM-style closure proxies.

## 5.3 Homophily Features
Same-genre / same-label / same-role neighbor fractions.  
Within- and cross-community interactions.

## 5.4 Weak-Tie & Bridging Features
Edge embeddedness / overlap.  
Weak-tie fractions, participation coefficients.  
Structural-hole indicators.

## 5.5 Productivity & Opportunity Features
Release counts, collaborator counts, tenure.  
Activity alignment windows.

## 5.6 ERGM-Informed Residuals
Observed vs expected degree, triangles, weak-tie levels.  
Closure- and bridging-pressure deviations.

## 5.7 Node2Vec Embeddings
Learn 128–256 dimensional embeddings.  
Capture latent structure.  
Concatenate with engineered features.

# 6. Predictive Modeling
## 6.1 Success Metric
Spotify follower count (log-transformed).

## 6.2 Model Comparisons
Baseline (metadata).  
+ Network features.  
+ ERGM-informed features.  
+ Node2vec.  
Full model: metadata + network + ERGM + embeddings.

## 6.3 Training Setup
Train/test split by debut cohort.  
XGBoost primary model; ridge + RF for comparison.  
Hyperparameter tuning, cross-validation.

## 6.4 Results
Network features improve performance.  
ERGM + weak-tie features give further lift.  
Node2vec improves still more.  
Feature importance highlights weak ties, closure, bridging.

# 7. Discussion
## 7.1 Interpretation
Early collaboration structure predicts long-term success.  
Weak ties & cross-community bridging matter.  
ERGM-informed features add value.

## 7.2 Practical Implications
Data-driven A&R scouting.  
Network-aware prediction complements audio-based models.

## 7.3 Limitations
Missing metadata, geographic bias.  
Spotify as proxy for success.  
Static early-career window.

## 7.4 Future Work
Dynamic ERGMs, temporal embeddings.  
Multimodal modeling.  
Cross-platform success metrics.

# 8. Conclusion
Summary of findings and contributions.  
Network structure is central to predicting artistic success.  
Combining ERGMs and ML yields actionable insights.




# Look into
how stable are some of these categorical factors (does removing them effect the coefficients of the other factors?)
for each of these factors, how much of the variance is "soaked up" by them. If a categorical variable is soaking up variance, it might be skewing the results of other factors.

- check out the diameter of the network


multiple paper writing (talk about different portions of what we did):
you can gain better understanding of the domain
you can talk about the methods


Talk about whats novel about the paper/process in the intro and background
spend time thinking about the novelty - natural tendency is to think about the linear work (no bad)
think about what you discovered. 