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
- **Model 4 — Weak ties.** Introduces `edgecov(low_overlap)` derived from neighborhood overlap/bridging scores. Its positive coefficient tests Goal 3 directly, showing weak ties are overrepresented even after density, structure, homophily, and exposure controls.

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
- **Weak-tie confirmation.** Evidence that `edgecov(low_overlap) > 0` provides the empirical hook for Goal 3; the same edge covariate underlies the node-level weak-tie fractions.

This produces a **mechanism-driven feature selection process** linking ERGM diagnostics to predictive modeling.

---

# 6. 📑 High-Level Paper Outline

### **1. Introduction**

- Motivation
    
- Importance of collaboration in creative industries
    
- Weak vs. strong ties
    
- Research gap
    
- Contribution of this study
    

### **2. Background & Theory**

- Granovetter and weak-tie theory
    
- Cultural production networks
    
- Prior computational models of creative collaboration
    
- Why network structure may predict artistic success
    

### **3. Data**

- MusicBrainz and Spotify
    
- Collaboration extraction
    
- First-n-songs design
    
- Network construction
    
- Preprocessing pipeline
    

### **4. Methods**

#### 4.1 Network Model (Bipartite → Projection)

- first-N-song window, MusicBrainz→Spotify integration, conversion to `network`
- projected artist graph stored in `data/graphs/better_graph`

#### 4.2 Node and Edge Attributes

- role, genre, label, geography metadata (`primary_*`, artist country/city)
- standardized productivity (`num_songs_std`, `collab_count_std`), tenure (`time_std`)
- weak-tie overlap indicator used as `edgecov`

#### 4.3 ERGM Ladder (Models 0–4)

- Model 0: density null (`edges = -6.30`)
- Model 1: add `gwesp`/`gwdegree` for structure
- Model 2: add homophily (`nodematch` genre/label/role/geography)
- Model 3: add exposure controls (`nodecov`, `absdiff`) with CD estimation
- Model 4: add weak-tie `edgecov(low_overlap)` to test Goal 3

#### 4.4 Node-Level Feature Engineering

- closure/triangle metrics (overall + within/across attributes), open wedges
- centrality/core measures, hub-distance, component sizes
- opportunity windows (open dyads, two-hop reach, standardized productivity/tenure)
- weak-tie fractions, community participation, ERGM residual diagnostics

#### 4.5 Predictive Modeling

- baseline vs. network vs. ERGM-informed models
- evaluation strategy and hypothesis-aligned feature pruning
    

### **5. Results**

- ERGM results: structural effects
    
- Network topology description
    
- Weak-tie insight
    
- Predictive model comparison
    
- Feature importance
    

### **6. Discussion**

- Interpretation
    
- Implications for creative industries
    
- Theoretical implications
    
- Limitations
    

### **7. Conclusion**

- Summary
    
- Future directions
    

### **Appendix**

- Full feature list
    
- ERGM diagnostics
    
- Sensitivity analyses
    
- Robustness checks
