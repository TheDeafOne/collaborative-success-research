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
    Attributes: roles, release dates, genre tags, team sizes
    
2. **Projected Artist–Artist Collaboration Network**  
    Edge weight metrics:
    - raw count of co-credits
    - team-size adjusted weight
    - recency-weighted collaboration strength
        
3. **Node Metadata**  
    Includes genres, roles, release windows, productivity (number of songs), and time-in-network.
    
4. **Graph assembly + attribute standardization (current workflow in `network_construction/ergm_new_features.ipynb`)**  
    - enforce canonical artist keys (`artist_mbid`/`mbid`) and convert all identifiers to character strings before graph construction  
    - standardize numeric node covariates (`productivity_total`, `productivity_std`, `collab_count_std`, `tenure_std`) and surface convenience aliases (`num_songs_std`, `time_std`) for ERGM terms  
    - sanitize edge attributes (weights, overlap scores, same-label indicators, low-overlap flags) so igraph/statnet receive consistent numeric inputs  
    - build an undirected igraph from `nodes.csv`/`edges.csv` and convert it to a `statnet` network object for modeling
    

---

### **5.2 Structural Models (ERGMs)**

The R/statnet workflow in `network_construction/ergm_new_features.ipynb` now walks through a staged specification so we can isolate where each mechanism becomes identifiable:

1. **Model 0 — Baseline density (`edges`)**  
   MPLE fit that confirms the collaboration network is extremely sparse (≈0.2% tie probability).

2. **Model 1 — Core structure (`edges + gwesp + gwdegree`)**  
   Adds geometrically weighted ESP (triadic closure) and degree to capture clustering and hub-formation; both show large, well-signed coefficients.

3. **Model 2 — Homophily (`+ nodematch("primary_genre")`)**  
   After testing all available attributes, genre matching is the only similarity term that remains stable, so the notebook keeps it and drops problematic role/label terms.

4. **Model 3 — Exposure controls (`+ nodecov(productivity_std) + nodecov(collab_count_std)`)**  
   Uses Contrastive Divergence for estimation once opportunity terms enter the model. Productivity and collaboration intensity absorb part of the tie propensity while leaving closure and degree effects intact.

5. **Model 4 — Weak-tie emphasis (`+ gwdsp`, optional `edgecov(low_overlap)`)**  
   Introduces geometrically weighted shared partners for dyads (open triads) plus a low-embeddedness edge covariate derived from edges with ≤1 mutual collaborator. This stage isolates weak-tie formation beyond what the closure term explains.

Across stages we rely on MPLE for quick screening, then refit key specifications via CD for stable inference and goodness-of-fit checks. Diagnostics from each step determine which mechanisms feed into subsequent feature engineering and hypothesis tests.

---

### **5.3 Node-Level Network Feature Engineering**

`ergm_new_features.ipynb` now exports a canonical `node_features.csv` that joins every artist/MBID with a battery of graph-derived metrics aligned to the ERGM terms. Feature families include:

- **Local cohesion + triads**: local clustering coefficient, triangle counts, open wedges, GWESP-style node scores that weight edges by shared-partner intensity.  
- **Positional structure**: degree/log-degree, eigenvector centrality, PageRank, k-core, betweenness, closeness, and minimum distance to the top-k hubs.  
- **Weak-tie + bridging signals**: fraction of an artist’s ties with ≤1 shared partner (`weak_tie_frac`), open-dyad opportunities, Louvain community participation coefficients, and component membership/size.  
- **Homophily context**: share of neighbors matching primary genre/label/role, triangles that stay within or span across those attributes, and participation in cross-genre triads.  
- **Opportunity controls**: productivity and tenure standard scores carried over from the data clean-up step so predictive models can control for exposure alongside structure.

These metrics give us observable counterparts to each ERGM mechanism; we can later augment them with simulation-based residuals once the final ERGM specification is locked.

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

Using the staged ERGM fits:

- Keep only feature families whose corresponding ERGM terms remain significant and well-signed after exposure controls (e.g., GWESP, GWDEGREE, genre homophily).  
- If a structural effect collapses once productivity/tenure terms enter, drop or down-weight its derived features in downstream models.  
- Use CD-based GOF to decide whether adding `gwdsp` or the low-overlap edge covariate materially improves fit; only then do we propagate weak-tie residuals.  
- Cross-check node-level weak-tie fractions, community participation, and open-dyad measures against the Model 4 coefficients to validate H1-oriented predictors.

This keeps feature selection **mechanism-driven** while ensuring weak-tie interpretations rest on an exposure-controlled ERGM.

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

- MusicBrainz and spotify
    
- Collaboration extraction
    
- First-n-songs design
    
- Network construction
    
- Preprocessing pipeline
    

### **4. Methods**

#### 4.1 Network Model (Bipartite → Projection)

#### 4.2 Node and Edge Attributes

#### 4.3 ERGM Framework

- structural terms
    
- homophily
    
- exposure
    
- weak ties
    

#### 4.4 Node-Level Feature Engineering

- structural features
    
- ERGM-derived expected/residual metrics
    

#### 4.5 Predictive Modeling

- baseline vs. network vs. ERGM-informed models
    
- evaluation strategy
    

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
