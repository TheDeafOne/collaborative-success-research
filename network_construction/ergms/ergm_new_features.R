suppressPackageStartupMessages({
  library(statnet)
  library(dplyr)
  library(intergraph)
  library(igraph)
  library(ergm.count)
})

# ---------------- helpers ----------------
timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
log_step <- function(msg) cat(sprintf("[%s] %s\n", timestamp(), msg))
log_error <- function(msg, log_path) {
  cat(sprintf("[%s] ERROR: %s\n", timestamp(), msg))
  cat(sprintf("[%s] %s\n", timestamp(), msg), file = log_path, append = TRUE)
}

get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  match <- grep(file_arg, cmd_args)
  if (length(match) > 0) {
    return(dirname(normalizePath(sub(file_arg, "", cmd_args[match]))))
  }
  # fallback to current working directory
  normalizePath(getwd())
}

ergm_formula_from_terms <- function(terms) {
  as.formula(paste("net ~", paste(terms, collapse = " + ")))
}

has_vertex_attr <- function(g, attr) attr %in% vertex_attr_names(g)

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
  path
}

# ---------------- config ----------------
# Use a small subgraph for quick pipeline validation.
sample_graph <- TRUE
sample_nodes_target <- 5000
sample_seed <- 42
required_nodematch_attrs <- c(
  "primary_genre", "primary_role",
  "artist_country", "primary_label",
  "artist_region_city"
)
required_numeric_attrs <- c("time_std", "collab_count_std", "num_songs_std")
# Attributes used in ERGM terms (used for validation)
nodematch_attrs <- c(
  "primary_genre", "primary_role",
  "artist_country", "primary_label",
  "artist_region_city"
)
numeric_covars <- c("time_std", "collab_count_std", "num_songs_std")

safe_fit <- function(
  name,
  formula,
  net,
  results_dir,
  control_ergm = control.ergm(MPLE.samplesize = 1e5, MPLE.covariance.samplesize = 0)
) {
  log_path <- file.path(results_dir, "errors.log")
  model <- NULL
  err <- NULL

  log_step(paste0("Fitting ", name, " with ergm (MPLE)"))
  tryCatch({
    model <- ergm(
      formula,
      estimate = "MPLE",
      control = control_ergm
    )
  }, error = function(e) {
    err <<- e
    log_error(paste("ergm failed for", name, ":", conditionMessage(e)), log_path)
    stop(e)
  })

  # Save outputs on success
  if (!is.null(model)) {
    saveRDS(model, file.path(results_dir, paste0(name, ".rds")))
    capture.output(summary(model), file = file.path(results_dir, paste0(name, "_summary.txt")))
    log_step(paste("Saved model and summary for", name))
  }

  list(model = model, error = err)
}

# ---------------- paths ----------------
script_dir <- get_script_dir()
data_dir <- normalizePath(file.path(script_dir, "..", "..", "data", "graphs", "only_connected"))
results_dir <- ensure_dir(file.path(script_dir, "results"))

log_step(paste("Script directory:", script_dir))
log_step(paste("Data directory:", data_dir))
log_step(paste("Results directory:", results_dir))

# ---------------- data load ----------------
log_step("Loading nodes and edges")
nodes <- read.csv(file.path(data_dir, "nodes.csv"), stringsAsFactors = FALSE)
edges <- read.csv(file.path(data_dir, "edges.csv"), stringsAsFactors = FALSE)

if (!all(c("u", "v") %in% names(edges))) stop("edges must have columns 'u' and 'v'")
if (!"artist_mbid" %in% names(nodes)) stop("nodes must have column 'artist_mbid'")
missing_required_cols <- setdiff(
  c(required_nodematch_attrs, required_numeric_attrs),
  names(nodes)
)
if (length(missing_required_cols) > 0) {
  stop(paste0(
    "nodes missing required columns: ",
    paste(missing_required_cols, collapse = ", ")
  ))
}

nodes_df <- as.data.frame(nodes, stringsAsFactors = FALSE)
edges_df <- as.data.frame(edges, stringsAsFactors = FALSE)

nodes_df$mbid <- trimws(as.character(nodes_df$artist_mbid))
edges_df$u <- trimws(as.character(edges_df$u))
edges_df$v <- trimws(as.character(edges_df$v))

if (any(is.na(edges_df$u) | edges_df$u == "")) {
  stop("edges have missing 'u' endpoints; no auto-fill allowed")
}
if (any(is.na(edges_df$v) | edges_df$v == "")) {
  stop("edges have missing 'v' endpoints; no auto-fill allowed")
}
if (any(is.na(nodes_df$mbid) | nodes_df$mbid == "")) {
  stop("nodes have missing 'artist_mbid' values; no auto-fill allowed")
}

log_step(sprintf("Nodes: %s, Edges: %s", nrow(nodes_df), nrow(edges_df)))

# If sampling, keep only a small induced subgraph to speed iteration
if (sample_graph) {
  set.seed(sample_seed)
  edge_ids <- unique(c(edges_df$u, edges_df$v))
  sampled <- FALSE
  for (attempt in seq_len(10)) {
    sample_ids <- sample(edge_ids, min(sample_nodes_target, length(edge_ids)))
    sample_edges <- edges_df[edges_df$u %in% sample_ids & edges_df$v %in% sample_ids, ]
    if (nrow(sample_edges) > 0) {
      edges_df <- sample_edges
      nodes_df <- nodes_df[nodes_df$mbid %in% sample_ids, ]
      sampled <- TRUE
      break
    }
  }
  if (!sampled && nrow(edges_df) > 0) {
    edge_sample_n <- min(sample_nodes_target, nrow(edges_df))
    edges_df <- edges_df[sample(seq_len(nrow(edges_df)), edge_sample_n), ]
    sample_ids <- unique(c(edges_df$u, edges_df$v))
    nodes_df <- nodes_df[nodes_df$mbid %in% sample_ids, ]
  }
  log_step(sprintf("Sampled graph: nodes=%s, edges=%s", nrow(nodes_df), nrow(edges_df)))
}

# Ensure vertex set covers all edge endpoints; create placeholder vertices for any missing IDs
nodes_df <- nodes_df %>% distinct(mbid, .keep_all = TRUE)
edge_ids <- unique(c(edges_df$u, edges_df$v))
missing_ids <- setdiff(edge_ids, nodes_df$mbid)
if (length(missing_ids) > 0) {
  log_step(sprintf("Adding %s placeholder vertices missing from nodes", length(missing_ids)))
  # Create placeholder rows with NA for other attributes
  placeholder <- data.frame(mbid = missing_ids, stringsAsFactors = FALSE)
  # Ensure placeholder has all node columns
  for (col in setdiff(names(nodes_df), names(placeholder))) {
    placeholder[[col]] <- NA
  }
  nodes_df <- bind_rows(nodes_df, placeholder)
}

# Require complete attributes for all ERGM terms; fail fast if any are missing.
missing_attr <- character(0)
for (attr in required_nodematch_attrs) {
  vals <- nodes_df[[attr]]
  if (any(is.na(vals) | vals == "")) {
    missing_attr <- c(missing_attr, attr)
  }
}
for (attr in required_numeric_attrs) {
  if (any(is.na(nodes_df[[attr]]))) {
    missing_attr <- c(missing_attr, attr)
  }
}
if (length(missing_attr) > 0) {
  missing_attr <- sort(unique(missing_attr))
  stop(paste0("Missing required node attributes (no auto-fill allowed): ", paste(missing_attr, collapse = ", ")))
}

nodes_df$name <- nodes_df$mbid
if (any(is.na(nodes_df$name) | nodes_df$name == "")) {
  stop("nodes have missing vertex names after augmentation; no auto-fill allowed")
}

# Final dedupe on name to avoid igraph duplicate-name error
nodes_df <- nodes_df %>% distinct(name, .keep_all = TRUE)
# Ensure name is the first column for igraph and character type
nodes_df$name <- as.character(nodes_df$name)
nodes_df <- nodes_df %>% select(name, everything())

log_step(sprintf("Total vertices after augmenting: %s", nrow(nodes_df)))

# ---------------- graph prep ----------------
log_step("Building igraph object")
g <- graph_from_data_frame(
  d = edges_df[, c("u", "v")],
  directed = FALSE,
  vertices = nodes_df
)
g <- simplify(g, remove.loops = TRUE, remove.multiple = TRUE)
V(g)$id <- V(g)$name  # explicit id attribute for network

log_step("Converting to statnet network")
net <- intergraph::asNetwork(g)

# ---------------- ERGM ladder ----------------
# Base terms
structural_terms <- c(
  "edges",
  "gwesp(0.5, fixed = TRUE)",
  "gwdegree(0.8, fixed = TRUE)"
)

# Model 0: density only
m0_terms <- c("edges")
m0_formula <- ergm_formula_from_terms(m0_terms)
m0 <- safe_fit(
  name = "m0_density",
  formula = m0_formula,
  net = net,
  results_dir = results_dir,
)

# Model 1: structural closure + degree
m1_formula <- ergm_formula_from_terms(structural_terms)
m1 <- safe_fit(
  name = "m1_structure",
  formula = m1_formula,
  net = net,
  results_dir = results_dir
)

# Model 2: add homophily terms
m2_terms <- structural_terms
for (attr in nodematch_attrs) {
  m2_terms <- c(m2_terms, sprintf('nodematch("%s")', attr))
}
m2_terms <- c(m2_terms, 'absdiff("time_std")')
m2_formula <- ergm_formula_from_terms(m2_terms)
m2 <- safe_fit(
  name = "m2_homophily",
  formula = m2_formula,
  net = net,
  results_dir = results_dir
)

# Model 3: add opportunity/exposure controls
exposure_covs <- c("collab_count_std", "num_songs_std")
m3_terms <- m2_terms
for (attr in exposure_covs) {
  m3_terms <- c(m3_terms, sprintf('nodecov("%s")', attr))
}
m3_formula <- ergm_formula_from_terms(m3_terms)
m3 <- safe_fit(
  name = "m3_exposure",
  formula = m3_formula,
  net = net,
  results_dir = results_dir
)

# Model 4: add weak-tie structural signal (open two-paths)
m4_terms <- c(m3_terms, "gwdsp(0.5, fixed = TRUE)")
m4_formula <- ergm_formula_from_terms(m4_terms)
m4 <- safe_fit(
  name = "m4_weak_ties",
  formula = m4_formula,
  net = net,
  results_dir = results_dir
)

# Track which models succeeded for downstream use
model_status <- data.frame(
  model = c("m0_density", "m1_structure", "m2_homophily", "m3_exposure", "m4_weak_ties"),
  success = c(!is.null(m0$model), !is.null(m1$model), !is.null(m2$model), !is.null(m3$model), !is.null(m4$model)),
  stringsAsFactors = FALSE
)
write.csv(model_status, file.path(results_dir, "model_status.csv"), row.names = FALSE)
log_step("Saved model_status.csv")

# ---------------- Node-level features ----------------
log_step("Computing node-level features")

n <- vcount(g)
node_ids <- V(g)$name

# Sparse adjacency + shared partners
A <- as_adjacency_matrix(g, sparse = TRUE)
SP <- A %*% A

# Edge embeddedness for weak-tie proxy
el <- as_edgelist(g, names = FALSE)
m <- nrow(el)
edge_embed <- numeric(m)
if (m > 0) {
  pb <- txtProgressBar(min = 0, max = m, style = 3)
  for (e in seq_len(m)) {
    i <- el[e, 1]; j <- el[e, 2]
    edge_embed[e] <- SP[i, j]
    if (e %% 5000 == 0) setTxtProgressBar(pb, e)
  }
  close(pb)
}

is_weak <- edge_embed <= 1

deg <- degree(g, mode = "all")
logdeg <- log1p(deg)

# Neighborhood-based stats
nb <- ego(g, order = 1)

# Community structure
cl <- cluster_louvain(g)
comm <- membership(cl)

participation_coef <- sapply(seq_len(n), function(i) {
  neigh <- setdiff(nb[[i]], i)
  if (length(neigh) == 0L) return(0)
  mean(comm[neigh] != comm[i])
})

# Clustering + triangles
clust_local <- transitivity(g, type = "local", isolates = "zero")
triangles_per_node <- count_triangles(g)
open_wedges <- choose(deg, 2) - triangles_per_node
open_wedges[deg < 2] <- 0

# Centrality
eig_cen <- eigen_centrality(g)$vector
pagerank <- page_rank(g)$vector
kcore <- coreness(g)
betw <- betweenness(g, directed = is_directed(g), normalized = TRUE)
close <- closeness(g, normalized = TRUE)

# Component size
comp <- components(g)
component_size <- comp$csize[comp$membership]

# Open dyad opportunities (two-hop minus one-hop)
twohop_unique <- lengths(neighborhood(g, order = 2)) - 1
open_dyads <- pmax(twohop_unique - deg, 0)

# Homophily shares where attributes exist
prop_same_attr <- function(gi, attr) {
  if (!has_vertex_attr(gi, attr)) return(rep(NA_real_, vcount(gi)))
  vals <- vertex_attr(gi, attr)
  sapply(seq_along(nb), function(i) {
    neigh <- setdiff(nb[[i]], i)
    if (length(neigh) == 0) return(NA_real_)
    mean(vals[neigh] == vals[i], na.rm = TRUE)
  })
}
same_genre_share <- prop_same_attr(g, "primary_genre")
same_label_share <- prop_same_attr(g, "primary_label")
same_role_share <- prop_same_attr(g, "primary_role")

df <- data.frame(
  node = node_ids,
  degree = deg,
  log_degree = logdeg,
  clust_local = clust_local,
  triangles = triangles_per_node,
  open_wedges = open_wedges,
  eigen_centrality = eig_cen,
  pagerank = pagerank,
  kcore = kcore,
  betweenness = betw,
  closeness = close,
  component_size = component_size,
  participation_coef = participation_coef,
  open_dyads = open_dyads,
  same_genre_share = same_genre_share,
  same_label_share = same_label_share,
  same_role_share = same_role_share,
  stringsAsFactors = FALSE
)

features_path <- file.path(data_dir, "node_features.csv")
write.csv(df, features_path, row.names = FALSE)
log_step(paste("Saved node_features.csv ->", features_path))

log_step("Done.")
