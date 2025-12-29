suppressPackageStartupMessages({
  library(statnet)
  library(dplyr)
  library(intergraph)
  library(igraph)
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
  normalizePath(getwd())
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
  path
}

has_vertex_attr <- function(g, attr) attr %in% vertex_attr_names(g)

# ---------------- config ----------------
# Keep in sync with ergm_new_features.R sampling to avoid mismatched diagnostics.
sample_graph <- TRUE
sample_nodes_target <- 50
sample_seed <- 42

# Diagnostics toggles.
run_gof <- TRUE
run_mcmc_diagnostics <- TRUE
refit_mcmle_for_mcmc <- FALSE  # set TRUE if you want MCMC diagnostics on MCMLE fits

# GOF configuration (reduce nsim for speed on large graphs).
gof_nsim <- 50
gof_formula <- ~degree + espartners + dspartners

# MCMLE configuration (only used if refit_mcmle_for_mcmc = TRUE).
mcmle_control <- control.ergm(
  MCMC.interval = 1024,
  MCMC.samplesize = 4096,
  MCMLE.maxit = 20
)

# Attributes used in ERGM terms (for NA cleanup, same as fitting script).
required_nodematch_attrs <- c(
  "primary_genre", "primary_role",
  "artist_country", "primary_label",
  "artist_region_city"
)
required_numeric_attrs <- c("time_std", "collab_count_std", "num_songs_std")
required_source_attrs <- c("years_active", "unique_collaborator_count", "tracks_total")

# ---------------- paths ----------------
script_dir <- get_script_dir()
data_dir <- normalizePath(file.path(script_dir, "..", "..", "data", "graphs", "only_connected"))
results_dir <- ensure_dir(file.path(script_dir, "results"))
diag_dir <- ensure_dir(file.path(results_dir, "diagnostics"))
log_path <- file.path(diag_dir, "diagnostics_errors.log")

log_step(paste("Script directory:", script_dir))
log_step(paste("Data directory:", data_dir))
log_step(paste("Results directory:", results_dir))
log_step(paste("Diagnostics directory:", diag_dir))

# ---------------- data load / graph prep ----------------
log_step("Loading nodes and edges")
nodes <- read.csv(file.path(data_dir, "nodes.csv"), stringsAsFactors = FALSE)
edges <- read.csv(file.path(data_dir, "edges.csv"), stringsAsFactors = FALSE)

if (!all(c("u", "v") %in% names(edges))) stop("edges must have columns 'u' and 'v'")
if (!"artist_mbid" %in% names(nodes)) stop("nodes must have column 'artist_mbid'")
missing_required_cols <- setdiff(
  c(required_nodematch_attrs, required_source_attrs),
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
nodes_df$placeholder <- FALSE

log_step(sprintf("Nodes: %s, Edges: %s", nrow(nodes_df), nrow(edges_df)))

# Sampling for quick diagnostics, keep in sync with fitting script
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

# Add placeholder vertices for missing IDs
nodes_df <- nodes_df %>% distinct(mbid, .keep_all = TRUE)
edge_ids <- unique(c(edges_df$u, edges_df$v))
missing_ids <- setdiff(edge_ids, nodes_df$mbid)
if (length(missing_ids) > 0) {
  log_step(sprintf("Adding %s placeholder vertices missing from nodes", length(missing_ids)))
  placeholder <- data.frame(mbid = missing_ids, placeholder = TRUE, stringsAsFactors = FALSE)
  for (col in setdiff(names(nodes_df), names(placeholder))) {
    placeholder[[col]] <- NA
  }
  nodes_df <- bind_rows(nodes_df, placeholder)
}

# Fill missing categorical attributes with "unknown".
for (attr in required_nodematch_attrs) {
  vals <- nodes_df[[attr]]
  is_missing <- is.na(vals) | vals == ""
  if (any(is_missing)) {
    nodes_df[[attr]][is_missing] <- "unknown"
  }
}

non_placeholder <- !nodes_df$placeholder

validate_source <- function(vals, label) {
  if (any(is.na(vals))) stop(paste0("Missing values in required source column: ", label))
  if (any(vals < 0)) stop(paste0("Negative values in required source column: ", label))
}

zscore_log1p <- function(vals, label) {
  validate_source(vals, label)
  lvals <- log1p(vals)
  s <- sd(lvals)
  if (is.na(s) || s == 0) stop(paste0("Zero variance in log1p(", label, ")"))
  (lvals - mean(lvals)) / s
}

nodes_df$time_std <- NA_real_
nodes_df$collab_count_std <- NA_real_
nodes_df$num_songs_std <- NA_real_

nodes_df$time_std[non_placeholder] <- zscore_log1p(
  nodes_df$years_active[non_placeholder],
  "years_active"
)
nodes_df$collab_count_std[non_placeholder] <- zscore_log1p(
  nodes_df$unique_collaborator_count[non_placeholder],
  "unique_collaborator_count"
)
nodes_df$num_songs_std[non_placeholder] <- zscore_log1p(
  nodes_df$tracks_total[non_placeholder],
  "tracks_total"
)

# Assign neutral values for placeholder nodes to avoid NA in ERGM covariates.
nodes_df$time_std[!non_placeholder] <- 0
nodes_df$collab_count_std[!non_placeholder] <- 0
nodes_df$num_songs_std[!non_placeholder] <- 0

# Final vertex name setup for igraph
nodes_df$name <- nodes_df$mbid
if (any(is.na(nodes_df$name) | nodes_df$name == "")) {
  stop("nodes have missing vertex names after augmentation; no auto-fill allowed")
}
nodes_df <- nodes_df %>% distinct(name, .keep_all = TRUE)
nodes_df$name <- as.character(nodes_df$name)
nodes_df <- nodes_df %>% select(name, everything())

log_step(sprintf("Total vertices after augmenting: %s", nrow(nodes_df)))

log_step("Building igraph object")
g <- graph_from_data_frame(
  d = edges_df[, c("u", "v")],
  directed = FALSE,
  vertices = nodes_df
)
g <- simplify(g, remove.loops = TRUE, remove.multiple = TRUE)
V(g)$id <- V(g)$name

log_step("Converting to statnet network")
net <- intergraph::asNetwork(g)

# ---------------- model loading ----------------
model_names <- c("m0_density", "m1_structure", "m2_homophily", "m3_exposure", "m4_weak_ties")
models <- list()

for (name in model_names) {
  model_path <- file.path(results_dir, paste0(name, ".rds"))
  if (!file.exists(model_path)) {
    log_step(paste("Model not found, skipping:", model_path))
    next
  }
  log_step(paste("Loading model:", model_path))
  models[[name]] <- readRDS(model_path)
}

# ---------------- diagnostics ----------------
for (name in names(models)) {
  model <- models[[name]]
  log_step(paste("Diagnostics for", name))

  # MCMC diagnostics (only for MCMLE fits).
  if (run_mcmc_diagnostics) {
    diag_path <- file.path(diag_dir, paste0(name, "_mcmc_diagnostics.png"))
    tryCatch({
      if (refit_mcmle_for_mcmc) {
        log_step(paste("Refitting", name, "with MCMLE for MCMC diagnostics"))
        model_mcmc <- update(model, estimate = "MCMLE", control = mcmle_control)
      } else {
        model_mcmc <- model
      }

      png(diag_path, width = 1200, height = 900)
      mcmc.diagnostics(model_mcmc)
      dev.off()
      saveRDS(model_mcmc, file.path(diag_dir, paste0(name, "_mcmle_refit.rds")))
      log_step(paste("Saved MCMC diagnostics plot:", diag_path))
    }, error = function(e) {
      log_error(paste("MCMC diagnostics failed for", name, ":", conditionMessage(e)), log_path)
    })
  }

  # Simulation-based goodness of fit
  if (run_gof) {
    gof_path <- file.path(diag_dir, paste0(name, "_gof.rds"))
    gof_plot_path <- file.path(diag_dir, paste0(name, "_gof.png"))
    tryCatch({
      gof_fit <- gof(
        model,
        GOF = gof_formula,
        control = control.gof.ergm(nsim = gof_nsim)
      )
      saveRDS(gof_fit, gof_path)
      png(gof_plot_path, width = 1200, height = 900)
      plot(gof_fit)
      dev.off()
      log_step(paste("Saved GOF results:", gof_path))
    }, error = function(e) {
      log_error(paste("GOF failed for", name, ":", conditionMessage(e)), log_path)
    })
  }
}

log_step("Diagnostics complete.")
