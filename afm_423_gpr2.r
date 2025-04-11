if(!require(quantmod)){install.packages("quantmod")}
if(!require(tidyverse)){install.packages("tidyverse")}
if(!require(keras)){install.packages("keras")}
if(!require(tensorflow)){install.packages("tensorflow")}
library(dplyr)
library(keras)
library(tensorflow)
library(quantmod)
library(tidyverse)

load("data_ml.RData")

# Helper function to normalize numeric columns
min_max_norm <- function(x) {
  if (is.numeric(x)) {
    return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
  } else {
    return(x)
  }
}

data_ml <- data_ml %>%
  select(-c(R3M_Usd, R6M_Usd, R12M_Usd))

add_vix <- function(data, features_short){

  getSymbols.FRED("VIXCLS", env = ".GlobalEnv", return.class = "xts")

  vix <- fortify(VIXCLS)
  colnames(vix) <- c("date", "vix")

  vix <- vix %>% # Take extraction and...
          full_join(data %>% dplyr::select(date), by = "date") %>% # merge data
          mutate(vix = na.locf(vix)) # Replace NA by previous
  vix <- vix[!duplicated(vix),] # Remove duplicates

  # data_cond <- data %>%                        
  #             dplyr::select(c("stock_id", "date", features_short, "R1M_Usd")) 

  data_cond <- data

  names_vix <- paste0(features_short, "_vix")    

  feat_vix <- data_cond %>%      
                    dplyr::select(all_of(features_short)) 
  vix <- data %>%       
                dplyr::select(date) %>% 
                left_join(vix, by = "date") 

  feat_vix <- feat_vix *      
                      matrix(vix$vix,       
                      length(vix$vix),       
                      length(features_short))                     
  colnames(feat_vix) <- names_vix  


  data_cond <- bind_cols(data_cond, feat_vix)   

  return(data_cond)

}

add_credit_spread <- function(data, features_short){

  getSymbols.FRED("BAMLC0A0CM",     
  env = ".GlobalEnv", 
  return.class = "xts")
  head(BAMLC0A0CM)

  cred_spread <- fortify(BAMLC0A0CM)     
  colnames(cred_spread) <- c("date", "spread")    
  cred_spread <- cred_spread %>%      
  full_join(data %>% dplyr::select(date), by = "date") %>%   
  mutate(spread = na.locf(spread))
  cred_spread <- cred_spread[!duplicated(cred_spread),]


  # data_cond <- data %>%                        
  #             dplyr::select(c("stock_id", "date", features_short, "R1M_Usd")) 

  data_cond <- data

  names_cred_spread <- paste0(features_short, "_cred_spread")    

  feat_cred_spread <- data_cond %>%      
                    dplyr::select(all_of(features_short)) 
  cred_spread <- data %>%       
                dplyr::select(date) %>% 
                left_join(cred_spread, by = "date") 

  feat_cred_spread <- feat_cred_spread *      
                      matrix(cred_spread$spread,       
                      length(cred_spread$spread),       
                      length(features_short))                     
  colnames(feat_cred_spread) <- names_cred_spread  


  data_cond <- bind_cols(data_cond, feat_cred_spread)   

  return(data_cond)

}

turnover <- function(weights, asset_returns, t_oos){
  turn <- 0
  for(t in 2:length(t_oos)){
    realised_returns <- returns %>% filter(date == t_oos[t]) %>% dplyr::select(-date)
    prior_weights <- weights[t-1,] * (1 + realised_returns) # Before rebalancing
    turn <- turn + apply(abs(weights[t,] - prior_weights/sum(prior_weights)),1,sum)
  }
  return(turn/(length(t_oos)-1))
}

perf_met <- function(portf_returns, weights, asset_returns, t_oos){
  avg_ret <- mean(portf_returns, na.rm = T)                     # Arithmetic mean 
  vol <- sd(portf_returns, na.rm = T)                           # Volatility
  Sharpe_ratio <- avg_ret / vol                                 # Sharpe ratio
  VaR_5 <- quantile(portf_returns, 0.05)                        # Value-at-risk
  turn <- 0                                                     # Initialisation of turnover
  for(t in 2:dim(weights)[1]){
    realized_returns <- asset_returns %>% filter(date == t_oos[t]) %>% dplyr::select(-date)
    prior_weights <- weights[t-1,] * (1 + realized_returns)
    turn <- turn + apply(abs(weights[t,] - prior_weights/sum(prior_weights)),1,sum)
  }
  turn <- turn/(length(t_oos)-1)                                # Average over time
  met <- data.frame(avg_ret, vol, Sharpe_ratio, VaR_5, turn)    # Aggregation of all of this
  rownames(met) <- "metrics"
  return(met)
}

perf_met_multi <- function(portf_returns, weights, asset_returns, t_oos, strat_name){
  J <- dim(weights)[2]              # Number of strategies
  met <- c()                        # Initialization of metrics
  for(j in 1:J){                    # One very ugly loop
    temp_met <- perf_met(portf_returns[, j], weights[, j, ], asset_returns, t_oos)
    met <- rbind(met, temp_met)
  }
  row.names(met) <- strat_name      # Stores the name of the strat
  return(met)
}

features_short <- select(data_ml, -c(stock_id, date, R1M_Usd)) %>% colnames()

data_ml <- add_credit_spread(data_ml, features_short)
data_ml <- add_vix(data_ml, features_short)

# Stock IDs and filtering
stock_ids <- levels(as.factor(data_ml$stock_id))
stock_days <- data_ml %>% group_by(stock_id) %>% summarize(nb = n())
stock_ids_short <- stock_ids[which(stock_days$nb == max(stock_days$nb))]

# Feature selection
print(paste("Number of features:", length(features_short)))
print(features_short)

data_short <- data_ml %>%
  filter(stock_id %in% stock_ids_short) %>%
  select(c("stock_id", "date", all_of(features_short), "R1M_Usd"))

data_short <- data_short %>%
  arrange(stock_id, date)

data_label <- data_short %>%
  select(stock_id, date, R1M_Usd)

# Normalize numeric features only (excluding stock_id and date)
data_short_numeric <- data_short %>% select(all_of(features_short), R1M_Usd)
data_short_normalized <- as.data.frame(lapply(data_short_numeric, min_max_norm))

# Model input dimensions
input_dim <- ncol(data_short_normalized)
print(paste("Input dimension:", input_dim))

# Define the autoencoder
encoding_dim <- 30

input_layer <- layer_input(shape = input_dim)

encoder <- input_layer %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = encoding_dim, activation = "relu")

decoder <- encoder %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = input_dim, activation = "linear")

autoencoder_model <- keras_model(inputs = input_layer, outputs = decoder)

autoencoder_model %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam"
)

# Fit model with error handling
tryCatch({
  history <- autoencoder_model %>% fit(
    x = as.matrix(data_short_normalized),
    y = as.matrix(data_short_normalized),
    epochs = 10,
    batch_size = 32,
    verbose = 1
  )

  encoder_model <- keras_model(inputs = input_layer, outputs = encoder)
  reduced_data <- encoder_model %>% predict(as.matrix(data_short_normalized))

  print("Autoencoder successfully trained!")
  print("Sample of reduced data:")
  print(head(reduced_data))

}, error = function(e) {
  print(paste("Error occurred:", e$message))
})

print("Autoencoder model summary:")
summary(autoencoder_model)

print("full data dimensions:")
print(dim(data_short_normalized))
head(data_short_normalized)
print("Sample of full data:")

print("reduced data dimensions:")
print(dim(reduced_data))
print("Sample of reduced data:")
print(head(reduced_data))

# Combine reduced data with stock_id and date from data_label
reduced_data_df <- as.data.frame(reduced_data)
colnames(reduced_data_df) <- paste0("feature_", 1:ncol(reduced_data_df))

final_data <- cbind(data_label, reduced_data_df)

# Print sample of the final combined dataset
print("Sample of final combined dataset:")
print(head(final_data))

# Generate Fibonacci numbers for clustering
max_clusters <- 5  # Define the maximum number of clusters

results <- list()

for (num_clusters in 1:max_clusters) {
  if (num_clusters > nrow(reduced_data)) break  # Skip if clusters exceed data points
  
  print(paste("Clustering with", num_clusters, "clusters"))

  # Perform clustering for each date
  final_data <- final_data %>%
    group_by(stock_id) %>%
    arrange(date) %>%
    ungroup()

  # Group by date and perform clustering
  final_data$cluster <- final_data %>%
    group_by(date) %>%
    group_map(~ {
      kmeans_result <- kmeans(.x[-1], centers = num_clusters, nstart = 25)
      tibble(date = .x$date[1], cluster = kmeans_result$cluster)
    }) %>%
    bind_rows()

  final_data$cluster <- as.numeric(unlist(final_data$cluster))

  returns <- final_data %>%                           # Compute returns, in matrix format, in 3 steps:
    filter(stock_id %in% stock_ids_short) %>%    # 1. Filtering the data
    dplyr::select(date, stock_id, R1M_Usd) %>%   # 2. Keep returns along with dates & firm names
    spread(key = stock_id, value = R1M_Usd)      # 3. Put in matrix shape 
  sep_oos <- as.Date("2007-01-01")                            # Starting point for backtest
  ticks <- final_data$stock_id %>%                               # List of all asset ids
    as.factor() %>%
    levels()
  N <- length(ticks)                                          # Max number of assets
  t_oos <- returns$date[returns$date > sep_oos] %>%           # Out-of-sample dates 
    unique() %>%                                            # Remove duplicates
    as.Date(origin = "1970-01-01")                          # Transform in date format
  Tt <- length(t_oos)                                         # Nb of dates, avoid T = TRUE
  nb_port <- 1                                                # Nb of portfolios/stragegies
  portf_weights_list <- list()
  portf_returns_list <- list()

  portf_compo <- function(train_data, test_data){
    N <- test_data$stock_id %>%             # Test data dictates allocation
      factor() %>% nlevels()
    w <- 1/N                                # EW portfolio
    w$weights <- rep(w,N)
    w$names <- unique(test_data$stock_id)   # Asset names
    return(w)
  }
  
  m_offset <- 12                                          # Offset in months for buffer period
  train_size <- 5                                         # Size of training set in years
  
  for (cluster_id in 1:num_clusters){
    cluster_data <- final_data %>% filter(cluster == cluster_id)  # Filter for that cluster_id
    
    # Recalculate since number of stocks may vary
    ticks <- unique(cluster_data$stock_id)
    N <- length(ticks)
    
    # Initialize per-cluster storage
    portf_weights <- array(0, dim = c(Tt, nb_port, N))          # Initialize portfolio weights
    portf_returns <- matrix(0, nrow = Tt, ncol = nb_port)       # Initialize portfolio returns

    for(t in 1:(length(t_oos)-1)){                          # Stop before last date: no fwd ret.!
      if(t%%12==0){print(t_oos[t])}                       # Just checking the date status

      train_data <- cluster_data %>% filter(date < t_oos[t] - m_offset * 30,   # Roll window w. buffer
                                       date > t_oos[t] - m_offset * 30 - 365 * train_size)
      
      test_data <- cluster_data %>% filter(date == t_oos[t])   # Test sample
      realized_returns <- test_data$R1M_Usd

      for(j in 1:nb_port){
        temp_weights <- portf_compo(train_data, test_data)
        ind <- match(temp_weights$names, ticks) %>% na.omit()
        portf_weights[t,j,ind] <- temp_weights$weights
        portf_returns[t,j] <- sum(temp_weights$weights * realized_returns)
      }
    }

    portf_weights_list[[as.character(cluster_id)]] <- portf_weights
    portf_returns_list[[as.character(cluster_id)]] <- portf_returns
  }
  
  asset_returns <- data_ml %>%                          # Compute return matrix: start from data
    dplyr::select(date, stock_id, R1M_Usd) %>%        # Keep 3 attributes 
    spread(key = stock_id, value = R1M_Usd)           # Shape in matrix format
  asset_returns[is.na(asset_returns)] <- 0              # Zero returns for missing points
  
  metrics_list <- list()
  
  for (cluster_id in 1:num_clusters) {
    print(paste(" Metrics for Cluster", cluster_id, ":"))
    
    current_assets <- as.character(unique(final_data$stock_id[final_data$cluster == cluster_id]))
    
    asset_returns_cluster <- asset_returns %>%
      dplyr::select(date, all_of(current_assets))

    met <- perf_met_multi(
      portf_returns = portf_returns_list[[cluster_id]],
      weights = portf_weights_list[[cluster_id]],
      asset_returns = asset_returns,
      t_oos = t_oos,
      strat_name = c("EW")  # You can customize if you have multiple strategies
    )
    
    print(met)
    metrics_list[[cluster_id]] <- met
  }
}
