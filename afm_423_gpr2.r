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

# if you remove, make sure to remove labels from the data_ml sepreately

features_short <- select(data_ml, -c(stock_id, date, R1M_Usd)) %>% colnames()


data_ml <- add_credit_spread(data_ml, features_short)
data_ml <- add_vix(data_ml, features_short)



# Stock IDs and filtering
stock_ids <- levels(as.factor(data_ml$stock_id))
stock_days <- data_ml %>% group_by(stock_id) %>% summarize(nb = n())
stock_ids_short <- stock_ids[which(stock_days$nb == max(stock_days$nb))]







# Feature selection
features_short <- setdiff(colnames(data_ml), c("stock_id", "date", "R1M_Usd"))
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
K <- ncol(data_short_normalized)
input_dim <- K
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

# Determine the number of clusters using Fibonacci numbers
fibonacci <- function(n) {
  if (n <= 1) return(n)
  return(fibonacci(n - 1) + fibonacci(n - 2))
}

# Generate Fibonacci numbers for clustering
max_clusters <- 5  # Define the maximum number of clusters
fibonacci_numbers <- sapply(1:max_clusters, fibonacci)

print("Fibonacci numbers for clustering:")
print(fibonacci_numbers)
# Perform clustering for each Fibonacci number
cluster_results <- list()

for (num_clusters in fibonacci_numbers) {
  if (num_clusters > nrow(reduced_data)) break  # Skip if clusters exceed data points
  
  print(paste("Clustering with", num_clusters, "clusters"))
  
  # Perform clustering for each timestamp
  # reduced_data_with_date <- cbind(data_label$date, reduced_data)
  # print(colnames(reduced_data_with_date))

  final_data <- final_data %>%
    group_by(stock_id) %>%
    arrange(date) %>%
    ungroup()

  # colnames(reduced_data_with_date)[1] <- "date"
  
  # Group by date and perform clustering
  final_data$cluster <- final_data %>%
    group_by(date) %>%
    group_map(~ {
      kmeans_result <- kmeans(.x[-1], centers = num_clusters, nstart = 25)
      tibble(date = .x$date[1], cluster = kmeans_result$cluster)
    }) %>%
    bind_rows()

  final_data$cluster <- as.numeric(unlist(final_data$cluster))
  print(head(final_data%>% select(date, stock_id, cluster)))


  #TODO Create a portfolio using the cluster for each date, then backtest the portfolio
  # Also calculate the turnover of the portfolio
 

}


