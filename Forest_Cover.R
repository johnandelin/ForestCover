#### Libraries ####
library(vroom)
library(tidyverse)
library(tidymodels)
library(lightgbm)
library(bonsai)
library(grid)


#### Loading the data
train <- vroom("train.csv")
test  <- vroom("test.csv")



# Convert outcome to factor
train <- train |> mutate(Cover_Type = factor(Cover_Type))

#### Data Cleaning Function
clean <- function(data){
  soil_mat <- data |> select(starts_with("Soil_Type")) |> as.matrix()
  wild_mat <- data |> select(starts_with("Wilderness")) |> as.matrix()
  
  data |> mutate(
    full_soil_type = factor(max.col(soil_mat, ties.method = "first")),
    full_wilderness_area = factor(max.col(wild_mat, ties.method = "first"))
  ) |> select(-starts_with("Soil"), -starts_with("Wilderness"))
}

clean_train <- clean(train)
clean_test  <- clean(test)

clean_train|>
  ggplot(aes(x = Cover_Type))+
  geom_bar(fill = c(1:7))+
  labs(
    title = "Count of Each Cover Type",
    x = "Cover Type",
    y = NULL
  ) 



clean_train |>
  ggplot(aes(x = full_soil_type, fill = full_soil_type)) +
  geom_bar() +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")  # top, right, bottom, left
  )+
  labs(
    title = "Soil Type by Number 1 - 40",
    x = NULL,
    y = NULL
  ) # missing 7 and 15



clean_train |>
  ggplot(aes(x = full_wilderness_area, fill = full_wilderness_area)) +
  geom_bar() +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")  # top, right, bottom, left
  )+
  labs(
    title = "Wilderness Type by Number 1 - 4",
    x = NULL,
    y = NULL
  ) 

#### Recipe with Feature Engineering
my_recipe <- recipe(Cover_Type ~ ., data = train) |>
  
  # Remove unwanted columns 
  step_rm(Id, Soil_Type7, Soil_Type15) |>
  
  # Hillshade engineered features
  step_mutate(
    Hillshade_Avg  = (Hillshade_9am + Hillshade_Noon + Hillshade_3pm) / 3,
    Hillshade_Diff = Hillshade_3pm - Hillshade_9am,
    Total_Hillshade = Hillshade_9am + Hillshade_Noon + Hillshade_3pm
  ) |>
  
  # Slope × Elevation
  step_mutate(Slope_Elev = Slope * Elevation) |>
  
  # Elevation & hydrology interactions
  step_mutate(
    Hydro_Plus_Elevation = Vertical_Distance_To_Hydrology + Elevation,
    Hydro_Dif_Elevation  = Vertical_Distance_To_Hydrology - Elevation
  ) |>
  
  # Distance combinations
  step_mutate(
    Hydro_Plus_Road = Horizontal_Distance_To_Hydrology + Horizontal_Distance_To_Roadways,
    Hydro_Dif_Road  = Horizontal_Distance_To_Hydrology - Horizontal_Distance_To_Roadways,
    Hydro_Dif_Fire  = Horizontal_Distance_To_Hydrology - Horizontal_Distance_To_Fire_Points,
    Hydro_Plus_Fire = Horizontal_Distance_To_Hydrology + Horizontal_Distance_To_Fire_Points,
    Hydro_Ratio = (Horizontal_Distance_To_Hydrology + 1) /
      (Vertical_Distance_To_Hydrology + 1),
    Hydro_Elevation = Vertical_Distance_To_Hydrology * Elevation
  ) |>
  
  # Signed vertical distance
  step_mutate(
    Signed_Vert_Hydro = sign(Vertical_Distance_To_Hydrology) *
      Horizontal_Distance_To_Hydrology
  ) |>
  
  # Steepness bins + dummies
  step_mutate(
    Steepness = case_when(
      Slope <= 10 ~ "Steep1",
      Slope <= 35 ~ "Steep2",
      TRUE ~ "Steep3"
    )
  ) |>
  step_mutate(Steepness = factor(Steepness)) |>
  step_dummy(Steepness) |>
  
  # Total hydrology distance
  step_mutate(
    Total_Hydro_distance = sqrt(
      Horizontal_Distance_To_Hydrology^2 +
        Vertical_Distance_To_Hydrology^2
    )
  ) |>
  
  # Aspect circular transform
  step_mutate(
    Aspect_sin = sin(Aspect * pi / 180),
    Aspect_cos = cos(Aspect * pi / 180)
  ) |>
  
  # Interaction between hydro features
  step_mutate(Vert_Horiz = Vertical_Distance_To_Hydrology * Horizontal_Distance_To_Hydrology)|>
  
  step_mutate(Fire_Road_Ratio = Horizontal_Distance_To_Fire_Points / (Horizontal_Distance_To_Roadways + 1))|>
  
  # Replace Inf with NA to be safe
  step_mutate_at(all_numeric_predictors(), fn = ~ ifelse(is.infinite(.x), NA, .x)) |>
  
  # Median impute numeric predictors if infinite values occur 
  step_impute_median(all_numeric_predictors()) |>
  
  # Remove zero variance predictors
  step_zv(all_predictors()) |>
  
  # Normalize numeric features
  step_normalize(all_numeric_predictors())



#### Model Specifications
# XGBoost - do NOT pass fractional mtry
xgb_spec <- boost_tree(
  trees = 2000,
  tree_depth = 12,
  learn_rate = 0.03,
  sample_size = 0.8
) |>
  set_engine("xgboost") |>
  set_mode("classification")

# LightGBM - do NOT pass fractional mtry
lgb_spec <- boost_tree(
  trees = 2000,
  tree_depth = 12,
  learn_rate = 0.03,
  sample_size = 0.8
) |>
  set_engine("lightgbm") |>
  set_mode("classification")


rf_spec <- rand_forest(
  trees = 1000,
  mtry = 8,   
  min_n = 2
) |>
  set_engine("ranger") |>
  set_mode("classification")

#### Workflows

xgb_wf <- workflow() |> add_recipe(my_recipe) |> add_model(xgb_spec)
lgb_wf <- workflow() |> add_recipe(my_recipe) |> add_model(lgb_spec)
rf_wf  <- workflow() |> add_recipe(my_recipe) |> add_model(rf_spec)

#### Fit Workflows

xgb_fit <- fit(xgb_wf, data = train)
lgb_fit <- fit(lgb_wf, data = train)
rf_fit  <- fit(rf_wf,  data = train)

#### Predict Probabilities on baked test set

xgb_probs <- predict(xgb_fit, new_data = test, type = "prob")
lgb_probs <- predict(lgb_fit, new_data = test, type = "prob")
rf_probs  <- predict(rf_fit,  new_data = test, type = "prob")

# ensure all are plain data.frames (numeric cols) for arithmetic

xgb_mat <- as.data.frame(xgb_probs)
lgb_mat <- as.data.frame(lgb_probs)
rf_mat  <- as.data.frame(rf_probs)


#### Soft Voting Ensemble (weights: 2,2,1)

probs_final <- (2 * xgb_mat + 2 * lgb_mat + 1 * rf_mat) / 5

# get predicted class
pred_index <- max.col(as.matrix(probs_final), ties.method = "first")
pred_class <- as.integer(gsub(".pred_", "", colnames(probs_final)[pred_index]))


#### Build Submission

submission <- tibble(
  Id = test$Id,
  Cover_Type = pred_class
)

vroom_write(submission, "./ForestPreds.csv", delim = ",")

