pacman::p_load(
 ProjectTemplate,
 doParallel,
 foreach,
 patchwork,
 vip,
 tidyverse,
 tidymodels,
 workflowsets,
 rules,
 mlflow,
 stacks,
 magrittr,
 jsonlite,
 naniar,
 DataExplorer,
 update = F
)

load.project()

##### load data #####

insurance_data <- read_csv("data/insurance.csv")
insurance_data
summary(insurance_data)

#### Visualizing data ####

DataExplorer::plot_intro(insurance_data)

insurance_data %>%
  ggplot(aes(x = charges)) +
  geom_histogram(color = "black", fill = "blue") +
  ggtitle("Distribución de variable de respuesta")

insurance_data %>%
  ggplot(aes(x = age, y = charges, color = sex)) +
  geom_point() +
  ggtitle("Relación entre edad y cargos (sex)")

insurance_data %>%
  ggplot(aes(x = age, y = charges, color = smoker)) +
  geom_point() +
  ggtitle("Relación entre edad y cargos (smoker)")

insurance_data %>%
  ggplot(aes(x = age, y = charges, color = region)) +
  geom_point() +
  ggtitle("Relación entre edad y cargos (región)")

insurance_data %>%
  ggplot(aes(x = children, y = charges, color = smoker)) +
  geom_point() +
  ggtitle("Relación entre hijos y cargos (smoker)")

insurance_data %>%
  ggplot(aes(x = bmi, y = charges, color = region, shape = smoker)) +
  geom_point() +
  geom_smooth(aes(x = age, y = charges)) +
  ggtitle("Relación entre bmi y cargos (smoker)")

insurance_data %>%
  ggplot(aes(x = smoker, y = charges, color = children)) +
  geom_point() +
  ggtitle("Relación entre hijos y cargos (smoker)")

insurance_data %>%
  ggplot(aes(x = region, y = charges)) +
  geom_point() +
  ggtitle("Relación entre hijos y cargos (smoker)")

#### Spliting data ####

set.seed(220314)

split_data <- insurance_data %>% initial_split(prop = 0.80, strata = charges)

training_data <- training(split_data)
testing_data <- testing(split_data)
kfcv_data <- vfold_cv(training_data, v = 5)

#### Feature Engineering ####


feature_eng <- recipe(charges ~ ., data = training_data) %>%
  step_mutate(
    young = ifelse(age < 30, 1, 0),
    children_cat = ifelse(children > 0, 1, 0)
    ) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ starts_with("sex"):starts_with("smoker")) %>%
  step_interact(~ bmi:starts_with("smoker")) %>%
  step_interact(~ bmi:starts_with("sex")) %>%
  step_interact(~ age:starts_with("smoker")) %>%
  step_interact(~ young:starts_with("smoker")) %>%
  step_interact(~ starts_with("children_cat"):starts_with("smoker"))

juice(prep(feature_eng)) %>% glimpse()
bake(prep(feature_eng), testing_data) %>% glimpse()


#### Model definition ####

linear_model <- linear_reg(
  mode = "regression",
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

knn_model <- nearest_neighbor(
  mode = "regression",
  neighbors = tune("K"),
  dist_power = tune(),
  weight_func = tune()) %>%
  set_engine("kknn")

rforest_model <- rand_forest(
  mode = "regression",
  trees = 3000,
  mtry = tune(),
  min_n = tune()) %>%
  set_engine("ranger", importance = "impurity")

gamma_model <- linear_reg(
  mode = "regression",
  penalty = 0) %>%
  set_engine("glmnet", family = Gamma(link = "inverse"))

#### Workflowset definition ####

workflow_set_models <- workflow_set(
  preproc = list(rec = feature_eng),
  models = list(
    gm = gamma_model,
    lm = linear_model,
    knn = knn_model,
    rf = rforest_model
    ),
  cross = TRUE
)

#### Fijación de parámetros ####

lm_params <- linear_model %>%
  parameters() %>%
  update(
    penalty = penalty(range = c(-5, 1), trans = log10_trans()),
    mixture = mixture(range = c(0, 1))
  )

knn_params <- knn_model %>%
  parameters() %>%
  update(
    K = dials::neighbors(c(25, 250)),
    dist_power = dist_power(range = c(1, 5)),
    weight_func = weight_func(values = c("rectangular", "gaussian", "cos", "euclidian"))
  )

rforest_params <- rforest_model %>%
  parameters() %>%
  update(
    mtry = mtry(range = c(1, 15)),
    min_n = min_n(range = c(2,40))
  )


#### GridSearch Definition ####

workflow_tunning_set_models <- workflow_set_models %>%
  option_add(param_info = knn_params, id = c("rec_knn")) %>%
  option_add(param_info = rforest_params, id = c("rec_rf")) %>%
  option_add(param_info = lm_params, id = c("rec_lm"))


grid_ctrl <- control_grid(
  save_pred = T,
  save_workflow = TRUE,
  parallel_over = "everything"
)

#### Tunning Multiple Models ####

cl <- makeCluster(( detectCores() - 1 ))
registerDoParallel(cl)

tunning_models_result <- workflow_tunning_set_models %>%
  workflow_map(
    fn = "tune_grid",
    seed = 134679,
    resamples = kfcv_data,
    grid = 50,
    metrics =  metric_set(rmse, rsq, mae, mape),
    control = grid_ctrl,
    verbose = TRUE
  )

stopCluster(cl)

tunning_models_result %>% saveRDS("cache/regression_workflowsets.rds")
tunning_models_result <- readRDS("cache/regression_workflowsets.rds")

################################################################################

#### Metrics ####

tunning_models_result$info
tunning_models_result$result

tunning_models_result %>%
  collect_metrics(summarize = T) %>%
  arrange(mean)

tunning_models_result %>%
  rank_results(select_best = T) %>%
  select(-c(.config, n, preprocessor, std_err)) %>%
  pivot_wider(names_from = .metric, values_from = mean)

autoplot(
  tunning_models_result,
  rank_metric = "rsq",  # <- how to order models
  metric = "rsq",       # <- which metric to visualize
  select_best = F
  ) +     # <- one point per workflow
  ggtitle("Model Comparisson") #+


tunning_models_result %>% autoplot(id = "rec_rf", metric = "rmse")
tunning_models_result %>% autoplot(id = "rec_rf", metric = "rsq")

tunning_models_result %>% autoplot(id = "rec_knn", metric = "rmse")
tunning_models_result %>% autoplot(id = "rec_knn", metric = "rsq")

tunning_models_result %>% autoplot(id = "rec_lm", metric = "rmse")
tunning_models_result %>% autoplot(id = "rec_lm", metric = "rsq")

################################################################################

#### RANDOM fOREST ####
collect_metrics(tunning_models_result) %>%
  filter(.metric == "rsq") %>%
  arrange(desc(mean))

extract_workflow_set_result(tunning_models_result, "rec_rf") %>%
  autoplot(metric = "rsq")

autoplot(tunning_models_result, metric = "rmse")

extract_workflow_set_result(tunning_models_result, "rec_rf") %>%
        show_best(n = 10, metric = "rmse")


best_regularized_rf_model_1se <- tunning_models_result %>%
    extract_workflow_set_result("rec_rf") %>%
    select_by_one_std_err(metric = "rmse", "rmse")

best_rf_model <- tunning_models_result %>%
 extract_workflow_set_result("rec_rf") %>%
 select_best(metric = "rmse", "rmse")

final_regularized_rf_model <- tunning_models_result %>%
  extract_workflow("rec_rf") %>%
  finalize_workflow(best_regularized_rf_model_1se) %>%
  parsnip::fit(data = training_data)

final_global_regularized_rf_model <- tunning_models_result %>%
  extract_workflow("rec_rf") %>%
  finalize_workflow(best_rf_model) %>%
  parsnip::fit(data = training_data)

saveRDS(final_regularized_rf_model, "cache/insurance_rf_model.rds")
final_regularized_rf_model <- readRDS("cache/insurance_rf_model.rds")

################################################################################
#### Predictions ####

predict(
  final_regularized_rf_model,
  training_data) %>%
  dplyr::bind_cols(training_data) %>%
  mutate(error = .pred - charges,
         mape = abs(error)/charges,
         flag = if_else(mape > 0.4, "red", "blue"),
         id = row_number()) %>%
  ggplot(aes(.pred, charges)) +
  geom_point(aes(color = flag)) +
  geom_abline() +
  ggtitle("Prediction VS Actual values") +
  xlab("Predictions") +
  ylab("Actual Charges")


insurance_predictions <- predict(
  final_regularized_rf_model,
  testing_data) %>%
  dplyr::bind_cols(testing_data) %>%
  mutate(error = .pred - charges,
         mape = abs(error)/charges,
         flag = if_else(mape > 0.4, "red", "blue"),
         id = row_number()
  )

prvsact <- insurance_predictions %>%
  ggplot(aes(.pred, charges)) +
  geom_point(color = if_else(insurance_predictions$flag == "red", "red", "blue")) +
  geom_abline() +
  ggtitle("Prediction VS Actual values") +
  xlab("Predictions") +
  ylab("Actual Charges")

plotly::ggplotly(prvsact)


outliers <- insurance_predictions %>%
  filter(charges > 10000, .pred < 20000, flag == "red")
outliers

insurance_predictions %>%
  ggplot(aes(charges, error)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red")

final_regularized_rf_model %>%
  extract_fit_parsnip() %>%
  vip::vip(num_features = 20L, geom = "col") +
  ggtitle("Importancia de variables")

insurance_predictions %>%
  summarise(
    rmse = sqrt(mean(error^2)),
    mae = mean(abs(error)),
    mape = mean(abs(error)/charges),
    rsq = cor(.pred, charges)^2
  )

# insurance_predictions %>%
#   filter(!id %in% outliers$id) %>%
#   summarise(
#     rmse = sqrt(mean(error^2)),
#     mae = mean(abs(error)),
#     mape = mean(abs(error)/charges),
#     rsq = cor(.pred, charges)^2
#   )

