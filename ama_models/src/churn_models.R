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

churn_data <- read_csv("data/Churn Modeling.csv")
churn_data
summary(churn_data)

#### Visualizing data ####

DataExplorer::plot_intro(churn_data)

churn_data %>%
  ggplot(aes(x = Exited)) +
  geom_bar(color = "black", fill = "blue") +
  ggtitle("Distribución de variable de respuesta")

churn_data %>%
  ggplot(aes(group = Exited, y = CreditScore)) +
  geom_boxplot(fill = "purple") +
  ggtitle("Relación entre churn y credit score")

churn_data %>%
  ggplot(aes(group = Exited, y = Gender)) +
  geom_bar(fill = "purple") +
  ggtitle("Relación entre churn y sexo")

churn_data %>%
  ggplot(aes(x = Age)) +
  geom_boxplot(fill = "purple") +
  coord_flip() +
  facet_wrap(~Exited) +
  ggtitle("Relación entre churn y edad")

churn_data %>%
  ggplot(aes(x = Tenure)) +
  geom_boxplot(fill = "purple") +
  coord_flip() +
  facet_wrap(~Exited) +
  ggtitle("Relación entre churn y tenure")

churn_data %>%
  ggplot(aes(x = Balance)) +
  geom_boxplot(fill = "blue") +
  coord_flip() +
  facet_wrap(~Exited) +
  ggtitle("Relación entre churn y balance")

churn_data %>%
  ggplot(aes(x = NumOfProducts)) +
  geom_boxplot(fill = "red") +
  coord_flip() +
  facet_wrap(~Exited) +
  ggtitle("Relación entre churn y número de productos")

churn_data %>%
  ggplot(aes(group = Exited, y = NumOfProducts)) +
  geom_bar(fill = "purple") +
  coord_flip() +
  facet_wrap(~Exited) +
  ggtitle("Relación entre churn y número de productos")

churn_data %>%
  ggplot(aes(group = Exited, y = HasCrCard)) +
  geom_bar(fill = "purple") +
  coord_flip() +
  facet_wrap(~Exited) +
  ggtitle("Relación entre churn y tenencia de tarjeta")

churn_data %>%
  ggplot(aes(group = Exited, y = IsActiveMember)) +
  geom_bar(fill = "purple") +
  coord_flip() +
  facet_wrap(~Exited) +
  ggtitle("Relación entre churn y estatus de miembro activo")

churn_data %>%
  ggplot(aes(x = EstimatedSalary)) +
  geom_boxplot(fill = "red") +
  coord_flip() +
  facet_wrap(~Exited) +
  ggtitle("Relación entre churn y número de productos")




#### Spliting data ####

set.seed(220314)

split_data <- churn_data %>% initial_split(prop = 0.80, strata = Exited)

training_data <- training(split_data)
testing_data <- testing(split_data)
kfcv_data <- vfold_cv(training_data, v = 5)

#### Feature Engineering ####


feature_eng <- recipe(Exited ~ ., data = training_data) %>%
  update_role(RowNumber, new_role = "ID", old_role = "predictor") %>%
  update_role(CustomerId, new_role = "ID", old_role = "predictor") %>%
  update_role(Surname, new_role = "ID", old_role = "predictor") %>%
  step_mutate(Exited = as.factor(Exited)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ NumOfProducts:EstimatedSalary) %>%
  step_interact(~ CreditScore:EstimatedSalary)

juice(prep(feature_eng)) %>% glimpse()
bake(prep(feature_eng), testing_data) %>% glimpse()


#### Model definition ####

logistic_model <- logistic_reg(
  mode = "classification",
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

knn_model <- nearest_neighbor(
  mode = "classification",
  neighbors = tune("K"),
  dist_power = tune(),
  weight_func = tune()) %>%
  set_engine("kknn")

rforest_model <- rand_forest(
  mode = "classification",
  trees = 3000,
  mtry = tune(),
  min_n = tune()) %>%
  set_engine("ranger", importance = "impurity")


#### Workflowset definition ####

workflow_set_models <- workflow_set(
  preproc = list(rec = feature_eng),
  models = list(
    log = logistic_model,
    knn = knn_model,
    rf = rforest_model
    ),
  cross = TRUE
)

#### Fijación de parámetros ####

log_params <- logistic_model %>%
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
  option_add(param_info = log_params, id = c("rec_log"))


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
    metrics =  metric_set(roc_auc, precision, recall),
    control = grid_ctrl,
    verbose = TRUE
  )

stopCluster(cl)

tunning_models_result %>% saveRDS("cache/classification_workflowsets.rds")
tunning_models_result <- readRDS("cache/classification_workflowsets.rds")

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
  rank_metric = "roc_auc",  # <- how to order models
  metric = "roc_auc",       # <- which metric to visualize
  select_best = F
  ) +     # <- one point per workflow
  ggtitle("Model Comparisson") #+


tunning_models_result %>% autoplot(id = "rec_rf", metric = "roc_auc")

tunning_models_result %>% autoplot(id = "rec_knn", metric = "roc_auc")

tunning_models_result %>% autoplot(id = "rec_lm", metric = "roc_auc")

################################################################################

#### RANDOM fOREST ####
collect_metrics(tunning_models_result) %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean))

extract_workflow_set_result(tunning_models_result, "rec_rf") %>%
  autoplot(metric = "roc_auc")

autoplot(tunning_models_result, metric = "rmse")

extract_workflow_set_result(tunning_models_result, "rec_rf") %>%
        show_best(n = 10, metric = "roc_auc")


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

saveRDS(final_regularized_rf_model, "models/insurance_rf_model.rds")
final_regularized_rf_model <- readRDS("models/insurance_rf_model.rds")

################################################################################
#### Predictions ####

predict(
  final_regularized_rf_model,
  training_data) %>%
  dplyr::bind_cols(training_data) %>%
  mutate(error = .pred - Exited,
         mape = abs(error)/Exited,
         flag = if_else(mape > 0.4, "red", "blue"),
         id = row_number()) %>%
  ggplot(aes(.pred, Exited)) +
  geom_point(aes(color = flag)) +
  geom_abline()


insurance_predictions <- predict(
  final_regularized_rf_model,
  testing_data) %>%
  dplyr::bind_cols(testing_data) %>%
  mutate(error = .pred - Exited,
         mape = abs(error)/Exited,
         flag = if_else(mape > 0.4, "red", "blue"),
         id = row_number()
  )

insurance_predictions %>%
  ggplot(aes(.pred, Exited)) +
  geom_point(color = if_else(insurance_predictions$flag == "red", "red", "blue")) +
  geom_abline()

outliers <- insurance_predictions %>%
  filter(Exited > 10000, .pred < 20000, flag == "red")
outliers

insurance_predictions %>%
  ggplot(aes(Exited, error)) +
  geom_point()

final_regularized_rf_model %>%
  extract_fit_parsnip() %>%
  vip::vip(num_features = 20L, geom = "col") +
  ggtitle("Importancia de variables")

insurance_predictions %>%
  summarise(
    rmse = sqrt(mean(error^2)),
    mae = mean(abs(error)),
    mape = mean(abs(error)/Exited),
    rsq = cor(.pred, Exited)^2
  )

insurance_predictions %>%
  filter(!id %in% outliers$id) %>%
  summarise(
    rmse = sqrt(mean(error^2)),
    mae = mean(abs(error)),
    mape = mean(abs(error)/Exited),
    rsq = cor(.pred, Exited)^2
  )

