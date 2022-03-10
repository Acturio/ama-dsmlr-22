library(plotly)
library(reshape2)
library(ggplot2)
library(tidyr)
library(MASS)

fit <- lm(Sepal.Length ~ Petal.Length, data = iris)

rls <- iris %>% 
  plot_ly(x = ~Petal.Length) %>% 
  add_markers(y = ~Sepal.Length) %>% 
  add_lines(x = ~Petal.Length, y = fitted(fit))

rls


htmlwidgets::saveWidget(rls, "rls.html")



my_df <- iris
petal_lm <- lm(Petal.Length ~ Sepal.Length + Sepal.Width, data = my_df)
graph_reso <- 0.05

#Setup Axis
axis_x <- seq(min(my_df$Sepal.Length), max(my_df$Sepal.Length), by = graph_reso)
axis_y <- seq(min(my_df$Sepal.Width), max(my_df$Sepal.Width), by = graph_reso)

#Sample points
petal_lm_surface <- expand.grid(Sepal.Length = axis_x,Sepal.Width = axis_y, KEEP.OUT.ATTRS = F)
petal_lm_surface$Petal.Length <- predict.lm(petal_lm, newdata = petal_lm_surface)
petal_lm_surface <- acast(petal_lm_surface, Sepal.Width ~ Sepal.Length, value.var = "Petal.Length") 

hcolors=c("red","blue","green")[my_df$Species]
iris_plot <- plot_ly(
  my_df, 
  x = ~Sepal.Length, 
  y = ~Sepal.Width, 
  z = ~Petal.Length,
  text = ~Species, # EDIT: ~ added
  type = "scatter3d", 
  mode = "markers",
  size = 0.05,
  marker = list(color = hcolors)
  )

iris_plot <- add_trace(
  p = iris_plot,
  z = petal_lm_surface,
  x = axis_x,
  y = axis_y,
  type = "surface"
  )

iris_plot
htmlwidgets::saveWidget(iris_plot, "iris_plot.html")

##################

set.seed(24601) 

covmat <- matrix(c(1.0,   0.2,   0.6, 
                   0.2,   1.0,  -0.5, 
                   0.6,  -0.5,   1.0), nrow=3) # the true cov matrix for my data
data <- mvrnorm(300, mu=c(0,0,0), Sigma=covmat) # generate random data that match that cov matrix
colnames(data) <- c("X1", "X2", "DV")
data <- as.data.frame(data)
data$group <- gl(n=3, k=ceiling(nrow(data)/3), labels=c("a", "b", "c", "d"))
data$DV <- with(data, ifelse(group=="c" & X1 > 0, DV+rnorm(n=1, mean=1), 
                             ifelse(group=="b" & X1 > 0, DV+rnorm(n=1, mean=2) , DV)))
data$DV <- ifelse(data$DV > 0, 1, 0)

b0 <- -0.5872841 # intercept
X1 <- 2.6508212
X2 <- -2.2599250
groupb <- 2.2110951
groupc <- 0.6649971
X1.X2 <- 0.1201166
X1.groupb <- 2.7323113
X1.groupc <- -0.6816327
X2.groupb <- 0.8476695
X2.groupc <- 0.4682893

X1_range <- seq(from=min(data$X1), to=max(data$X1), by=.01)
X2_val <- mean(data$X2)

a_logits <- b0 + 
  X1*X1_range + 
  X2*X2_val + 
  groupb*0 + 
  groupc*0 + 
  X1.X2*X1_range*X2_val + 
  X1.groupb*X1_range*0 + 
  X1.groupc*X1_range*0 + 
  X2.groupb*X2_val*0 + 
  X2.groupc*X2_val*0 

b_logits <- b0 + 
  X1*X1_range + 
  X2*X2_val + 
  groupb*1 + 
  groupc*0 + 
  X1.X2*X1_range*X2_val + 
  X1.groupb*X1_range*1 + 
  X1.groupc*X1_range*0 + 
  X2.groupb*X2_val*1 + 
  X2.groupc*X2_val*0

c_logits <- b0 + 
  X1*X1_range + 
  X2*X2_val + 
  groupb*0 + 
  groupc*1 + 
  X1.X2*X1_range*X2_val + 
  X1.groupb*X1_range*0 + 
  X1.groupc*X1_range*1 + 
  X2.groupb*X2_val*0 + 
  X2.groupc*X2_val*1

a_probs <- exp(a_logits)/(1 + exp(a_logits))
b_probs <- exp(b_logits)/(1 + exp(b_logits))
c_probs <- exp(c_logits)/(1 + exp(c_logits))

plot.data <- data.frame(a=a_probs, b=b_probs, c=c_probs, X1=X1_range)
plot.data <- gather(plot.data, key=group, value=prob, a:c) %>% 
  dplyr::sample_n(size = 600)

plot1 <- ggplot(plot.data, aes(x=X1, y=prob, color=group)) + 
 geom_jitter(size = 0.5, width = 0.05, height = 0.05) + 
 geom_line(lwd=0.5) + 
 labs(x="X1", y="P(outcome)") +
  theme(legend.title=element_blank())

rlogit <- plotly::ggplotly(plot1)
htmlwidgets::saveWidget(rlogit, "rlogit.html")

#########################


library(kableExtra)

prop <-  c(691, 639, 969, 955, 508)
total <- sum(prop)
props <- data.frame(x = prop, x_prop = prop/total) 
feature_eng1 <- props %>%
  kbl() %>%
  kable_classic_2(full_width = F)
htmlwidgets::saveWidget(feature_eng1, "feature_eng1.html")




edades <- data.frame(
 'Edad' = c(7, 78, 17, 25, NA), 
 'Grupo' = c('Niños', 'Adultos mayores', 'Adolescentes', 'Adultos', NA))

feature_eng2 <- edades %>% 
 kbl() %>%
 kable_classic_2(full_width = F)
htmlwidgets::saveWidget(feature_eng2, "feature_eng2.html")




edades <- data.frame(
 'Edad' = c(7, 78, NA, 25, NA), 
 'Grupo' = c('Niños', 'Adultos mayores', NA, 'Adultos', NA))

feature_eng2 <- edades %>% 
 kbl() %>%
 kable_classic_2(full_width = F)
htmlwidgets::saveWidget(feature_eng2, "feature_eng2.html")













