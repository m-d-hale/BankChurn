
#This is a project to predict customer churn in a bank. The data is sourced from kaggle and is a dataset of a bank's customers. 
#The aim is to predict which customers are likely to leave the bank.

#Load libraries
###############

#get rid of all objects in the environment
#rm(list = ls())

options(timeout = 120)

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(httr)) install.packages("httr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
if(!require(Information)) install.packages("Information", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(vip)) install.packages("vip", repos = "http://cran.us.r-project.org")
if(!require(rattle)) install.packages("rattle", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(catboost)) install.packages("remotes", repos = "http://cran.us.r-project.org")
if(!require(catboost)) remotes::install_url('https://github.com/catboost/catboost/releases/download/v1.2.3/catboost-R-windows-x86_64-1.2.3.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))

library(tidyverse)
library(httr)
library(caret)
library(ggthemes)
library(ggplot2)
library(dplyr)
library(scales)
library(reshape2)
library(Metrics)
library(Information)
library(rpart)
library(rpart.plot)
library(vip)
library(rattle)
library(Rborist)
library(gbm)
library(xgboost)
library(catboost)


#Check versions of packages
installed_packages <- installed.packages()
package_versions <- installed_packages[, "Version"]
names(package_versions) <- rownames(installed_packages)
print(package_versions)

#Load the data sourced from kaggle
###################################

#Note the link to the dataset used is here: https://www.kaggle.com/datasets/rangalamahesh/bank-churn/data
#This was downloaded and saved alongside code/pdf/Rmd filess in the github repo here: https://github.com/m-d-hale/BankChurn

#Load the data (note, using the training data only given pre-created test set has no dependent/output/label var)
ChurnData <- read.csv('train.csv', stringsAsFactors = FALSE)

#Create a training and test set.
set.seed(123, sample.kind="Rounding")
test_index <- createDataPartition(y = ChurnData$Exited, times = 1, p = 0.2, list = FALSE)
df_train <- ChurnData[-test_index,]
df_test <- ChurnData[test_index,]

#Initial Data Exploration on training set
##########################################

# Get Basic Data Info for train/test initial sets
#################################################

#Rows, column frequencies
format(nrow(df_train), big.mark = ",", scientific = FALSE)
format(nrow(df_test), big.mark = ",", scientific = FALSE)
ncol(df_train)
head(df_train)

#Table to present variables in the training/test sets, with type and sample entries
df_train_rnd <- df_train %>% slice_sample(n=10) 
tmp <- capture.output(str(df_train_rnd))
tmp2 <- data.frame(tmp[2:length(tmp)])
colnames(tmp2) <- c("AllInfo")
tmp3 <- tmp2 %>% separate(col="AllInfo",into=c("v1","v2"),sep=":") %>% 
  mutate(Variable = substr(v1,3,nchar(v1)),
         Type = substr(v2,2,4),
         Examples = substr(v2,6,100)) %>%
  select(Variable,Type, Examples)
tmp3 %>% knitr::kable()

#Count of missing values per variable
tmp1 <- df_train %>% sapply(., function(x) sum(is.na(x))) %>% as.data.frame()
colnames(tmp1) <- c("MissingCount")
tmp1 %>% knitr::kable()


#List of numerical variables
numvars_all <- names(df_train)[sapply(df_train, is.numeric)]

#Create summary table for numerical variables
summary_table <- data.frame(Min = sapply(df_train[numvars_all], min, na.rm = TRUE),
                            Mean = round(sapply(df_train[numvars_all], mean, na.rm = TRUE),2),
                            Max = sapply(df_train[numvars_all], max, na.rm = TRUE))
summary_table %>% knitr::kable()

#Have a look at customer ID - are there any duplicates?
df_train %>% group_by(CustomerId) %>% summarize(n = n()) %>% filter(n > 1) %>% arrange(desc(n)) %>% head() %>% knitr::kable()
df_train %>% filter(CustomerId == 15565796) %>% knitr::kable()
#So... shows there are multiple customerIds in the dataset, but these are different customers with same ID.


#Plot histograms for numerical variables
########################################

#List of numerical variables
numvars <- c("CreditScore", "Age", "Balance", "EstimatedSalary")

#empty list to store histograms
hist_list <- list() 

#loop through vars in numvars list and produce histogram for each
for (i in 1:length(numvars)) {
  var <- numvars[i]
  bincalc <- (max(df_train[[var]]) - min(df_train[[var]]))/30
  hist_list[[i]] <- ggplot(df_train, aes_string(x=var)) + 
    geom_histogram(binwidth = bincalc, fill = "steelblue", color = "black") +
    labs(title = var, x = var, y = "Frequency") +
    theme_economist() +
    theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
          axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)))
}

#Print each of the histograms (NB: could do in loop of course, but this is to get each one to print in the Rmd file )
print(hist_list[[1]])
print(hist_list[[2]])
print(hist_list[[3]])
print(hist_list[[4]])


#Data Processing for Charts/Modelling
#####################################

#Band the age variable
df_train$AgeBand <- cut(df_train$Age, breaks = c(0, 18, 30, 40, 50, 60, 70, 80, 100), 
                        labels = c("0-18", "19-30", "31-40", "41-50", "51-60","61-70","71-80","81-100"))

#Band the balance variable
df_train$BalanceBand <- cut(df_train$Balance, breaks = c(-Inf, 0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, Inf), 
                            labels = c("<=0", "1-25k", "25k-50k", "50k-75k", "75k-100k", "100k-125k", "125k-150k", "150k-175k", "175k-200k", "200k+"))

#Band the credit score variable
df_train$CreditScoreBand <- cut(df_train$CreditScore, breaks = c(0, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000), 
                                labels = c("0-400", "401-450", "456-500", "501-550", "551-600", "601-650", "651-700", "701-750", "751-800", "801-850", "851-900", "901-950", "951-1000"))

#Band the salary variable
df_train$SalaryBand <- cut(df_train$EstimatedSalary, breaks = c(-Inf, 25000, 50000, 75000, 100000, 150000, 200000, Inf), 
                           labels = c("0-25k", "25k-50k", "50k-75k", "75k-100k", "100k-150k", "150k-200k","200k+"))


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

#Plot distributions and relationship to Exit Likelihood for character variables
#############################################################################

#Charting function
chrt_func <- function(var) {
  tmp <- df_train %>% 
    select(var, Exited) %>% 
    group_by_at(var) %>% 
    summarize(Vol = n(), ExitPct = mean(Exited))
  
  # Calculate the scalar for percentage and add it as a new column
  PctScalar = 1 / max(tmp$Vol)
  
  chrt <- tmp %>%
    ggplot() +
    geom_bar(aes_string(var, "Vol"), stat = "identity", fill = "steelblue") +
    geom_line(aes_string(var, paste("ExitPct /", PctScalar), group = 1), color = "red", size = 1.5) +
    scale_y_continuous(labels = scales::label_number_si(accuracy = 0.1), 
                       sec.axis = sec_axis(~.* PctScalar, name = "Average Exit Pct")) +
    ylab("Volume") +
    xlab(var) +
    theme_economist() +
    theme(axis.text.y.left = element_text(color = "steelblue"),
          axis.text.y.right = element_text(color = "red"),
          axis.title.y.left = element_text(color = "steelblue", margin = margin(t = 0, r = 10, b = 0, l = 0)),
          axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
          axis.title.y.right = element_text(color = "red", margin = margin(t = 0, r = 0, b = 0, l = 10)))
  
  return(chrt)
}

#List of character vars to plot
charvars <- c("Geography","Gender","Tenure","NumOfProducts","HasCrCard","IsActiveMember","AgeBand","BalanceBand","CreditScoreBand","SalaryBand")

#Plot each character var distribution and relationship to Exit Likelihood
chrt_func(charvars[1])
chrt_func(charvars[2])
chrt_func(charvars[3])
chrt_func(charvars[4])
chrt_func(charvars[5])
chrt_func(charvars[6])
chrt_func(charvars[7])
chrt_func(charvars[8]) + theme(axis.text.x = element_text(angle=90,hjust=1))
chrt_func(charvars[9]) + theme(axis.text.x = element_text(angle=90,hjust=1))
chrt_func(charvars[10]) 



#Exploratory Data Analysis
###########################

#Correlation Matrix
numvars_all2 <- names(df_train)[sapply(df_train, is.numeric)]
numvars_all2 <- numvars_all2[numvars_all2 != "id"]
numvars_all2 <- numvars_all2[numvars_all2 != "CustomerId"]

correlation_matrix <- round(cor(df_train[,numvars_all2]),2)
correlation_matrix

#Heatmap of Correlation Matrix

melted_cormat <- melt(correlation_matrix)
head(melted_cormat)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylab("") +
  xlab("")


#Gen just one half of correlation matrix
tri_fun <- function(cr_mt){
  cr_mt[lower.tri(cr_mt)]<- NA
  return(cr_mt)
}
upper_tri <- tri_fun(correlation_matrix)

# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)

# Heatmap
ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 90, vjust = 1, hjust = 1))+
  coord_fixed() +
  ylab("") +
  xlab("")


#Get information values for each variable (10 buckets)
IV <- df_train %>% mutate_if(is.character, as.factor) %>% 
  select(c(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, 
           IsActiveMember, EstimatedSalary, Exited )) %>%
  create_infotables(y="Exited", bins=10, parallel=TRUE)

IV_df = data.frame(IV$Summary)
IV_df %>% knitr::kable()

#Plot the IVs
plot_infotables(IV, IV$Summary$Variable, same_scale=FALSE)



#Experimenting with models in full training set
################################################


#Get a suite of measures to use to evaluate models
FitStats_func <- function(model_nm, prob,pred,act, act_num){
  
  #create confusion matrix
  confusion <- confusionMatrix(data=pred, reference=act, positive = "1")
  
  #confusion matrix driven stats
  accuracy <- confusion$overall['Accuracy']
  precision <- confusion$byClass['Precision']
  recall <- confusion$byClass['Recall']
  F1 <- confusion$byClass['F1']
  specificity <- confusion$byClass['Specificity']
  balancedAccuracy <- confusion$byClass['Balanced Accuracy']
  
  #Use predicted probabilities to get AUC (under different cut-offs) and log loss
  auc <- auc(act, prob)
  
  #Cap probabilities at marginally above 0 and marginally less than 1 to avoid log loss infinity calc issues
  epsilon <- 1e-15
  prob_capped <- pmin(pmax(prob, epsilon), 1 - epsilon)
  logloss <- logLoss(act_num, prob_capped)
  
  # Create a data frame with the statistics
  stats <- data.frame(ModelName = model_nm, Accuracy = accuracy, Precision = precision, Recall = recall, F1 = F1, 
                      Specificity = specificity, BalancedAccuracy = balancedAccuracy, AUC = auc, LogLoss = logloss)
  
  rownames(stats) <- NULL    #remove rownames
  
  return(stats)
}




#Get Exit var as factor
Exited_fct <- as.factor(df_train$Exited)


#glm (and rpart) needs character vars as factors, so create version of data with these as factors
df_train_fct <- df_train %>% mutate_if(is.character, as.factor)
df_train_fct$NumOfProducts <- as.factor(df_train_fct$NumOfProducts)
df_train_fct$HasCrCard <- as.factor(df_train_fct$HasCrCard)     
df_train_fct$IsActiveMember <- as.factor(df_train_fct$IsActiveMember)

#Group up NumOfProducts
df_train_fct$NumOfProducts2 <- ifelse(df_train_fct$NumOfProducts == "4" | df_train_fct$NumOfProducts == "3"  , 
                                      "3+", df_train_fct$NumOfProducts) %>% as.factor()

str(df_train_fct) #check vars

#K-fold Cross validation to get better view on out of sample performance, and to tune hyperparameters
k <- 5
grp_index <- createFolds(y = df_train_fct$Exited, k = k, list = TRUE, returnTrain = FALSE)

#NB: Clean up before running random forest; was having 'error in summary connection' issues; 
#seems like a parallel computing issue
#Function 'unregisters a doParallel cluster'
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}


#Model 0: Random Guess using mean of Exited as probability
#########

kfold_rand <- function(Description){
  for(i in 1:k){
    #Get the training and validation sets
    train_tmp <- df_train[-grp_index[[i]],]
    valid_tmp <- df_train[grp_index[[i]],]
    
    #Get mean prob of Exit and n
    p <- mean(train_tmp$Exited)   
    p_alt <- 1-p
    n <- nrow(valid_tmp)
    
    #Get actuals as factor for validation set
    Exit_fct_tmp <- as.factor(valid_tmp$Exited)
    
    #Gen probs in validation set and assign 1/0 based on mean prob
    set.seed(123, sample.kind="Rounding")
    tmpprob <- runif(n, min=0, max=1)
    tmppred <- ifelse(tmpprob > p_alt, 1, 0) %>%
      factor(levels=levels(Exit_fct_tmp))
    
    #Get the fit statistics
    tmp_stats <- FitStats_func(Description, tmpprob, tmppred, Exit_fct_tmp, valid_tmp$Exited)
    if(i == 1){
      stats_tbl <- tmp_stats
    } else {
      stats_tbl <- rbind(stats_tbl, tmp_stats)
    }
    
    if(i == k){
      stats_tbl <- stats_tbl %>% group_by(ModelName) %>% summarise_all(funs(mean))
    }
    
  }
  return(stats_tbl)
}

stats_tbl <- kfold_rand("Model 0: Above/Below average p")
stats_tbl_all <- stats_tbl



#Models 1&2: simple and multi-variate logistic regression
#########

#Decided to do feature selection using a forward selection algo 
#will loop through predictors and select best model based on BIC with 1,2,3 predictors etc.
#Initially using full training set, then will cross validate to get better view on out of sample performance

# Define the candidate predictors.
#NB: On initial run of the below, Ageband was a better predictor than age, so dropped numeric age variable
Predictors_Num <- c('CreditScore','Tenure','Balance','EstimatedSalary')
Predictors_Cat <- c('Geography','Gender','NumOfProducts2','HasCrCard','IsActiveMember','AgeBand','BalanceBand','CreditScoreBand','SalaryBand')

# Initialize a data frame to store the BIC for each predictor
BIC_df <- data.frame(predictor = character(), BIC = numeric())

# Loop over the numeric predictors
for (i in Predictors_Num) {
  formula <- as.formula(paste('Exited ~', i))
  model <- glm(formula, data = df_train_fct, family = binomial())
  BIC_df <- rbind(BIC_df, data.frame(predictor = i, BIC = BIC(model)))
}

# Loop over the categorical predictors
for (j in Predictors_Cat) {
  formula <- as.formula(paste('Exited ~', j))
  model <- glm(formula, data = df_train_fct, family = binomial())
  BIC_df <- rbind(BIC_df, data.frame(predictor = j, BIC = BIC(model)))
}

# Sort the data frame by BIC
BIC_df <- BIC_df[order(BIC_df$BIC), ]

# Select the predictor with the lowest BIC
ModPredsdf <- BIC_df[1,]
ModPreds <- BIC_df$predictor[1]

# Repeat the process for the remaining predictors
for (z in 1:10) {
  if (ModPreds[z] %in% Predictors_Cat) {
    Predictors_Cat <- setdiff(Predictors_Cat, ModPreds[z])
    ModPreds[z] <- paste('as.factor(', ModPreds[z], ')', sep = '')
  } else {
    Predictors_Num <- setdiff(Predictors_Num, ModPreds[z])
  }
  
  # Initialize a data frame to store the BIC for each predictor
  BIC_df2 <- data.frame(predictor = character(), BIC = numeric())
  
  # Loop over the remaining numeric predictors
  for (i in Predictors_Num) {
    formula <- as.formula(paste('Exited ~', paste(c(ModPreds, i), collapse = ' + ')))
    model <- glm(formula, data = df_train_fct, family = binomial())
    BIC_df2 <- rbind(BIC_df2, data.frame(predictor = i, BIC = BIC(model)))
  }
  
  # Loop over the remaining categorical predictors
  for (j in Predictors_Cat) {
    formula <- as.formula(paste('Exited ~', paste(c(ModPreds, paste('as.factor(', j, ')', sep = '')), collapse = ' + ')))
    model <- glm(formula, data = df_train_fct, family = binomial())
    BIC_df2 <- rbind(BIC_df2, data.frame(predictor = j, BIC = BIC(model)))
  }
  
  # Sort the data frame by BIC
  BIC_df2 <- BIC_df2[order(BIC_df2$BIC), ]
  
  # Select the predictor with the lowest BIC and add into BIC tracking df
  ModPredsdf <- rbind(ModPredsdf, BIC_df2[1, ])
  ModPreds <- c(ModPreds, BIC_df2$predictor[1])
  
}

# Print the data frame
ModPredsdf %>% knitr::kable()

#So, argument to use variables down to Estimated salary (9 variable model)
#Normally looking for 10 improvement in BIC to add a variable per Raftery 

#Plot the BIC for each predictor (keep ordering)
ModPredsdf %>% 
  mutate(Predictor = factor(predictor, levels = unique(predictor))) %>%
  ggplot(aes(x = Predictor, y = BIC)) +
  geom_point() +
  geom_line(group = 1) +
  theme_economist() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylab("BIC") +
  xlab("Predictor") +
  ggtitle("BIC for each logistic reg model when adding each predictor")

#Shows most of the value gain is achieved by the first 6 variables, with less gain after that


#Now k-fold the models with 1,2,3 etc vars and get fit stats in the validation sets
#Create a function first
Kfold_logi <- function(Description, Predictors, CutOff){
  for(i in 1:k){
    #Get the training and validation sets
    train_tmp <- df_train_fct[-grp_index[[i]],]
    valid_tmp <- df_train_fct[grp_index[[i]],]
    
    #Create a formula for the logistic regression model
    formula <- as.formula(paste("Exited ~", Predictors))
    tmpmod <- glm(formula, data = train_tmp, family = binomial)
    
    #Score the validation set using the model
    tmpprob <- predict(tmpmod, newdata = valid_tmp, type = "response")
    
    #Get actuals as factor for validation set
    Exit_fct_tmp <- as.factor(valid_tmp$Exited)
    
    #Get predicted probabilities of Exit, and convert to 1/0
    tmppred <- ifelse(tmpprob > CutOff, 1, 0) %>%
      factor(levels=levels(Exit_fct_tmp))
    
    #Get the fit statistics
    tmp_stats <- FitStats_func(Description, tmpprob, tmppred, Exit_fct_tmp, valid_tmp$Exited)
    if(i == 1){
      stats_tbl <- tmp_stats
    } else {
      stats_tbl <- rbind(stats_tbl, tmp_stats)
    }
    
    if(i == k){
      stats_tbl <- stats_tbl %>% group_by(ModelName) %>% summarise_all(funs(mean))
    }
    
  }
  return(stats_tbl)
}

#From trials with different vars, using the mean of Exited as the probability cut-off to start with

#Run kfold
stats_tbl <- Kfold_logi("Model 1a: Logistic using 1 var", ModPreds[1], 0.21)
stats_tbl_logi <- rbind(stats_tbl_logi, stats_tbl)

stats_tbl <- Kfold_logi("Model 1b: Logistic using 2 vars", paste(ModPreds[1:2], collapse = ' + '), 0.21)
stats_tbl_logi <- rbind(stats_tbl_logi, stats_tbl)

stats_tbl <- Kfold_logi("Model 1c: Logistic using 3 vars", paste(ModPreds[1:3], collapse = ' + '), 0.21)
stats_tbl_logi <- rbind(stats_tbl_logi, stats_tbl)

stats_tbl <- Kfold_logi("Model 1d: Logistic using 4 vars", paste(ModPreds[1:4], collapse = ' + '), 0.21)
stats_tbl_logi <- rbind(stats_tbl_logi, stats_tbl)

stats_tbl <- Kfold_logi("Model 1e: Logistic using 5 vars", paste(ModPreds[1:5], collapse = ' + '), 0.21)
stats_tbl_logi <- rbind(stats_tbl_logi, stats_tbl)

stats_tbl <- Kfold_logi("Model 1f: Logistic using 6 vars", paste(ModPreds[1:6], collapse = ' + '), 0.21)
stats_tbl_logi <- rbind(stats_tbl_logi, stats_tbl)

stats_tbl <- Kfold_logi("Model 1g: Logistic using 7 vars", paste(ModPreds[1:7], collapse = ' + '), 0.21)
stats_tbl_logi <- rbind(stats_tbl_logi, stats_tbl)

stats_tbl <- Kfold_logi("Model 1h: Logistic using 8 vars", paste(ModPreds[1:8], collapse = ' + '), 0.21)
stats_tbl_logi <- rbind(stats_tbl_logi, stats_tbl)

stats_tbl <- Kfold_logi("Model 1i: Logistic using 9 vars", paste(ModPreds[1:9], collapse = ' + '), 0.21)
stats_tbl_logi <- rbind(stats_tbl_logi, stats_tbl)

stats_tbl_logi %>% knitr::kable()

#Ok,so going to go with the 6 variable model. 
#Was the 'elbow' of the curve for BIC, and are only minor improvements in AUC, Log loss for adding more vars in
#Plus accuracy, F1 etc actually drop a bit for adding more vars in.
#Principle of model parsimony, so go for simpler model.

#Final step to optimise the cut-off for the 6 variable logistic reg model

CutOffs <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
Acc <- map_dbl(CutOffs, ~Kfold_logi("Model 1f: Logistic using 6 vars", paste(ModPreds[1:6], collapse = ' + '), .x)$Accuracy)
Acc 
#Plot
data.frame(CutOffs, Acc) %>%
  ggplot(aes(x = CutOffs, y = Acc)) +
  geom_point() +
  geom_line(group = 1) +
  theme_economist() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylab("Accuracy") +
  xlab("Cut-Offs") +
  ggtitle("Accuracy by Cut-Off for 6 variable model")

#So best cut-off for accuracy is around 0.5 on the 6-variable model according to the plot

#Optimise to get the 2dp figure (optimize function using line search method, similar to golden section)
neg_Acc <- function(cut_off) {
  -Kfold_logi("Model 1f: Logistic using 6 vars", paste(ModPreds[1:6], collapse = ' + '), cut_off)$Accuracy
}
round(optimize(neg_Acc, c(0.01, 0.99), tol=0.001)$minimum,2)

#So best cut-off according to this (to 2dp) is 0.53


#########################################################################################################
###Just run code below to get stats for best single var logistic reg and best multi-variate logistic ####
###to save time in producing the Rmd file ###############################################################
#########################################################################################################

#Manual list of best 6 predictors
SixPredsList <- c("as.factor(NumOfProducts2)","as.factor(AgeBand)","as.factor(IsActiveMember)",
                  "as.factor(Geography)","as.factor(Gender)","as.factor(BalanceBand)")

#Simple Logistic
stats_tbl <- Kfold_logi("Model 1: Simple Logistic: NumOfProducts", SixPredsList[1], 0.21)
stats_tbl_all <- rbind(stats_tbl_all, stats_tbl)

#6-var multivariate logistic
BestCutOff <- 0.53   #sourced from work above
stats_tbl <- Kfold_logi("Model 2: Multi-variate Logistic: NumOfProducts,Age,Member,Geography,Gender,Balance", paste(SixPredsList[1:6], collapse = ' + '), BestCutOff)
stats_tbl_all <- rbind(stats_tbl_all, stats_tbl)

# Print the model summary for model on all training data
formula <- as.formula(paste("Exited ~", paste(SixPredsList[1:6], collapse = ' + ')))
Logi6var <- glm(formula, data = df_train_fct, family = binomial)
summary(Logi6var)

#Ok, can see that there are a few places this could be improved - e.g. alteration of the age bandings (group sub 40) etc
#But in the main, doing a decent job, so will leave here and move on to other techniques.



#Model 3: Decision Trees
#########

#Start with a simple decision tree (CART). 
#Built using caret package, hyperparams tuned by resampling
#For Rpart only cp, the numeric complexity parameter, is tunable
modelLookup("rpart")

#Get data in shape for decision tree
df_train_tr <- df_train_fct %>% mutate(Exited_fct = as.factor(Exited)) %>% 
  select(-c('id','CustomerId','Surname','Exited','AgeBand','BalanceBand','CreditScoreBand','SalaryBand','NumOfProducts2'))

str(df_train_fct)

#v1. training using caret - simple to start with fixed complexity parameter of 0.01
unregister_dopar() #clean up parallel computing
tree_train_1 <- train(Exited_fct~., data=df_train_tr, method="rpart", trControl=trainControl(method="none"), tuneGrid = data.frame(cp = 0.01))
print(tree_train_1)

#Plot the tree
rpart.plot(tree_train_1$finalModel, type = 3, box.palette = "auto")

#Get ruleset
rules_1 <- rpart.rules(tree_train_1$finalModel, cover=TRUE)
print(rules_1)


#v2. Try again with cross-validation to get better cp
unregister_dopar() #clean up parallel computing
tree_train_2 <- train(Exited_fct~., data=df_train_tr, method="rpart", 
                    trControl=trainControl(method = "cv", number = 10), tuneGrid = data.frame(cp= seq(0.0001,0.001,len=10)))
print(tree_train_2)
plot(tree_train_2) #shows complexity parameter vs accuracy plot, so best at cp of 0.0003

#Confirmed best cp using below
optCP <- tree_train_2$bestTune
optCP

#Plot the tree
rpart.plot(tree_train_2$finalModel, type = 3, box.palette = "auto")
#so much more complex tree than the first one

#Get ruleset
rules_2 <- rpart.rules(tree_train_2$finalModel, cover=TRUE)
print(rules_2)

#Plot most important vars in the tree
var_importance <- vip::vip(tree_train_2, num_features = 10)
print(var_importance)
#So variable importance pretty minimal below the 6 main vars of Age, NumProducts, Active Member, Balance, Geography and Gender.


#Run own k-fold on the structure here (using opt cp from caret) to get metrics for simple and more complex trees
Kfold_tr <- function(Description, CompParam){
  for(i in 1:k){
    #Get the training and validation sets
    train_tmp <- df_train_tr[-grp_index[[i]],]
    valid_tmp <- df_train_tr[grp_index[[i]],]
    valid_tmp2 <- df_train_fct[grp_index[[i]],]
    
    #train the tree
    tmpmod <- train(Exited_fct~. , data=df_train_tr, method="rpart", trControl=trainControl(method="none"), tuneGrid = data.frame(cp = CompParam))
    
    #Score the validation set using the model
    tmpprob <- predict(tmpmod, newdata = valid_tmp, type = "prob")[,2]
    tmppred <- predict(tmpmod, newdata = valid_tmp, type = "raw")
  
    #Get actuals as factor for validation set
    Exit_fct_tmp <- as.factor(valid_tmp2$Exited)
    
    #Get the fit statistics
    tmp_stats <- FitStats_func(Description, tmpprob, tmppred, Exit_fct_tmp, valid_tmp2$Exited)
    if(i == 1){
      stats_tbl <- tmp_stats
    } else {
      stats_tbl <- rbind(stats_tbl, tmp_stats)
    }
    
    if(i == k){
      stats_tbl <- stats_tbl %>% group_by(ModelName) %>% summarise_all(funs(mean)) #average stats over the k validation sets
    }
    
  }
  return(stats_tbl)
}

#Run kfold
stats_tbl <- Kfold_tr("Model 3a: Simpler Decision Tree", 0.01)
stats_tbl_all <- rbind(stats_tbl_all, stats_tbl)

stats_tbl <- Kfold_tr("Model 3b: More Complex Decision Tree", optCP)
stats_tbl_all <- rbind(stats_tbl_all, stats_tbl)



#Model 4: Random Forest
#########

#Now try a random forest of decision trees

#Sample the data to get a smaller training set
set.seed(123)
df_train_tr_sample <- df_train_tr %>% slice_sample(n=5000)

#Initial random forest using caret and sample data to get idea of runtime and best tuning params
#Use 5-fold cross validation.
params <- expand.grid(.minNode = seq(25, 100, 25),
                        .predFixed = seq(from = 1, to = 5, by = 1))

rf_train <- train(Exited_fct~., 
                  data=df_train_tr_sample, 
                  method="Rborist",
                  trControl=trainControl(method="cv",number=5),
                  tuneGrid = params)
print(rf_train)
plot(rf_train)

# Get best hyperparameters and associated accuracy
best_params <- rf_train$bestTune
rf_train$results[rf_train$results$minNode == best_params$minNode & 
                               rf_train$results$predFixed == best_params$predFixed, ] %>% 
  select(minNode, predFixed, Accuracy) %>%
  knitr::kable()

#So in this case max accuracy achieved with minNode of 25 and predFixed of 3.


#Start with these params and train random forest on full training set
params <- expand.grid(.minNode = 25,
                      .predFixed = 3)
rf_train2 <- train(Exited_fct~., 
                  data=df_train_tr, 
                  method="Rborist",
                  trControl=trainControl(method="cv",number=5),
                  tuneGrid = params)
print(rf_train2)

#So, 5-fold accuracy here is 86.4%.So a bit under the complex decision tree which was 86.9%.
#See if adjusting the hyperparams is useful or not..
params <- expand.grid(.minNode = seq(15, 35, 10),
                      .predFixed = seq(from = 2, to = 4, by = 1))
rf_train3 <- train(Exited_fct~., 
                   data=df_train_tr, 
                   method="Rborist",
                   trControl=trainControl(method="cv",number=5),
                   tuneGrid = params)
print(rf_train3)
plot(rf_train3)
# Get best hyperparameters and associated accuracy
best_params <- rf_train3$bestTune
rf_train3$results[rf_train3$results$minNode == best_params$minNode & 
                    rf_train3$results$predFixed == best_params$predFixed, ] %>% 
  select(minNode, predFixed, Accuracy) %>%
  knitr::kable()

#Ok, marginal improvement in accuracy to 86.44% with minNode of 35 and predFixed of 4
#Check larger min node and larger predFixed

params <- expand.grid(.minNode = c(35,50,75),
                      .predFixed = seq(from = 3, to = 5, by = 1))
rf_train4 <- train(Exited_fct~., 
                   data=df_train_tr, 
                   method="Rborist",
                   trControl=trainControl(method="cv",number=5),
                   tuneGrid = params)
print(rf_train4)
plot(rf_train4)
best_params <- rf_train4$bestTune
rf_train4$results[rf_train4$results$minNode == best_params$minNode & 
                    rf_train4$results$predFixed == best_params$predFixed, ] %>% 
  select(minNode, predFixed, Accuracy) %>%
  knitr::kable()

#So another marginal improvement using minNode of 75 and predFixed of 4 to 86.5%.

#Try getting rid of the variables you know don't add much value, and see if that helps
#Feels like it should be more important for rf given the sampling of vars to use in each tree
#For df with all vars, then the least important ones just shouldn't feature in any trees, so should be less of an issue

unregister_dopar()
params <- expand.grid(.minNode = c(25,50,75),
                      .predFixed = seq(from = 2, to = 5, by = 1))

rf_train5 <- train(Exited_fct~ Age + NumOfProducts + IsActiveMember + Geography + Gender + Balance, 
                   data=df_train_tr, 
                   method="Rborist",
                   trControl=trainControl(method="cv",number=5),
                   tuneGrid = params)
print(rf_train5)
plot(rf_train5)
best_params <- rf_train5$bestTune
rf_train5$results[rf_train5$results$minNode == best_params$minNode & 
                    rf_train5$results$predFixed == best_params$predFixed, ] %>% 
  select(minNode, predFixed, Accuracy) %>%
  knitr::kable()

#Didn't improve things from an accuracy pov (86.3%). So stick with the full set of vars. 

#Final RF model:
params <- expand.grid(.minNode = 75,
                      .predFixed = 4)
rf_trainFin <- train(Exited_fct~., 
                   data=df_train_tr, 
                   method="Rborist",
                   trControl=trainControl(method="cv",number=5),
                   tuneGrid = params)
print(rf_trainFin)

best_params <- rf_trainFin$bestTune
rf_trainFin$results[rf_trainFin$results$minNode == best_params$minNode & 
                      rf_trainFin$results$predFixed == best_params$predFixed, ] %>% 
  select(minNode, predFixed, Accuracy) %>%
  knitr::kable()


#Run through own k-fold to get metrics for the best random forest model
Kfold_rf <- function(Description, MinNode, PredFixed){
  
  params <- expand.grid(.minNode = MinNode,
                        .predFixed = PredFixed)
  
  for(i in 1:k){
    #Get the training and validation sets
    train_tmp <- df_train_tr[-grp_index[[i]],]
    valid_tmp <- df_train_tr[grp_index[[i]],]
    valid_tmp2 <- df_train_fct[grp_index[[i]],]
    
    #train the rf
    tmpmod <- train(Exited_fct~., 
                    data=train_tmp, 
                    method="Rborist",
                    trControl=trainControl(method="none"),
                    tuneGrid = params)
    
    #Score the validation set using the model
    tmpprob <- predict(tmpmod, newdata = valid_tmp, type = "prob")[,2]
    tmppred <- predict(tmpmod, newdata = valid_tmp, type = "raw") 
      
    #Get actuals as factor for validation set
    Exit_fct_tmp <- as.factor(valid_tmp2$Exited)
    
    #Get the fit statistics
    tmp_stats <- FitStats_func(Description, tmpprob, tmppred, Exit_fct_tmp, valid_tmp2$Exited)
    if(i == 1){
      stats_tbl <- tmp_stats
    } else {
      stats_tbl <- rbind(stats_tbl, tmp_stats)
    }
    
    if(i == k){
      stats_tbl <- stats_tbl %>% group_by(ModelName) %>% summarise_all(funs(mean)) #average stats over the k validation sets
    }
    
  }
  return(stats_tbl)
}


#Run kfold
stats_tbl <- Kfold_rf("Model 4: Random Forest", 75, 4)

#So, looks decent on most metrics but not great on the log loss vs logistic reg and decision trees.
#Check why using one of the k-fold steps...

i <- 1

params <- expand.grid(.minNode = 75,
                      .predFixed = 4)

  #Get the training and validation sets
  train_tmp <- df_train_tr[-grp_index[[i]],]
  valid_tmp <- df_train_tr[grp_index[[i]],]
  valid_tmp2 <- df_train_fct[grp_index[[i]],]
  
  #train the rf
  tmpmod <- train(Exited_fct~., 
                  data=train_tmp, 
                  method="Rborist",
                  trControl=trainControl(method="none"),
                  tuneGrid = params)
  
  #Score the validation set using the model
  tmpprob <- predict(tmpmod, newdata = valid_tmp, type = "prob")[,2]
  tmppred <- predict(tmpmod, newdata = valid_tmp, type = "raw") 
  
  #Get actuals as factor for validation set
  Exit_fct_tmp <- as.factor(valid_tmp2$Exited)
  
  #Get the fit statistics
  tmp_stats <- FitStats_func("test rf", tmpprob, tmppred, Exit_fct_tmp, valid_tmp2$Exited)
  tmp_stats


  #So, poor log loss driven by predicted probabilities vs actuals...
  print(mean(tmpprob))
  print(mean(valid_tmp2$Exited))
  
  #And can see here the average probabilities are quite a long way off, even if the AUC and other metrics are ok.
  #So, seems to suggest the discrimination of the model is decent, but the accuracy of the probability estimates 
  #is off on the aggregate
  #Try scaling the probabilities, and re-running the k-fold...
  
  Kfold_rf2 <- function(Description, MinNode, PredFixed){
    
    params <- expand.grid(.minNode = MinNode,
                          .predFixed = PredFixed)
    
    for(i in 1:k){
      #Get the training and validation sets
      train_tmp <- df_train_tr[-grp_index[[i]],]
      valid_tmp <- df_train_tr[grp_index[[i]],]
      valid_tmp2 <- df_train_fct[grp_index[[i]],]
      
      #train the rf
      tmpmod <- train(Exited_fct~., 
                      data=train_tmp, 
                      method="Rborist",
                      trControl=trainControl(method="none"),
                      tuneGrid = params)
      
      #calibration model using training set to scale the probabilities
      probs <- predict(tmpmod, newdata = train_tmp, type = "prob")[,2]
      calibration_model <- glm(Exited_fct ~ probs, data = train_tmp, family = binomial)
      
      #Score the validation set using the model
      tmpprob <- predict(tmpmod, newdata = valid_tmp, type = "prob")[,2]
      tmppred <- predict(tmpmod, newdata = valid_tmp, type = "raw") 
      
      #Get actuals as factor for validation set
      Exit_fct_tmp <- as.factor(valid_tmp2$Exited)
      
      #Scale the predicted probabilities
      tmpprob_cali <- predict(calibration_model, newdata = data.frame(probs = tmpprob), type = "response")
      
      #Get the fit statistics
      tmp_stats <- FitStats_func(Description, tmpprob_cali, tmppred, Exit_fct_tmp, valid_tmp2$Exited)
      if(i == 1){
        stats_tbl <- tmp_stats
      } else {
        stats_tbl <- rbind(stats_tbl, tmp_stats)
      }
      
      if(i == k){
        stats_tbl <- stats_tbl %>% group_by(ModelName) %>% summarise_all(funs(mean)) #average stats over the k validation sets
      }
      
    }
    return(stats_tbl)
  }
  
  #Run kfold
  stats_tbl <- Kfold_rf2("Model 4: Random Forest", 75, 4)
  
  #Ok, much better on logloss and good discriminatory metrics still, stick with this version and add to table.
  stats_tbl_all <- rbind(stats_tbl_all, stats_tbl)


#Model 5: Support Vector Machine
#########

# Try on a small sample with all vars
set.seed(123)
df_train_tr_sample <- df_train_tr %>% slice_sample(n=5000)

start_time <- Sys.time()
svm_train <- train(Exited_fct ~ ., 
                   data = df_train_tr_sample, 
                   method = "svmRadial",
                   trControl=trainControl(method="cv",number=5))
print(svm_train)
plot(svm_train)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#Getting accuracy of about 85.5%; 15.1 seconds to run

#Try on small sample with subset of vars
start_time <- Sys.time()
svm_train2 <- train(Exited_fct ~ Age + NumOfProducts + IsActiveMember + Geography + Gender + Balance, 
                   data = df_train_tr_sample, 
                   method = "svmRadial",
                   trControl=trainControl(method="cv",number=5))
print(svm_train2)
plot(svm_train2)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime
#Accuracy at 85.5%,so same, but runtime of 13.7 seconds

#Increase the sample size and run again 
set.seed(123)
df_train_tr_sample <- df_train_tr %>% slice_sample(n=10000)
start_time <- Sys.time()
svm_train3 <- train(Exited_fct ~ Age + NumOfProducts + IsActiveMember + Geography + Gender + Balance, 
                    data = df_train_tr_sample, 
                    method = "svmRadial",
                    trControl=trainControl(method="cv",number=5))
print(svm_train3)
plot(svm_train3)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime
#Accuracy lower at 85.2%, and took just over 1 minute to run.

#Increase the sample size and run again 
set.seed(123)
df_train_tr_sample <- df_train_tr %>% slice_sample(n=15000)
start_time <- Sys.time()
svm_train4 <- train(Exited_fct ~ Age + NumOfProducts + IsActiveMember + Geography + Gender + Balance, 
                    data = df_train_tr_sample, 
                    method = "svmRadial",
                    trControl=trainControl(method="cv",number=5))
print(svm_train4)
plot(svm_train4)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime
#Accuracy up to 85.7%, and took just over 2 minutes to run.

?svmRadial

#Try tuning params
params <- expand.grid(sigma = c(0.05, 0.1, 0.15), C = c(1, 10, 100))
start_time <- Sys.time()
svm_train5 <- train(Exited_fct ~ Age + NumOfProducts + IsActiveMember + Geography + Gender + Balance, 
                    data = df_train_tr_sample, 
                    method = "svmRadial",
                    trControl=trainControl(method="cv",number=5),
                    tuneGrid = params)
print(svm_train5)
plot(svm_train5)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#Best tune
best_params <- svm_train5$bestTune
svm_train5$results[svm_train5$results$sigma == best_params$sigma & 
                     svm_train5$results$C == best_params$C, ] %>% 
  select(sigma, C, Accuracy) %>%
  knitr::kable()
#Ok, so sigma at 0.1 and C at 1 best - near where default tuning was; accuracy pretty similar at 85.8%
#But took 6x as long at 12 mins.
#So go back to default.

#Try upping the sample size to 20,000
set.seed(123)
df_train_tr_sample <- df_train_tr %>% slice_sample(n=20000)
start_time <- Sys.time()
svm_train6 <- train(Exited_fct ~ Age + NumOfProducts + IsActiveMember + Geography + Gender + Balance, 
                    data = df_train_tr_sample, 
                    method = "svmRadial",
                    trControl=trainControl(method="cv",number=5))
print(svm_train6)
plot(svm_train6)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime
#Accuracy at 85.9%, 4.5 minutes to run. So similar ish accuracy again.
#The final values used for the model were sigma = 0.1151251 and C = 0.5.



#Run through own k-fold to get metrics for this last SVM model
set.seed(123)
df_train_fct_sample <- df_train_fct %>% slice_sample(n=20000)
df_train_tr_sample <- df_train_fct_sample %>% mutate(Exited_fct = as.factor(make.names(Exited))) %>% 
  select(-c('id','CustomerId','Surname','Exited','AgeBand','BalanceBand','CreditScoreBand','SalaryBand','NumOfProducts2'))
grp_index_small <- createFolds(y = df_train_fct_sample$Exited, k = k, list = TRUE, returnTrain = FALSE)


#Create a function to run kfold svm
Kfold_svm <- function(Description, sigma_inp, c_inp){
  
  params <- expand.grid(sigma = sigma_inp, C = c_inp)
  
  for(i in 1:k){
    #Get the training and validation sets
    train_tmp <- df_train_tr_sample[-grp_index_small[[i]],]
    valid_tmp <- df_train_tr_sample[grp_index_small[[i]],]
    valid_tmp2 <- df_train_fct_sample[grp_index_small[[i]],]
    
    #train the svm
    tmpmod <- train(Exited_fct ~ Age + NumOfProducts + IsActiveMember + Geography + Gender + Balance, 
                    data = train_tmp, 
                    method = "svmRadial",
                    trControl=trainControl(method="none", classProbs = TRUE),
                    tuneGrid = params)
                    
    #Score the validation set using the model
    tmpprob <- predict(tmpmod, newdata = valid_tmp, type = "prob")[,2]
    tmppred <- predict(tmpmod, newdata = valid_tmp, type = "raw") 
    
    #Get actuals as factor for validation set
    Exit_fct_tmp <- as.factor(valid_tmp2$Exited)
    #convert predictions back to 0 and 1 factor (had to change to X0 X1 for the svmRadial training)
    levels(tmppred) <- c(0, 1)
  
    #Get the fit statistics
    tmp_stats <- FitStats_func(Description, tmpprob, tmppred, Exit_fct_tmp, valid_tmp2$Exited)
    if(i == 1){
      stats_tbl <- tmp_stats
    } else {
      stats_tbl <- rbind(stats_tbl, tmp_stats)
    }
    
    if(i == k){
      stats_tbl <- stats_tbl %>% group_by(ModelName) %>% summarise_all(funs(mean)) #average stats over the k validation sets
    }
    
  }
  return(stats_tbl)
}


#Run kfold
stats_tbl <- Kfold_svm("Model 5: SVM", 0.1151251, 0.5)
stats_tbl_all <- rbind(stats_tbl_all, stats_tbl)



#Model 6: Gradient Boosting
#########

set.seed(123)
df_train_gb_sample <- df_train_tr %>% slice_sample(n=5000)

start_time <- Sys.time()
gbm_train <- train(Exited_fct ~ ., 
                   data = df_train_gb_sample, 
                   method = "gbm",
                   trControl=trainControl(method="cv",number=5))
print(gbm_train)
plot(gbm_train)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#Best tune
best_params <- gbm_train$bestTune

gbm_train$results[gbm_train$results$n.trees == best_params$n.trees & 
                     gbm_train$results$interaction.depth == best_params$interaction.depth &
                    gbm_train$results$shrinkage == best_params$shrinkage &
                    gbm_train$results$n.minobsinnode == best_params$n.minobsinnode,] %>% 
  select(n.trees, interaction.depth, shrinkage, n.minobsinnode, Accuracy) %>%
  knitr::kable()

#Took 6 seconds to run the above, with best tuned model having 85.9% accuracy

#Try on larger sample
set.seed(123)
df_train_gb_sample <- df_train_tr %>% slice_sample(n=20000)

start_time <- Sys.time()
gbm_train2 <- train(Exited_fct ~ ., 
                   data = df_train_gb_sample, 
                   method = "gbm",
                   trControl=trainControl(method="cv",number=5))
print(gbm_train2)
plot(gbm_train2)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#Best tune
best_params <- gbm_train2$bestTune

gbm_train2$results[gbm_train2$results$n.trees == best_params$n.trees & 
                     gbm_train2$results$interaction.depth == best_params$interaction.depth &
                     gbm_train2$results$shrinkage == best_params$shrinkage &
                     gbm_train2$results$n.minobsinnode == best_params$n.minobsinnode,] %>% 
  select(n.trees, interaction.depth, shrinkage, n.minobsinnode, Accuracy) %>%
  knitr::kable()
#20.7 seconds and no improvement in accuracy. Same as the smaller sample at 85.9%


#Try with larger max tree depth and more trees
params <- expand.grid(n.trees = c(100, 200, 300),
                      interaction.depth = c(1, 2, 3),
                      shrinkage = c(0.1, 0.2, 0.3),
                      n.minobsinnode = c(10, 20, 30))

start_time <- Sys.time()
gbm_train3 <- train(Exited_fct ~ ., 
                   data = df_train_gb_sample, 
                   method = "gbm",
                   trControl=trainControl(method="cv",number=5),
                   tuneGrid = params)
print(gbm_train3)
plot(gbm_train3)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#Best tune
best_params <- gbm_train3$bestTune

gbm_train3$results[gbm_train3$results$n.trees == best_params$n.trees & 
                     gbm_train3$results$interaction.depth == best_params$interaction.depth &
                     gbm_train3$results$shrinkage == best_params$shrinkage &
                     gbm_train3$results$n.minobsinnode == best_params$n.minobsinnode,] %>% 
  select(n.trees, interaction.depth, shrinkage, n.minobsinnode, Accuracy) %>%
  knitr::kable()
#So gets to 86.1% accuracy, for about 5.5 minutes runtime.
#The final values used for the model were n.trees = 200, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

var_importance <- vip::vip(gbm_train3, num_features = 10)
print(var_importance)


#Run through own k-fold to get metrics for this last GBM model
Kfold_gbm <- function(Description, n_trees, int_depth, shrink, min_obs){
  
  params <- expand.grid(n.trees = n_trees,
                        interaction.depth = int_depth,
                        shrinkage = shrink,
                        n.minobsinnode = min_obs)
  
  for(i in 1:k){
    #Get the training and validation sets
    train_tmp <- df_train_tr[-grp_index[[i]],]
    valid_tmp <- df_train_tr[grp_index[[i]],]
    valid_tmp2 <- df_train_fct[grp_index[[i]],]
    
    #train the gbm
    tmpmod <- train(Exited_fct ~ ., 
                    data = train_tmp, 
                    method = "gbm",
                    trControl=trainControl(method="none"),
                    tuneGrid = params)
    
    #Score the validation set using the model
    tmpprob <- predict(tmpmod, newdata = valid_tmp, type = "prob")[,2]
    tmppred <- predict(tmpmod, newdata = valid_tmp, type = "raw") 
    
    #Get actuals as factor for validation set
    Exit_fct_tmp <- as.factor(valid_tmp2$Exited)
    
    #Get the fit statistics
    tmp_stats <- FitStats_func(Description, tmpprob, tmppred, Exit_fct_tmp, valid_tmp2$Exited)
    if(i == 1){
      stats_tbl <- tmp_stats
    } else {
      stats_tbl <- rbind(stats_tbl, tmp_stats)
    }
    
    if(i == k){
      stats_tbl <- stats_tbl %>% group_by(ModelName) %>% summarise_all(funs(mean)) #average stats over the k validation sets
    }
    
  }
  return(stats_tbl)
}

#Run kfold
stats_tbl <- Kfold_gbm("Model 6: GBM", 200, 3, 0.1, 10)
stats_tbl_all <- rbind(stats_tbl_all, stats_tbl)

#probs <- predict(model, newdata = df_test, type = "prob")


#XGBOOST
#########################
set.seed(123)
df_train_gb_sample <- df_train_tr %>% slice_sample(n=5000)

start_time <- Sys.time()
xgb_train <- train(Exited_fct ~ ., 
                   data = df_train_gb_sample, 
                   method = "xgbTree",
                   trControl=trainControl(method="cv",number=5))
print(xgb_train)
plot(xgb_train)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#Best tune
best_params <- xgb_train$bestTune

xgb_train$results[xgb_train$results$nrounds == best_params$nrounds & 
                    xgb_train$results$max_depth == best_params$max_depth &
                    xgb_train$results$eta == best_params$eta &
                    xgb_train$results$gamma == best_params$gamma &
                    xgb_train$results$colsample_bytree == best_params$colsample_bytree &
                    xgb_train$results$min_child_weight == best_params$min_child_weight &
                    xgb_train$results$subsample == best_params$subsample,] %>% 
  select(nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight, subsample, Accuracy) %>% knitr::kable()

#36 seconds to produce and 86% accuracy

#Try larger sample
set.seed(123)
df_train_gb_sample <- df_train_tr %>% slice_sample(n=20000)

start_time <- Sys.time()
xgb_train2 <- train(Exited_fct ~ ., 
                   data = df_train_gb_sample, 
                   method = "xgbTree",
                   trControl=trainControl(method="cv",number=5))
print(xgb_train2)
plot(xgb_train2)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#Best tune
best_params <- xgb_train2$bestTune

xgb_train2$results[xgb_train2$results$nrounds == best_params$nrounds & 
                     xgb_train2$results$max_depth == best_params$max_depth &
                     xgb_train2$results$eta == best_params$eta &
                     xgb_train2$results$gamma == best_params$gamma &
                     xgb_train2$results$colsample_bytree == best_params$colsample_bytree &
                     xgb_train2$results$min_child_weight == best_params$min_child_weight &
                     xgb_train2$results$subsample == best_params$subsample,] %>% 
  select(nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight, subsample, Accuracy) %>% knitr::kable()
#1.5 minutes, accuracy to 86.2%


#Try on the full set
start_time <- Sys.time()
xgb_train3 <- train(Exited_fct ~ ., 
                    data = df_train_tr, 
                    method = "xgbTree",
                    trControl=trainControl(method="cv",number=5))
print(xgb_train3)
plot(xgb_train3)
end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#Best tune
best_params <- xgb_train3$bestTune

xgb_train3$results[xgb_train3$results$nrounds == best_params$nrounds & 
                     xgb_train3$results$max_depth == best_params$max_depth &
                     xgb_train3$results$eta == best_params$eta &
                     xgb_train3$results$gamma == best_params$gamma &
                     xgb_train3$results$colsample_bytree == best_params$colsample_bytree &
                     xgb_train3$results$min_child_weight == best_params$min_child_weight &
                     xgb_train3$results$subsample == best_params$subsample,] %>% 
  select(nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight, subsample, Accuracy) %>% knitr::kable()
#About ten minutes to run, accuracy of 86.6%
#The final values used for the model were nrounds = 50, max_depth = 3, eta = 0.4, gamma = 0, colsample_bytree =
#0.6, min_child_weight = 1 and subsample = 1.

#Run through own k-fold to get metrics for this last XGBoost model
Kfold_xgb <- function(Description, nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight, subsample){
  
  params <- expand.grid(nrounds = nrounds,
                        max_depth = max_depth,
                        eta = eta,
                        gamma = gamma,
                        colsample_bytree = colsample_bytree,
                        min_child_weight = min_child_weight,
                        subsample = subsample)
  
  for(i in 1:k){
    #Get the training and validation sets
    train_tmp <- df_train_tr[-grp_index[[i]],]
    valid_tmp <- df_train_tr[grp_index[[i]],]
    valid_tmp2 <- df_train_fct[grp_index[[i]],]
    
    #train the xgb
    tmpmod <- train(Exited_fct ~ ., 
                    data = train_tmp, 
                    method = "xgbTree",
                    trControl=trainControl(method="none"),
                    tuneGrid = params)
    
    #Score the validation set using the model
    tmpprob <- predict(tmpmod, newdata = valid_tmp, type = "prob")[,2]
    tmppred <- predict(tmpmod, newdata = valid_tmp, type = "raw") 
    
    #Get actuals as factor for validation set
    Exit_fct_tmp <- as.factor(valid_tmp2$Exited)
    
    #Get the fit statistics
    tmp_stats <- FitStats_func(Description, tmpprob, tmppred, Exit_fct_tmp, valid_tmp2$Exited)
    if(i == 1){
      stats_tbl <- tmp_stats
    } else {
      stats_tbl <- rbind(stats_tbl, tmp_stats)
    }
    
    if(i == k){
      stats_tbl <- stats_tbl %>% group_by(ModelName) %>% summarise_all(funs(mean)) #average stats over the k validation sets
    }
    
  }
  return(stats_tbl)
}

#Run kfold
stats_tbl <- Kfold_xgb("Model 7: XGB", 50, 3, 0.4, 0, 0.6, 1, 1)
stats_tbl_all <- rbind(stats_tbl_all, stats_tbl)


#CATBOOST
##########

#Get data in shape for catboost
df_train_cbo <- df_train_fct %>% 
  select(-c('id','CustomerId','Surname','AgeBand','BalanceBand','CreditScoreBand','SalaryBand','NumOfProducts2'))

str(df_train_cbo)

#sample
set.seed(123)
df_train_cbo_sample <- df_train_cbo 
#%>% slice_sample(n=10000)

labels <- df_train_cbo_sample$Exited
labels2 <- as.factor(make.names(df_train_cbo_sample$Exited))
labelsdf <- df_train_cbo_sample %>% select('Exited')
featuresdf <- df_train_cbo_sample %>% select(-c('Exited'))

# Define the control parameters for the train function
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Define the grid of parameters to tune
grid <- expand.grid(depth = 8,
                    learning_rate = c(0.1, 0.11, 0.12),
                    iterations = 100,
                    l2_leaf_reg = c(0.001,0.0015,0.002),
                    rsm = 0.95,
                    border_count = 128)

start_time <- Sys.time()

# Train the model and tune the parameters
catboo_train2 <- train(featuresdf, labels2,
               method = catboost.caret,
               trControl = ctrl,
               tuneGrid = grid)

print(catboo_train2)
plot(catboo_train2)
max(catboo_train2$results$Accuracy)

end_time <- Sys.time()
runtime <- end_time - start_time
runtime

# So runtime of about 4 minutes for an 86.6% accuracy on first go
# Lots of tuning later, and still not far off 86.6% accuracy.
#The final values used for the model were depth = 8, learning_rate = 0.1, iterations = 100, l2_leaf_reg = 0.001, rsm = 0.95
#and border_count = 128.

# Just check with more folds...
ctrl <- trainControl(method = "cv", number = 20, classProbs = TRUE)

# Define the grid of parameters to tune
grid <- expand.grid(depth = 8,
                    learning_rate = 0.1,
                    iterations = 100,
                    l2_leaf_reg = 0.001,
                    rsm = 0.95,
                    border_count = 128)

start_time <- Sys.time()

# Train the model and tune the parameters
catboo_train3 <- train(featuresdf, labels2,
                       method = catboost.caret,
                       trControl = ctrl,
                       tuneGrid = grid)

print(catboo_train3)
max(catboo_train3$results$Accuracy)

end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#So, 86.6% again. Think we're close to the limit here... multiple techniques and 86.7% is about as good as its got.
#Last try with neural network

#First, do the k-fold for the catboost model, with params tuned from 5-fold version 
#(The final values used for the model were depth = 8, learning_rate = 0.1, iterations = 100, l2_leaf_reg = 0.001, rsm = 0.95
#and border_count = 128.)

#Run through own k-fold to get metrics for this last CatBoost model
df_train_cbo <- df_train_fct %>% 
  select(-c('id','CustomerId','Surname','AgeBand','BalanceBand','CreditScoreBand','SalaryBand','NumOfProducts2'))

Kfold_cbo <- function(Description, depth, learning_rate, iterations, l2_leaf_reg, rsm, border_count){
  
  params <- expand.grid(depth = depth,
                        learning_rate = learning_rate,
                        iterations = iterations,
                        l2_leaf_reg = l2_leaf_reg,
                        rsm = rsm,
                        border_count = border_count)
  
  for(i in 1:k){
    #Get the training and validation sets
    train_tmp <- df_train_cbo[-grp_index[[i]],]
    labels2_tmp <- as.factor(make.names(train_tmp$Exited))
    featuresdf_tmp <- train_tmp %>% select(-c('Exited'))
    
    valid_tmp <- df_train_cbo[grp_index[[i]],]
    valid_tmp2 <- df_train_fct[grp_index[[i]],]
    
    
    #train the catboost
    tmpmod <- train(featuresdf_tmp, labels2_tmp,
                          method = catboost.caret,
                          trControl=trainControl(method="none"),
                          tuneGrid = params)
    
    #Score the validation set using the model
    tmpprob <- predict(tmpmod, newdata = valid_tmp, type = "prob")[,2]
    tmppred <- predict(tmpmod, newdata = valid_tmp, type = "raw") 
    
    #Get actuals as factor for validation set
    Exit_fct_tmp <- as.factor(valid_tmp2$Exited)
    #convert predictions back to 0 and 1 factor (had to change to X0 X1 for the svmRadial training)
    levels(tmppred) <- c(0, 1)
    
    #Get the fit statistics
    tmp_stats <- FitStats_func(Description, tmpprob, tmppred, Exit_fct_tmp, valid_tmp2$Exited)
    if(i == 1){
      stats_tbl <- tmp_stats
    } else {
      stats_tbl <- rbind(stats_tbl, tmp_stats)
    }
    
    if(i == k){
      stats_tbl <- stats_tbl %>% group_by(ModelName) %>% summarise_all(funs(mean)) #average stats over the k validation sets
    }
    
  }
  return(stats_tbl)
}    

#Run kfold
stats_tbl <- Kfold_cbo("Model 8: CatBoost", 8, 0.1, 100, 0.001, 0.95, 128)
stats_tbl_all <- rbind(stats_tbl_all, stats_tbl)


#Model 7: Neural Network
#########
#NB: Need sigmoid function in the output layer to get probabilities

#Get data in shape for neural net
df_train_nn <- df_train_fct %>% 
  select(-c('id','CustomerId','Surname','AgeBand','BalanceBand','CreditScoreBand','SalaryBand','NumOfProducts2')) %>%
  mutate(Exited_fct = as.factor(make.names(Exited))) %>% select(-Exited)

levels(df_train_nn$Exited_fct) <- make.names(levels(df_train_nn$Exited_fct))

#sample
set.seed(123)
df_train_nn_sample <- df_train_nn %>% slice_sample(n=20000)

# Define the grid of parameters to tune
grid <- expand.grid(decay = c(0.9, 0.5, 0.3, 0.1, 0.001),
                    size = c(3, 5, 10))

#Number of folds for k-fold
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)

#Train neural net
start_time <- Sys.time()
nnet_train <- train(Exited_fct ~ ., 
                    data = df_train_nn, 
                    method = "nnet", 
                    trControl = ctrl,
                    tuneGrid = grid)

print(nnet_train)
plot(nnet_train)
max(nnet_train$results$Accuracy)

end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#Ok, not getting above 82% accuracy with the neural net. So, not as good as the other methods. And taking 20 mins to train.
#The final values used for the model were size = 10 and decay = 0.5


#Try pre-processing the predictors
start_time <- Sys.time()

#Pre-process the data (scale/normalise continuous vars - subtract mean and divide by sd)
preProcValues <- preProcess(df_train_nn, method = c("center", "scale"))
df_train_nn2 <- predict(preProcValues, df_train_nn)

# Define the grid of parameters to tune
grid <- expand.grid(decay = c(0.9, 0.5, 0.3, 0.1, 0.001),
                    size = c(3, 5, 10))

#Number of folds for k-fold
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)

nnet_train <- train(Exited_fct ~ ., 
                    data = df_train_nn2, 
                    method = "nnet", 
                    trControl = ctrl,
                    tuneGrid = grid)

print(nnet_train)
plot(nnet_train)
max(nnet_train$results$Accuracy)

end_time <- Sys.time()
runtime <- end_time - start_time
runtime

#Ok, preprocessing has a pretty large impact here, with accuracy up to 86.6%.
#Tree based methods you expect to be less impacted by scaling, but neural nets sensitive to it
#The final values used for the model were size = 10 and decay = 0.1


#Run K-fold for the best tuned neural net
Kfold_nn <- function(Description, size, decay){
  
  params <- expand.grid(decay = decay,
                        size = size)
  
  for(i in 1:k){
    #Get the training and validation sets
    train_tmp <- df_train_nn2[-grp_index[[i]],]
    valid_tmp <- df_train_nn2[grp_index[[i]],]
    valid_tmp2 <- df_train_fct[grp_index[[i]],]
    
    #train the neural net
    tmpmod <- train(Exited_fct ~ ., 
                    data = train_tmp, 
                    method = "nnet",
                    trControl=trainControl(method="none"),
                    tuneGrid = params)
    
    #Score the validation set using the model
    tmpprob <- predict(tmpmod, newdata = valid_tmp, type = "prob")[,2]
    tmppred <- predict(tmpmod, newdata = valid_tmp, type = "raw") 
    
    #Get actuals as factor for validation set
    Exit_fct_tmp <- as.factor(valid_tmp2$Exited)
    #convert predictions back to 0 and 1 factor (had to change to X0 X1 for the svmRadial training)
    levels(tmppred) <- c(0, 1)
    
    #Get the fit statistics
    tmp_stats <- FitStats_func(Description, tmpprob, tmppred, Exit_fct_tmp, valid_tmp2$Exited)
    if(i == 1){
      stats_tbl <- tmp_stats
    } else {
      stats_tbl <- rbind(stats_tbl, tmp_stats)
    }
    
    if(i == k){
      stats_tbl <- stats_tbl %>% group_by(ModelName) %>% summarise_all(funs(mean)) #average stats over the k validation sets
    }
    
  }
  return(stats_tbl)
}

#Run kfold
stats_tbl <- Kfold_nn("Model 9: Neural Net", 10, 0.1)
stats_tbl_all <- rbind(stats_tbl_all, stats_tbl)


#Decide which model to go with
##################################

#Save the table, so don't have a huge runtime to reproduce all modellign when the Rmd knits
save(stats_tbl_all, file = "stats_tbl_all.RData")

#Get stats all table to 3dp
stats_tbl_all_fin <- stats_tbl_all %>% mutate_if(is.numeric, round, 3) 
stats_tbl_all_fin %>% knitr::kable()

#Table with smaller model names
Newnames <- c("0_Base","1_LogReg_smpl","2_LogReg_cmplx","3a_Tree_simple",
              "3b_Tree_complex","4_RF","5_SVM","6_GBM","7_XGB","8_CatBoost","9_NNet")

stats_tbl_all %>% select(-ModelName) %>% cbind(., ModelName = Newnames) %>% 
  mutate_if(is.numeric, round, 3) %>% 
  select(ModelName,everything()) %>% knitr::kable()



#Plot the metrics
stats_tbl_all2 <- stats_tbl_all %>% select(-ModelName) %>% cbind(., ModelName = Newnames) %>%
  mutate(LogLoss_inv = 1- LogLoss) %>% select(-LogLoss) %>% filter(ModelName != "0_Base")

stats_tbl_all_t <- stats_tbl_all2 %>% gather("Metric","Value",-ModelName)

# Create line chart
Metrics_order <- colnames(stats_tbl_all2)
stats_tbl_all_t$Metric <- factor(stats_tbl_all_t$Metric, levels = Metrics_order)
selected_ModelName <- "7_XGB"

# Create chart
stats_tbl_all_t %>% ggplot(aes(x = Metric, y = Value, color = ModelName, group = ModelName)) +
  geom_line() +
  geom_line(data = subset(stats_tbl_all_t, ModelName == selected_ModelName), size = 1.5) +
  geom_point() +
  labs(x = "Metric", y = "Value") +
  theme_minimal() +
  theme(legend.text = element_text(size = 10),
        axis.title.x = element_text(margin = margin(t = 10)),
        axis.text.x = element_text(angle=90,hjust=1),
        axis.title.y = element_text(margin = margin(r = 10)))


#Go with XGBoost Does well across the board on both the discrimination metrics (separating exits from non exits) 
#as well as the accuracy of the probabilites too.

#Catboost performance might be marginally, marginally better, but 
#the training time is so much quicker for xgboost for very similar performance

#Re-write chart code above to highlight XGB performance


#Final Acid Test, using the XGBoost on the test set
####################################################

#Process test set per training set for consistency
df_test$AgeBand <- cut(df_test$Age, breaks = c(0, 18, 30, 40, 50, 60, 70, 80, 100), 
                        labels = c("0-18", "19-30", "31-40", "41-50", "51-60","61-70","71-80","81-100"))

#Band the balance variable
df_test$BalanceBand <- cut(df_test$Balance, breaks = c(-Inf, 0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, Inf), 
                            labels = c("<=0", "1-25k", "25k-50k", "50k-75k", "75k-100k", "100k-125k", "125k-150k", "150k-175k", "175k-200k", "200k+"))

#Band the credit score variable
df_test$CreditScoreBand <- cut(df_test$CreditScore, breaks = c(0, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000), 
                                labels = c("0-400", "401-450", "456-500", "501-550", "551-600", "601-650", "651-700", "701-750", "751-800", "801-850", "851-900", "901-950", "951-1000"))

#Band the salary variable
df_test$SalaryBand <- cut(df_test$EstimatedSalary, breaks = c(-Inf, 25000, 50000, 75000, 100000, 150000, 200000, Inf), 
                           labels = c("0-25k", "25k-50k", "50k-75k", "75k-100k", "100k-150k", "150k-200k","200k+"))


df_test_fct <- df_test %>% mutate_if(is.character, as.factor)
df_test_fct$NumOfProducts <- as.factor(df_test_fct$NumOfProducts)
df_test_fct$HasCrCard <- as.factor(df_test_fct$HasCrCard)     
df_test_fct$IsActiveMember <- as.factor(df_test_fct$IsActiveMember)
df_test_fct$NumOfProducts2 <- ifelse(df_test_fct$NumOfProducts == "4" | df_test_fct$NumOfProducts == "3"  , 
                                      "3+", df_test_fct$NumOfProducts) %>% as.factor()
df_test_tr <- df_test_fct %>% mutate(Exited_fct = as.factor(Exited)) %>% 
  select(-c('id','CustomerId','Surname','Exited','AgeBand','BalanceBand','CreditScoreBand','SalaryBand','NumOfProducts2'))


#Train the final model on the full training set
params <- expand.grid(nrounds = 50,
                      max_depth = 3,
                      eta = 0.4,
                      gamma = 0,
                      colsample_bytree = 0.6,
                      min_child_weight = 1,
                      subsample = 1)
  
#train the xgb
finmod <- train(Exited_fct ~ ., 
                data = df_train_tr, 
                method = "xgbTree",
                trControl=trainControl(method="none"),
                tuneGrid = params)

#Score the validation set using the model
finprob <- predict(finmod, newdata = df_test_tr, type = "prob")[,2]
finpred <- predict(finmod, newdata = df_test_tr, type = "raw") 
    
#Get actuals as factor for validation set
Exit_fct_tmp <- as.factor(df_test_fct$Exited)
    
#Get the fit statistics
stats_tbl_tst <- FitStats_func("Final Model: XGB", finprob, finpred, Exit_fct_tmp, df_test_fct$Exited)
stats_tbl_tst

#So similar performance on the test set vs the k-fold validation sets; slight drop in accuracy, but not much.

#Get variable importance plot
var_importance <- vip::vip(finmod, num_features = 10)
print(var_importance)



##############################################
##############################################
##############################################


#APPENDIX (code tried, but not eventually used)
##############################################

#CATBOOST OUTSIDE OF CARET
#########################

# Convert data to catboost pool format
cboost_Traindf = catboost.load_pool(data = featuresdf, 
                                    label = labels)

# Define the parameters for the catboost train
fit_params <- list(iterations = 100,
                   loss_function = 'Logloss',
                   border_count = 32,
                   depth = 5,
                   learning_rate = 0.03,
                   l2_leaf_reg = 3.5)

catboo_train <- catboost.train(cboost_Traindf, params = fit_params)

#get feature importance
catboost.get_feature_importance(catboo_train)

#temp validation set
set.seed(234)
df_valid_cbo_sample <- df_train_cbo %>% slice_sample(n=5000)

lbl_valid <- df_valid_cbo_sample$Exited
lbl_valid_fct <- as.factor(lbl_valid)
featdf_valid <- df_valid_cbo_sample %>% select(-c('Exited'))

cboost_validdf = catboost.load_pool(data = featdf_valid, 
                                    label = lbl_valid)

prob <- catboost.predict(model = catboo_cv, 
                         pool = cboost_validdf,
                         prediction_type = "Probability")

pred <- catboost.predict(model = catboo_cv, 
                         pool = cboost_validdf,
                         prediction_type = "Class")

pred_fct <- as.factor(pred)

confusion <- confusionMatrix(data = pred_fct, reference = lbl_valid_fct)
accuracy <- confusion$overall['Accuracy']
accuracy


params <- list(
  loss_function = 'Logloss',
  iterations = 100,
  depth = 6,
  learning_rate = 0.1,
  custom_metric = list('AUC', 'Accuracy')
)

# Convert data to catboost pool format
train_data <- catboost.load_pool(data = df_train_tr[, -1], label = df_train_tr[, 1])

# Train the model
model <- catboost.train(train_data, params = params)


#####################

#Check confusion matrix precision/recall etc manually to make sure working ok...
df_train %>% mutate(TN = ifelse(Exited_fct == 0 & pred1 == 0, 1, 0),
                   FP = ifelse(Exited_fct == 0 & pred1 == 1, 1, 0),
                   FN = ifelse(Exited_fct == 1 & pred1 == 0, 1, 0),
                   TP = ifelse(Exited_fct == 1 & pred1 == 1, 1, 0)) %>%
  summarise(TN = sum(TN), FP = sum(FP), FN = sum(FN), TP = sum(TP)) %>%
  mutate(Precision = TP / (TP + FP),
         Recall = TP / (TP + FN),
         F1 = 2 * Precision * Recall / (Precision + Recall))

#Confusion Matrix
confusion <- confusionMatrix(data=df_train$pred1, reference=df_train$Exited)

accuracy <- confusion$overall['Accuracy']
precision <- confusion$byClass['Precision']
recall <- confusion$byClass['Recall']
F1 <- confusion$byClass['F1']

# Compute the area under the ROC curve
roc_obj <- roc(df_train$true_ratings, df_train$predicted_ratings)
auc <- auc(roc_obj)

# Create a data frame with the statistics
stats <- data.frame(Accuracy = accuracy, Precision = precision, Recall = recall, F1 = F1, AUC = auc)

# Print the statistics
print(stats)












