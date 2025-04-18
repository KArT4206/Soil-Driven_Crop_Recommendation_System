# Load necessary libraries
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(corrplot)

# Load the dataset (adjust path to your local CSV file from Kaggle)
data <- read.csv("Crop_recommendation.csv")

# View basic info
str(data)
summary(data)

# Rename columns for easier handling (if needed)
colnames(data) <- c("Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall", "Crop")

# Check for missing values
colSums(is.na(data))

# Convert target variable to factor (for classification)
data$Crop <- as.factor(data$Crop)

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

# Plot distribution of soil nutrients
ggplot(data, aes(x=Nitrogen)) + geom_histogram(bins=30, fill="skyblue", color="black") + ggtitle("Nitrogen Distribution")
ggplot(data, aes(x=Phosphorus)) + geom_histogram(bins=30, fill="lightgreen", color="black") + ggtitle("Phosphorus Distribution")
ggplot(data, aes(x=Potassium)) + geom_histogram(bins=30, fill="salmon", color="black") + ggtitle("Potassium Distribution")

# Correlation matrix
cor_matrix <- cor(data[,1:7])
corrplot(cor_matrix, method = "circle")

# -----------------------------
# Regression Model (Predicting Crop Suitability Score)
# -----------------------------

# Example: Create a numeric 'suitability score' (mock for demonstration)
set.seed(100)
data$Suitability <- as.numeric(data$Crop) + rnorm(nrow(data), 0, 1)

# Multiple Linear Regression
model <- lm(Suitability ~ Nitrogen + Phosphorus + Potassium + pH + Temperature + Rainfall, data = data)
summary(model)

# -----------------------------
# Machine Learning: Random Forest Classification
# -----------------------------

# Split dataset into training and testing sets
set.seed(123)
index <- createDataPartition(data$Crop, p=0.8, list=FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]

# Train Random Forest model
rf_model <- randomForest(Crop ~ Nitrogen + Phosphorus + Potassium + Temperature + Humidity + pH + Rainfall,
                         data=train_data, ntree=100, importance=TRUE)

# Predict on test data
predictions <- predict(rf_model, test_data)

# Evaluate model
conf_matrix <- confusionMatrix(predictions, test_data$Crop)
print(conf_matrix)

# Feature Importance Plot
varImpPlot(rf_model)

# -----------------------------
# Hypothesis Testing
# -----------------------------

# One-Way ANOVA: Impact of pH on Suitability
anova_model <- aov(Suitability ~ cut(pH, breaks=3), data=data)
summary(anova_model)

# Chi-Square Test: Crop vs pH level groups
data$pH_group <- cut(data$pH, breaks=3, labels=c("Low", "Medium", "High"))
chisq.test(table(data$Crop, data$pH_group), simulate.p.value = TRUE, B = 10000)


# -----------------------------
# Save Model for Later Use (Optional)
# -----------------------------
# saveRDS(rf_model, "crop_rf_model.rds")
# loaded_model <- readRDS("crop_rf_model.rds")

