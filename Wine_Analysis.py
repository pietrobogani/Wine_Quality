import pandas as pd

wine_red = pd.read_csv("winequality-red.csv", sep=";")
wine_white = pd.read_csv("winequality-white.csv", sep=";")

# Add a column of 0 to the red wine dataset and a column of 1 to the white wine dataset
wine_red['new_column'] = 0
wine_white['new_column'] = 1

# Merge the two datasets
wine = pd.concat([wine_white, wine_red], ignore_index=True)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Check the frequency of the rates for red wine and produce a bar plot
frequency_table_red = wine_red['quality'].value_counts()
percentage_red = (frequency_table_red / len(wine_red)) * 100

axes[0].bar(frequency_table_red.index, frequency_table_red, color='lightblue')
axes[0].set_title("Frequency of Quality (Red Wine)")
axes[0].set_xlabel("Quality")
axes[0].set_ylabel("Frequency")

# Check the frequency of the rates for white wine and produce a bar plot
frequency_table_white = wine_white['quality'].value_counts()
percentage_white = (frequency_table_white / len(wine_white)) * 100

axes[1].bar(frequency_table_white.index, frequency_table_white, color='lightblue')
axes[1].set_title("Frequency of Quality (White Wine)")
axes[1].set_xlabel("Quality")
axes[1].set_ylabel("Frequency")


# Check the frequency of the rates for both wines at the same time and produce a bar plot
frequency_table = wine['quality'].value_counts()
percentage = (frequency_table / len(wine)) * 100

axes[2].bar(frequency_table.index, frequency_table, color='lightblue')
axes[2].set_title("Frequency of Quality (Both Wines)")
axes[2].set_xlabel("Quality")
axes[2].set_ylabel("Frequency")

#Show the "head" of the data table
wine_overview = wine.iloc[:5, :10]
print(wine_overview)

plt.tight_layout()
plt.show()

import numpy as np

#Is there any statistical difference in quality between the two wines? (spoiler: no)

# Extract the quality values for the two classes
class_0_quality = wine[wine['new_column'] == 0]['quality']
class_1_quality = wine[wine['new_column'] == 1]['quality']

# Calculate the observed difference in means
observed_difference = np.abs(np.mean(class_1_quality) - np.mean(class_0_quality))

# Perform the permutational test
num_permutations = 1000
combined_data = np.concatenate((class_0_quality, class_1_quality))
permutation_differences = np.zeros(num_permutations)

for i in range(num_permutations):
    np.random.shuffle(combined_data)
    permuted_class_0 = combined_data[:len(class_0_quality)]
    permuted_class_1 = combined_data[len(class_0_quality):]
    permutation_differences[i] = np.abs(np.mean(permuted_class_1) - np.mean(permuted_class_0))

# Calculate the p-value as the proportion of permutation differences greater than or equal to the observed difference
p_value = np.sum(permutation_differences >= observed_difference) / num_permutations

# Print the observed difference and p-value
print("Observed Difference in Means:", observed_difference)
print("Permutational Test p-value:", p_value)
print(np.sum(permutation_differences >= observed_difference))

#------------------ RED WINE, PREDICT QUALITY ---------------------------------------


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set the seed for reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# Split the dataset into training (80%) and testing (20%) sets
np.random.seed(42)
train_features, test_features, train_target, test_target = train_test_split(
    wine_red[['fixed acidity' , 'volatile acidity' , 'citric acid' , 'residual sugar' ,'chlorides'
                              , 'free sulfur dioxide' , 'density' , 'pH' , 'sulphates' , 'alcohol']],
    wine_red['quality'],
    test_size=0.2
)

# Adjust the target variable to range 0 - 5 instead of 3 - 8
train_target -= 3
test_target -= 3

# Convert the target variable to categorical
train_target = tf.keras.utils.to_categorical(train_target, num_classes=6)
test_target = tf.keras.utils.to_categorical(test_target, num_classes=6)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(500, activation='ReLU'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(300, activation='ReLU'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='ReLU'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='ReLU'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='ReLU'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(6, activation='softmax')  # Use softmax activation for classification
])

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(train_features.shape)
print(train_target.shape)
# Train the neural network
model.fit(train_features, train_target, epochs=200 ,batch_size = 32, verbose=1,validation_data=(test_features, test_target))

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(test_features, test_target, verbose=0)
print("Accuracy:", accuracy)

# "Relaxed" accuracy. Considered correct if estimated in the +-1 range (spoiler: it gets much higher)

# Get the predictions for the features
predictions = model.predict(test_features)

# Find the indexes of the highest number for each row
vector_predictions = np.argmax(predictions, axis=1) + 3
vector_real_predictions = np.argmax(test_target, axis=1) + 3

accuracy2 = (np.sum(vector_predictions == vector_real_predictions) +
             np.sum(vector_predictions == vector_real_predictions-1) +
             np.sum(vector_predictions == vector_real_predictions+1)) / len(test_target)
print("Accuracy2:", accuracy2)

#------------------ WHITE WINE, PREDICT QUALITY ---------------------------------------


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set the seed for reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# Split the dataset into training (80%) and testing (20%) sets
np.random.seed(42)
train_features, test_features, train_target, test_target = train_test_split(
    wine_white[['fixed acidity' , 'volatile acidity' , 'citric acid' , 'residual sugar' ,'chlorides'
                              , 'free sulfur dioxide' , 'density' , 'pH' , 'sulphates' , 'alcohol']],
    wine_white['quality'],
    test_size=0.2
)
# Adjust the target variable to range from 0 - 6 instead of 3 - 9
train_target -= 3
test_target -= 3

# Convert the target variable to categorical
train_target = tf.keras.utils.to_categorical(train_target, num_classes=7)
test_target = tf.keras.utils.to_categorical(test_target, num_classes=7)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(500, activation='LeakyReLU'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(300, activation='LeakyReLU'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='LeakyReLU'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='LeakyReLU'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='LeakyReLU'),
    tf.keras.layers.Dense(7, activation='softmax')  # Use softmax activation for classification
])

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the neural network
model.fit(train_features, train_target, epochs=200,batch_size = 32, verbose=1,validation_data=(test_features, test_target))

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(test_features, test_target, verbose=0)
print("Accuracy:", accuracy)

# "Relaxed" accuracy. Considered correct if estimated in the +-1 range (spoiler: it gets much higher)


# Get the predictions for the features
predictions = model.predict(test_features)

# Find the indexes of the highest number for each row
vector_predictions = np.argmax(predictions, axis=1) + 3
vector_real_predictions = np.argmax(test_target, axis=1) + 3

accuracy2 = (np.sum(vector_predictions == vector_real_predictions) +
             np.sum(vector_predictions == vector_real_predictions-1) +
             np.sum(vector_predictions == vector_real_predictions+1)) / len(test_target)
print("Accuracy2:", accuracy2)

#Accuracy è molto più elevata così

#------------------ WINE TOGETHER, PREDICT QUALITY ---------------------------------------


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set the seed for reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# Split the dataset into training (80%) and testing (20%) sets
np.random.seed(42)
train_features, test_features, train_target, test_target = train_test_split(
    wine[['fixed acidity' , 'volatile acidity' , 'citric acid' , 'residual sugar' ,'chlorides'
                              , 'free sulfur dioxide' , 'density' , 'pH' , 'sulphates' , 'alcohol']],
    wine['quality'],
    test_size=0.2
)
# Adjust the target variable to range from 0 to 6
train_target -= 3
test_target -= 3

# Convert the target variable to categorical
train_target = tf.keras.utils.to_categorical(train_target, num_classes=7)
test_target = tf.keras.utils.to_categorical(test_target, num_classes=7)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(500, activation='LeakyReLU'),
    tf.keras.layers.Dropout(0.2),  # Dropout layer with a dropout rate of 0.2 (adjust as needed)
    tf.keras.layers.Dense(300, activation='LeakyReLU'),
    tf.keras.layers.Dropout(0.2),  # Dropout layer with a dropout rate of 0.2 (adjust as needed)
    tf.keras.layers.Dense(100, activation='LeakyReLU'),
    tf.keras.layers.Dropout(0.2),  # Dropout layer with a dropout rate of 0.2 (adjust as needed)
    tf.keras.layers.Dense(100, activation='LeakyReLU'),
    tf.keras.layers.Dropout(0.2),  # Dropout layer with a dropout rate of 0.2 (adjust as needed)
    tf.keras.layers.Dense(100, activation='LeakyReLU'),
    tf.keras.layers.Dense(7, activation='softmax')  # Use softmax activation for classification
])

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(train_features.shape)
print(train_target.shape)
# Train the neural network
model.fit(train_features, train_target, epochs= 200, batch_size = 32, verbose=1,validation_data=(test_features, test_target))

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(test_features, test_target, verbose=0)
print("Accuracy:", accuracy)

# "Relaxed" accuracy. Considered correct if estimated in the +-1 range (spoiler: it gets much higher)


# Get the predictions for the features
from sklearn.metrics import mean_absolute_error

predictions = model.predict(test_features)

# Find the indexes of the highest number for each row
vector_predictions = np.argmax(predictions, axis=1) + 3
vector_real_predictions = np.argmax(test_target, axis=1) + 3

# Print the vector of indexes
print(vector_predictions)
print(vector_real_predictions)

accuracy2 = (np.sum(vector_predictions == vector_real_predictions) +
             np.sum(vector_predictions == vector_real_predictions-1) +
             np.sum(vector_predictions == vector_real_predictions+1)) / len(test_target)
print("Accuracy2:", accuracy2)

# How far off am I on average?
mae = mean_absolute_error(vector_real_predictions, vector_predictions)
print("Mean Absolute Error:", mae)

#------------------ WINE TOGETHER, PREDICT IF RED OR WHITE ---------------------------------------


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Set the seed for reproducibility
np.random.seed(123)
tf.random.set_seed(123)

#Naive classifier
naive_accuracy = wine_white.shape[0] / (wine_white.shape[0] + wine_red.shape[0])
print("Naive classifier accuracy: ", naive_accuracy)

# Split the dataset into training (80%) and testing (20%) sets
np.random.seed(42)
train_features, test_features, train_target, test_target = train_test_split(
    wine[[ 'fixed acidity' , 'volatile acidity' , 'citric acid' , 'residual sugar' ,'chlorides'
                              , 'free sulfur dioxide' , 'density' , 'pH' , 'sulphates' , 'alcohol','quality']],
    wine['new_column'],
    test_size=0.2
)

# Convert the target variable to categorical
train_target = tf.keras.utils.to_categorical(train_target, num_classes=2)
test_target = tf.keras.utils.to_categorical(test_target, num_classes=2)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='ELU'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='ELU'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(30, activation='ELU'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the neural network
model.fit(train_features, train_target, epochs=1000, batch_size = 32, verbose=1, validation_data=(test_features, test_target))

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(test_features, test_target, verbose=0)
print("Accuracy:", accuracy)

# Convert the categorical data back to original form
original_test_target = np.argmax(test_target, axis=1)

# Compute the ROC-AUC score
probabilities = model.predict(test_features)
roc_auc = roc_auc_score(original_test_target, probabilities[:, 1])
print("ROC-AUC Score:", roc_auc)




# Generate confusion matrix
cm = confusion_matrix(test_target, predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#------------- Let's check the accuracy I can reach with a decision tree and let's do features selection ---------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Split the dataset into training (80%) and testing (20%) sets
np.random.seed(42)
train_features, test_features, train_target, test_target = train_test_split(
    wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
           'free sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']],
    wine['new_column'],
    test_size=0.2
)

# Create an instance of DecisionTreeClassifier
model = DecisionTreeClassifier()

# Train the decision tree model
model.fit(train_features, train_target)

# Make predictions on the testing data
predictions = model.predict(test_features)

# Evaluate the model
accuracy = np.mean(predictions == test_target)
print("Accuracy:", accuracy)


# Get feature importances
importances = model.feature_importances_

# Create a DataFrame to display feature importances
feature_importances = pd.DataFrame({"Feature": train_features.columns, "Importance": importances})
feature_importances.sort_values(by="Importance", ascending=False, inplace=True)

# Print feature importances
print("Feature Importances:")
print(feature_importances)


import matplotlib.pyplot as plt
from sklearn import tree

# Plot the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(model, feature_names=train_features.columns, class_names=["Class 0", "Class 1"], filled=True)
plt.show()


# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances["Feature"], feature_importances["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances")
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate confusion matrix
cm = confusion_matrix(test_target, predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



from sklearn.metrics import roc_auc_score

# Calculate the predicted probabilities for the positive class
probabilities = model.predict_proba(test_features)[:, 1]

# Calculate the ROC-AUC score
roc_auc = roc_auc_score(test_target, probabilities)
print("ROC-AUC Score:", roc_auc)

#--------------------- Let's plot the empirical distribution of 'chlorides' to see if it is significantly different in the two classes, as the decision tree suggests

import matplotlib.pyplot as plt
import seaborn as sns

# Subset the dataset based on class labels
class_0_data = wine[wine['new_column'] == 0]
class_1_data = wine[wine['new_column'] == 1]

# Extract the "chlorides" values for each class
chloride_class_0 = class_0_data['chlorides']
chloride_class_1 = class_1_data['chlorides']

# Plot the empirical distribution for each class
plt.figure(figsize=(8, 6))

# Plot the KDE plot
sns.kdeplot(chloride_class_0, color='red')
sns.kdeplot(chloride_class_1, color='blue')
plt.hist(chloride_class_0, density = True, bins=20, alpha=0.5, color='red', label='Red Wine')
plt.hist(chloride_class_1, density = True, bins=20, alpha=0.5, color='blue', label='White wine')
plt.xlabel('Chlorides')
plt.ylabel('Density')
plt.title('Estimated Density of Chloride by Class')
plt.legend()
plt.show()

#--------------------- I perform a non-parametric test on the mean of 'chlorides' to check if there is significant difference

#   H0:    mean_chlorides_red_wine == mean_chlorides_white_wine    vs    #   H1:    mean_chlorides_red_wine != mean_chlorides_white_wine

import scipy.stats as stats

# Perform Mann-Whitney U test:
statistic, p_value = stats.mannwhitneyu(chloride_class_0, chloride_class_1, alternative='two-sided')

# Print the test result
print("Mann-Whitney U test")
print("Test Statistic:", statistic)
print("p-value:", p_value)



#p-value = 0. I reject H0.
#We conclude 'chlorides' is a good discriminant between red and white wine
