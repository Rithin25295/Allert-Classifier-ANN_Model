
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the given dataset
df1 = pd.read_csv("processed_alerts_obfuscated_v2.csv")
df2 = pd.read_csv("demographics.csv")

#Examining the data
df1.head(10)
df1.shape,df2.shape

df1['insert_date'] = pd.to_datetime(df1['insert_date'])

df1['alert_escalation_date'] = pd.to_datetime(df1['alert_escalation_date'])

df= pd.merge(df1,df2,how='inner',on='employee_id')

#Classifying into two classes in total
df['label']= df['classification'].apply(lambda x: 'FP' if x=='FP' else 'TP')

#Indexing the indicators
df['indicator_id'] = df.groupby(['indicators']).ngroup()


#Separating the final dataset
final = df[['State_Name','indicator_id','grouping','label']]
final.to_csv("Final.csv",index = False)


dataset= pd.read_csv("Final.csv")
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 39, init = 'uniform', activation = 'relu', input_dim = 77))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 39, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 100, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix for model evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
