#%% LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
#%% Readin the dataset

data = pd.read_csv("star_data.csv")

#%% EDA - 1
print(data.shape) # (240, 7)

describe = data.describe()
print(describe)

info = data.info()
print(info)

print(data.isna().sum())

"""
Temperature (K)           0
Luminosity(L/Lo)          0
Radius(R/Ro)              0
Absolute magnitude(Mv)    0
Star type                 0
Star color                0
Spectral Class            0
dtype: int64
"""

#%% EDA - 2

# Pair Plot
sns.pairplot(data,diag_kind="kde",markers="+")
plt.show()

# Correlation Matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix,annot=True,fmt=".2f")
plt.title("Correlation Matrix Between Features")
plt.show()



#%% EDA - 3 Label Encoder

# Manually

star_color_uniqe = pd.DataFrame(data["Star color"].unique())
sprectral_class_unique = pd.DataFrame(data["Spectral Class"].unique())


star_color_dict = {'Red':1, 'Blue White':2, 'White':3, 'Yellowish White':4, 'Blue white':4,
       'Pale yellow orange':5, 'Blue':5, 'Blue-white':6, 'Whitish':7,
       'yellow-white':8, 'Orange':9, 'White-Yellow':10, 'white':11, 'Blue ':12,
       'yellowish':13, 'Yellowish':14, 'Orange-Red':15, 'Blue white ':16,
       'Blue-White':17}


spectral_class_dict = {'M':1, 'B':2, 'A':3, 'F':4, 'O':5, 'K':6, 'G':7}


data["Star color"] = data["Star color"].map(star_color_dict)
data["Spectral Class"] = data["Spectral Class"].map(spectral_class_dict)


# With LabelEncoder library
le = LabelEncoder()
columnsToEncode = list(data.select_dtypes(include=["object"]))
for feature in columnsToEncode:
    data[feature] = le.fit_transform(data[feature])

#%% To Gettin  X and Y Coordinates

y = data["Spectral Class"].values
x_ = data.drop(["Spectral Class"],axis = 1)


#%% Normalization

x_ = (x_ - np.min(x_)) / (np.max(x_) - np.min(x_))


#%% Train Test Split

x_train,x_test,y_train,y_test = train_test_split(x_,y,test_size=0.3,random_state = 42)




#%% Confusion Matrix

def DrawConfusionMatrix(predicted,ML_algo):
    y_pred = predicted
    y_true = y_test
    cm = confusion_matrix(y_true,y_pred)
    
    f,ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.title("Confusion Matrix for {}".format(ML_algo))
    plt.show()




#%% Decision Tree Classifier

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_predicted_dt = dt.predict(x_test)
print("Accuracy of Decision Tree => {}".format(dt.score(x_test,y_test)*100)) # Accuracy of Decision Tree => 88.88888888888889
DrawConfusionMatrix(y_predicted_dt,"Decision Tree Classifier")

# Viualize of the Decision Tree
plt.figure(figsize=(25,25))
plot_tree(dt,feature_names=['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)',
       'Absolute magnitude(Mv)', 'Star type', 'Star color'],class_names=["0","1","2","3","4","5","6","7"],filled = True)


##########################

plot_tree(dt, filled = True)




#%% KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_predicted_knn = knn.predict(x_test)
print("Accuracy of KNN for k = 3 => {}".format(knn.score(x_test,y_test)*100)) # Accuracy of KNN for k = 3 => 88.88888888888889
DrawConfusionMatrix(y_predicted_knn,"KNN Classifier for k = 3")




# choosing the best k value 


score_list = []

for each in range(1,168):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))


plt.plot(range(1,168),score_list)
plt.title("K-Value & Accuracy")
plt.xlabel("K-Value")
plt.ylabel("Accuracy")
plt.show()



# best k = 3 






