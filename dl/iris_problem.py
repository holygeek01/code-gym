import numpy
from pandas import read_csv
from keras.models import Sequential 
from keras.layers import Dense 
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder 
from sklearn.pipeline import Pipeline 
#fix the random seed for reproducability
seed=7
numpy.random.seed(seed)
#load data from csv
dataframe = read_csv("iris.csv", header=None)
dataset=dataframe.values
X=dataset[:,0:4].astype(float)
Y=dataset[:,4]
#one-hot encoding using labelEncoder and categorical()
encoder=LabelEncoder()
encoder.fit(Y)
encoded_Y=encoder.transform(Y)
#one-hot encodding
dummy_Y=np_utils.to_categorical(encoded_Y)
#print(dummy_Y)
def baseline_model():
#function to create and compile the model 
	model=Sequential()
	model.add(Dense(4, input_dim=4, kernel_initializer='normal',activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
	#compiling the model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
#setting hyperparameters  for the model(epochs, batch size)
estimator=KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=2, verbose=0)
#defining K-fold cross validation with 10 folds 
kfold=KFold(n_splits=10, shuffle=True, random_state=seed)
results=cross_val_score(estimator,X, dummy_Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))