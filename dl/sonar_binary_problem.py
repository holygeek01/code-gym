import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline 
#fixing random seed for reproducibility 

seed=7
numpy.random.seed(seed)
#load and split data 
dataframe=read_csv("sonar.csv")
dataset=dataframe.values
X=dataset[:,0:60].astype(float)
Y=dataset[:,60]
encoder=LabelEncoder()
encoded_Y=encoder.transform(Y)
def create_baseline():
#method to create model(60 relu,1 sigmoid)
	model=Sequential()
	model.add(Dense(60, input_dim=60,kernel_initializer='normal',activation='relu'))
	model.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[accuracy])
	return model
#evaluate model with standardised data
estimator=KerasClassifier(build_fn='create_baseline', epochs=100, batch_size=5, verbose=0)
kfold=StratifiedKFold(n_split=10, shuffle=True, random_state=seed)
result=cross_val_score(X,encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))