import pandas as pd
from keras.layers import Dense,Dropout,Activation
from keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.utils import np_utils
import numpy as np
import h5py

#reading dataset
data = pd.read_csv('.\pulsar_stars.csv')

#splitting into training set and labels
x = data.iloc[:,:8]
y = data.iloc[:,8:]

#training and testing split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#training and valiation split
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1)

#Standardization
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_val = sc.fit_transform(x_val)
x_test = sc.fit_transform(x_test)

#one hot encoding
y_train = np_utils.to_categorical(y_train,2)
y_test = np_utils.to_categorical(y_test,2)
y_val = np_utils.to_categorical(y_val,2)

#Network
model = Sequential()

model.add(Dense(4,input_shape=(8,),kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))
#model.add(Dropout(0.25))

model.add(Dense(8,kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))

model.add(Dense(16,kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))

model.add(Dense(32,kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))

model.add(Dense(64,kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=20,verbose=2,validation_data=(x_val,y_val),callbacks=[EarlyStopping(monitor='val_acc',patience=2)])

score = model.evaluate(x_test,y_test,verbose=0)
print('Accuracy: ',score[1]*100,'%')

model.save('Project3.h5')


