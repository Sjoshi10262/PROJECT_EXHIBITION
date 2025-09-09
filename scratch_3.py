import os
from idlelib import history

import matplotlib
import mne
import numpy as np
from glob import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D,Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as py


all_file_path = glob(r'C:\Users\LENOVO\OneDrive\Desktop\project\PM*.edf')
print("Total Files:", len(all_file_path))

labels=[]
for i in all_file_path:
    a=os.path.basename(i)#it returns file name
    number_str=a.replace("PM","").replace(".edf","")#
    number=int(number_str)
    if number%2==1:
        labels.append(0)#0->healthy
    else:
        labels.append(1)#1->patient
print("Labels:", labels)

#preprocessing
def preprocess(all_file_path):
    raw=mne.io.read_raw_edf(all_file_path,verbose=False,preload=True)
    raw=raw.resample(128)
    data=raw.get_data()
    return data

#padding and trimming as data are not in same size
def pad_trim(data,target_len=20000):
    if data.shape[1]>target_len:
        return data[:,:target_len]#too long
    else:
        pad_width=target_len-data.shape[1]
        return np.pad(data,((0,0),(0,pad_width)),mode='constant')

x=[]
for i in all_file_path:
    data=preprocess(i)
    data=pad_trim(data,20000)
    x.append(data)
x=np.array(x)
print("final shape:",x)

#epoching
def create_epochs(data,window=256):
    epochs=[]
    for i in range(0,data.shape[1]-window,window):
        segment=data[:,i:i+window]
        epochs.append(segment)
    return np.array(epochs)
all_epoch=[]
all_labels=[]
for i,d in enumerate(x):
    ep=create_epochs(d)
    all_epoch.append(ep)
    all_labels.extend([labels[i]]*len(ep))
x_data=np.vstack(all_epoch)
y_data=np.array(all_labels)
print("x_data:",x_data.shape)
print("y_data:",y_data.shape)

#train-test split
X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=42)
print("X_train:",X_train.shape)
print("y_train:",y_train.shape)
print("X_test:",X_test.shape)
print("y_test:",y_test.shape)

#cnn models
model=Sequential([
    Conv1D(filters=16,kernel_size=3,activation='relu',input_shape=(X_train.shape[1],X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid'),
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print("loss:", loss)
print("acc:", acc)
model.save('cnn_model.h5')


plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

code= r"""import os
from idlelib import history

import matplotlib
import mne
import numpy as np
from glob import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D,Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as py


all_file_path = glob(r'C:\Users\LENOVO\OneDrive\Desktop\project\PM*.edf')
print("Total Files:", len(all_file_path))

labels=[]
for i in all_file_path:
    a=os.path.basename(i)#it returns file name
    number_str=a.replace("PM","").replace(".edf","")#
    number=int(number_str)
    if number%2==1:
        labels.append(0)#0->healthy
    else:
        labels.append(1)#1->patient
print("Labels:", labels)

#preprocessing
def preprocess(all_file_path):
    raw=mne.io.read_raw_edf(all_file_path,verbose=False,preload=True)
    raw=raw.resample(128)
    data=raw.get_data()
    return data

#padding and trimming as data are not in same size
def pad_trim(data,target_len=20000):
    if data.shape[1]>target_len:
        return data[:,:target_len]#too long
    else:
        pad_width=target_len-data.shape[1]
        return np.pad(data,((0,0),(0,pad_width)),mode='constant')

x=[]
for i in all_file_path:
    data=preprocess(i)
    data=pad_trim(data,20000)
    x.append(data)
x=np.array(x)
print("final shape:",x)

#epoching
def create_epochs(data,window=256):
    epochs=[]
    for i in range(0,data.shape[1]-window,window):
        segment=data[:,i:i+window]
        epochs.append(segment)
    return np.array(epochs)
all_epoch=[]
all_labels=[]
for i,d in enumerate(x):
    ep=create_epochs(d)
    all_epoch.append(ep)
    all_labels.extend([labels[i]]*len(ep))
x_data=np.vstack(all_epoch)
y_data=np.array(all_labels)
print("x_data:",x_data.shape)
print("y_data:",y_data.shape)

#train-test split
X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=42)
print("X_train:",X_train.shape)
print("y_train:",y_train.shape)
print("X_test:",X_test.shape)
print("y_test:",y_test.shape)

#cnn models
model=Sequential([
    Conv1D(filters=16,kernel_size=3,activation='relu',input_shape=(X_train.shape[1],X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid'),
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print("loss:", loss) 
print("acc:", acc)

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""
with open("project_code.txt","w") as f:
    f.write(code)
    print(code)
