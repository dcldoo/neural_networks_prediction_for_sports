import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

data = pd.read_csv("basketball_data.csv")

X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = keras.Sequential([keras.layers.Dense(16,activation='relu',input_shape=(X.shape[1],)),
                          keras.layers.Dense(1)
                          ])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X,y, epochs=100,batch_size=1, validation_data=(X,y))

predictions = model.predict(X)

results = []
for i in predictions:
  if i >= -1.5 and i < -0.5:
    results.append(-1)
  elif i >=-0.5 and i <0.5:
    results.append(0)
  else:
    results.append(1)

number = 0

for i in range(161):
  if y[i] == results[i]:
    number +=1

percentage = number /161*100
print("Neural Network Accuracy: ", percentage, "%")

opponent_list = data["Opponent"]
opponent = input("Choose opponent: ")

is_empty = True
for i in range(161):
    if opponent_list[i] == opponent:
        is_empty = False
        if results[i] == 1:
            print("Poland Wins!")
        elif results[i] == -1:
            print("Poland Lose!")
        else:
            print("Draw!")

if is_empty == True:
    print("There is no such opponent in the database!")

