import pandas
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

data = pandas.read_csv('volleyball_data.csv')
X = data.drop(['Opponent', 'Result'], axis=1)
y = data['Result']
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = keras.Sequential([
    keras.layers.Dense(12, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=1, validation_data=(X, y))

predictions = model.predict(X)
results = []

for i in predictions:
    if i > -1.5 and i < 0:
        results.append(-1)
    elif i > 0 and i < 1.5:
        results.append(1)
    else:
        results.append(0)

number = 0
for i in range(93):
    if results[i] == y[i]:
        number += 1
percentage = number / 93 * 100
print("Neural Network Accuracy: ", percentage, "%")

opponent_list = data["Opponent"]
opponent = input("Choose opponent: ")

is_empty = True
for i in range(93):
    if opponent_list[i] == opponent:
        is_empty = False
        if results[i] == 1:
            print("Poland Wins!")
        elif results[i] == -1:
            print("Poland Lose!")
        else:
            break;

if is_empty == True:
    print("There is no such opponent in the database!")
