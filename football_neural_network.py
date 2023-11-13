import pandas
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from flask import Flask



def get_predictions(data):
    x = data.drop(['Opponent', 'Result'], axis=1)
    y = data['Result']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(x.shape[1],)),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=100, batch_size=1, validation_data=(x, y))

    return model.predict(x)


def get_results(predictions):
    res = []

    for i in predictions:
        if i > -1.5 and i < -0.5:
            res.append(-1)
        elif i > 0.5 and i < 1.5:
            res.append(1)
        else:
            res.append(0)

    return res


def get_accuracy(results, data):
    number = 0
    for i in range(len(results)):
        if results[i] == data['Result'][i]:
            number += 1
    return number / (len(results)) * 100


football_data = pandas.read_csv('football_data.csv')
football_predictions = get_predictions(football_data)
football_results = get_results(football_predictions)
print("Neural Network Accuracy: ", get_accuracy(football_results, football_data), "%")
football_list = football_data["Opponent"]
app = Flask(__name__)


@app.route('/discipline/<string:discipline>/opponent/<string:opponent>')
def check(discipline, opponent):
    if discipline == 'football':
        for i in range(len(football_list)):
            if football_list[i] == opponent:
                if football_results[i] == 1:
                    return {"discipline": discipline, "opponent": opponent, "result": "Poland Wins!"}
                elif football_results[i] == -1:
                    return {"discipline": discipline, "opponent": opponent, "result": "Poland Lose!"}
                else:
                    return {"discipline": discipline, "opponent": opponent, "result": "Draw!"}

        return {"discipline": discipline, "opponent": opponent, "result": "There is no such opponent in the database!"}


if __name__ == '__main__':
    app.run(debug=True)
