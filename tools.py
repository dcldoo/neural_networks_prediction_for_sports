
from sklearn.preprocessing import StandardScaler

def get_predictions(model, data):
    x = data.drop(['Opponent', 'Result'], axis=1)
    y = data['Result']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

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