import os.path as path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def get_predictions(model, data, sport):
    x = data.drop(['Opponent', 'Result'], axis=1)
    y = data['Result']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if path.isdir('./models/' + sport):
        model = tf.keras.models.load_model('./models/' + sport)
    else:
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, epochs=100, batch_size=1, validation_data=(x, y))
        model.save('./models/' + sport, save_format='tf')

    return model.predict(x)


def get_results(predictions, model):
    res = []

    for i in predictions:
        if i > -1.5 and i <= model.up_lose_boundery:
            res.append(-1)
        elif i >= model.down_win_boundery and i < 1.5:
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