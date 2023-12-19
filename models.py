import tensorflow as tf


class FootballModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(16, activation='relu')
    self.dense2 = tf.keras.layers.Dense(1)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

class VolleyballModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(12, activation='relu')
    self.dense2 = tf.keras.layers.Dense(8, activation='relu')
    self.dense3 = tf.keras.layers.Dense(1)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    return self.dense3(x)
