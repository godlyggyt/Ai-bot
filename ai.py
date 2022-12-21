# import the necessary libraries
import numpy as np
import tensorflow as tf

# define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(1), loss='mean_squared_error')

# provide some training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# train the model
model.fit(xs, ys, epochs=500)

# use the model to predict a value
print(model.predict([10.0]))
