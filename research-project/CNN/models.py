import tensorflow as tf


def cnn1(size, lr):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(size[1], size[2], size[3])))
            
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))


    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def cnn2(size, lr):
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(size[1], size[2], size[3])))
      model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))       
      model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
      model.add(tf.keras.layers.Dropout(0.3))
      
      model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

      model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
      model.add(tf.keras.layers.Dropout(0.3))

      model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
     
      model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
      model.add(tf.keras.layers.Dropout(0.3))
      
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(128, activation='relu'))
      model.add(tf.keras.layers.Dropout(0.4))
      model.add(tf.keras.layers.Dense(10, activation='softmax'))


      optimizer = keras.optimizers.Adam(learning_rate=lr)
      model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

      return model
            
def model_selector(model_name, size, lr):
    if model_name== "cnn1":
        model=cnn1(size, lr)
    elif model_name== "cnn2":
        model=cnn2(size, lr)
    return model