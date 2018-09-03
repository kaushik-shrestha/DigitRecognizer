import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from . import model_builder as mb
from . import reader
from .. import config as cfg

x_train, y_train = reader.read_train()

reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
check_pointer = ModelCheckpoint(filepath=cfg.root + 'graph/model.h5', verbose=1, save_best_only=True)

model = mb.build()
model.summary()

history = model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=86,
        validation_split=0.05,
        callbacks=[reduce_lr, check_pointer],
        shuffle=True
    )

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
