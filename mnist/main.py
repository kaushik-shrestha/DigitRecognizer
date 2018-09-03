from keras.models import load_model
from . import config
from .conv import reader
import numpy as np
import pandas as pd


model = load_model(config.root + 'graph/model.h5')
test = reader.read_test()
results = model.predict(test)
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")
num_col = pd.Series(range(1, 28001), name="ImageId")
submission = pd.concat([num_col, results], axis=1)
submission.to_csv(config.root + "data/submission.csv", index=False)
