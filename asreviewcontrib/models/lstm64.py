from asreview.models.classifiers.base import BaseTrainClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Input, LSTM, Dropout
from scikeras.wrappers import KerasClassifier


class LongShortTermMemory64(BaseTrainClassifier):

    name = "lstm64"

    def __init__(self):

        super().__init__()
        self._model = None

    def fit(self, X, y):
        """Fit the model to the data."""
        
        model = Sequential()
        model.add(LSTM(128, input_shape=(1, X.shape[1]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, input_shape=(1, X.shape[1])))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self._model = KerasClassifier(model)

        X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return self._model.fit(X, y, epochs=10, batch_size=64)

    def predict_proba(self, X):
        X = X.reshape(X.shape[0], 1, X.shape[1])
        return self._model.predict_proba(X)