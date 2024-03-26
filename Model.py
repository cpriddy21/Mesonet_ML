""" Model.py"""
from keras import Sequential
from keras.src.layers import Dense
from sklearn.ensemble import RandomForestClassifier


class Model:
    @staticmethod
    def create_model(model_type, input_shape):
        if model_type == 'random_forest':
            return RandomForestClassifier()
        elif model_type == 'neural_network':
            model = Sequential([
                Dense(64, activation='relu', input_shape=(input_shape.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            return model
        else:
            raise ValueError(f"Unknown model type")
