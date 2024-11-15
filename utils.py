import joblib


def load_model(model_name):
    return joblib.load(f'models/{model_name}.model')
