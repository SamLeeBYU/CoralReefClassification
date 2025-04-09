from preprocess import CoralDataPreprocessor
from load import CoralHealthDataset

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt

omega = np.log((np.arange(1, 5) + 1)**2)
Omega = np.array([
    [0,       omega[1], omega[3]],  # True class 1 (Dead)
    [omega[0], 0,       omega[1]],  # True class 2 (Unhealthy)
    [omega[2], omega[0], 0]         # True class 3 (Healthy)
])

class CoralEnsembleTrainer:
    def __init__(self, X, y, num_classes, loss_matrix):
        #Load in the dataset from preprocess.py
        self.X = X
        self.y = y
        
        self.num_classes = num_classes

        #This should be set to our defined ecologically sensitive loss matrix
        self.loss_matrix = loss_matrix
        self.models = []
        self.gamma = np.array([1/3, 1/3, 1/3])
        
        #Categorical labels needed for tensor flow
        self.y_train_cat = None
        self.y_test_cat = None

    def build_model(self, use_augmentation=False):
        model = models.Sequential()
        model.add(layers.Input(shape=(128, 128, 3)))

        #Add "random" data augmentation layers
        if use_augmentation:
            model.add(layers.RandomRotation(0.2))
            model.add(layers.RandomFlip('horizontal'))
            model.add(layers.RandomZoom(0.2))

        #Convolutional and pooling layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #Fully connected layers for classification
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        #We use a categorical crossentropy loss function since it has a smooth gradient
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_models(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=484, stratify=self.y)
        
        #This is an annoying thing about tensor flow: We have to make dummy variables to indicate the discrete classes
        self.y_train_cat = tf.keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test_cat = tf.keras.utils.to_categorical(y_test, self.num_classes)

        #We train three models (though this could be extended)
        model_configs = [False, True, True]
        for aug in model_configs:
            model = self.build_model(use_augmentation=aug)
            model.fit(X_train, self.y_train_cat, epochs=15, batch_size=32, validation_split=0.2, verbose=0)
            self.models.append(model)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    #Get the set of predictions from each model
    def get_predictions(self, X):
        return [model.predict(X) for model in self.models]

    #Our final ensemble prediction method to compute the "probability" of each class
    def predict_proba(self, X):
        preds = self.get_predictions(X)
        gamma = self.gamma if hasattr(self, 'gamma') else np.array([1/3] * len(self.models))
        gamma = gamma / np.sum(gamma)
        return sum(g * p for g, p in zip(gamma, preds))

    #Final ensemble prediction method
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    #Our specialized loss function as defined in the paper
    def ecological_loss(self, gamma, y_preds, y_true):
        gamma = np.array(gamma)
        gamma /= np.sum(gamma)
        y_ensemble = sum(g * y_pred for g, y_pred in zip(gamma, y_preds))
        y_pred_labels = np.argmax(y_ensemble, axis=1)
        cm = confusion_matrix(y_true, y_pred_labels, labels=[0, 1, 2])
        cm_normalized = cm / np.sum(cm)
        return np.sum(cm_normalized * self.loss_matrix) / (np.sum(self.loss_matrix) / 6)

    #We use a stochastic global optimization routine that doesn't require any computation of a numerical gradient (since it doesn't exist in our case)
    def optimize_gamma(self, cv=0):
        if cv <= 1:
            y_preds = self.get_predictions(self.X_train)
            result = dual_annealing(self.ecological_loss, bounds=[(-10, 10)] * len(y_preds), args=(y_preds, self.y_train))
            self.gamma = result.x / np.sum(result.x)
        else:
            #We use k-fold CV to find the overall best gamma weights
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            gammas = []
            for train_idx, val_idx in kf.split(self.X):
                X_fold, y_fold = self.X[train_idx], self.y[train_idx]
                preds = [model.predict(X_fold) for model in self.models]
                result = dual_annealing(self.ecological_loss, bounds=[(-10, 10)] * len(preds), args=(preds, y_fold))
                gammas.append(result.x / np.sum(result.x))
            self.gamma = np.mean(gammas, axis=0)

    def plot_confusion_matrix(self, y_true, y_pred):
        labels = [0, 1, 2]  # Dead, Unhealthy, Healthy
        display_labels = ["Dead", "Unhealthy", "Healthy"]

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues, values_format='d')

        disp.ax_.xaxis.set_ticks_position('top')
        disp.ax_.xaxis.set_label_position('top')

        plt.title("Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

if __name__ == "__main__":
    #Load and preprocess the dataset  
    coral = CoralHealthDataset(data_dirs=["data/coral/dead", "data/coral/unhealthy", "data/coral/healthy"])
    preprocessor = CoralDataPreprocessor(dataset_dict=coral.dataset)
    coral_processed = preprocessor.process_dataset()
    X, y = preprocessor.get_data()

    model = CoralEnsembleTrainer(X, y, num_classes=3, loss_matrix=Omega)
    model.train_models()
    model.optimize_gamma(cv=3)
    preds = model.predict(model.X_test)
    model.plot_confusion_matrix(model.y_test, preds)
