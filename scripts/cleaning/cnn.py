from preprocess import CoralDataPreprocessor
from load import CoralHealthDataset

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt

import os
import json

class CoralEnsembleTrainer:
    def __init__(self, X, y, num_classes=3, loss_matrix=None, epochs=32, batch_size=32):

        self.m = 3 #Number of models in the ensemble
        self.epochs = epochs
        self.batch_size = batch_size

        self.num_classes = num_classes

        #This should be set to our defined ecologically sensitive loss matrix
        self.loss_matrix = loss_matrix
        if loss_matrix is None:
            omega = np.log((np.arange(1, 5) + 1)**2)
            Omega = np.array([
                [0,       omega[1], omega[3]],  # True class 1 (Dead)
                [omega[0], 0,       omega[1]],  # True class 2 (Unhealthy)
                [omega[2], omega[0], 0]         # True class 3 (Healthy)
            ]).T
            self.loss_matrix = Omega

        #Load in the dataset from preprocess.py
        self.X = X
        self.y = y

        #Perform a train-test split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=484, stratify=self.y)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        #This is an annoying thing about tensor flow: We have to make dummy variables to indicate the discrete classes
        self.y_train_cat = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test_cat = tf.keras.utils.to_categorical(self.y_test, self.num_classes)
        
        self.models = []
        self.gammas = np.ones((self.m, self.num_classes))
        self.omega = np.ones(self.m)/self.m

    def build_model(self, use_augmentation=False):
        model = models.Sequential()
        model.add(layers.Input(shape=(128, 128, 3)))

        #Add "random" data augmentation layers
        if use_augmentation:
            model.add(layers.RandomRotation(0.2))
            model.add(layers.RandomFlip('horizontal_and_vertical'))
            model.add(layers.RandomZoom(0.2))

        #Convolution Block 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        #model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        # model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        #Convolution Block 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        #model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        # model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.35))

        #Fully connected layers
        model.add(layers.Flatten())

        model.add(layers.Dense(128, activation='relu'))
        #model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.35))

        model.add(layers.Dense(128, activation='relu'))
        #model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(self.num_classes, activation='softmax'))

        #We use a categorical crossentropy loss function since it has a smooth gradient
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_models(self):

        #We train three models (though this could be extended)
        model_configs = [False, True, True][0:self.m]
        for aug in model_configs:
            model = self.build_model(use_augmentation=aug)
            model.fit(self.X_train, self.y_train_cat, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, verbose=1)
            self.models.append(model)

    #Get the set of predictions from each model
    def get_predictions(self, X):
        return [self.models[m].predict(X) * self.gammas[m].reshape(1, -1) for m in range(self.m)]

    #Our final ensemble prediction method to compute the "probability" of each class
    def predict_proba(self, X, omega=None):
        if omega is None:
            omega = self.omega
        #(Calibrated) predictions from each model
        preds = self.get_predictions(X)
        return sum(w * p for w, p in zip(omega, preds))

    #Final ensemble prediction method
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    #Our specialized loss function as defined in the paper
    def ecological_loss(self, y_preds, y_true):
        cm = confusion_matrix(y_true, y_preds, labels=list(range(self.num_classes)))
        cm_normalized = cm / np.sum(cm)
        return np.sum(cm_normalized * self.loss_matrix) / (np.sum(self.loss_matrix) / (self.num_classes**2 - self.num_classes))
    
    def eco_optimize(self, omega, y_preds, y_true):
        #Constraint
        omega = omega/np.sum(omega)

        #Compute y ensemble predictions
        y_ensemble = sum(w * y_preds for w, y_preds in zip(omega, y_preds))
        y_pred_labels = np.argmax(y_ensemble, axis=1)

        cm = confusion_matrix(y_true, y_pred_labels, labels=list(range(self.num_classes)))
        cm_normalized = cm / np.sum(cm)
        return np.sum(cm_normalized * self.loss_matrix) / (np.sum(self.loss_matrix) / (self.num_classes**2 - self.num_classes))

    @staticmethod
    def compute_gamma(Y, X):

        numerator = np.sum(X * Y, axis=0)  # Sum over i: [sum(x_i1 y_i1), ..., sum(x_ik y_ik)]
        denominator = np.sum(X**2, axis=0)  # [sum(x_i1^2), ..., sum(x_ik^2)]
        
        lambda_num = 1 - np.sum(numerator / denominator)
        lambda_den = np.sum(1 / denominator)
        lambda_val = 2 * lambda_num / lambda_den
        
        gamma = (lambda_val / 2 + numerator) / denominator
        return gamma

    #A method to calibrate the gamma weights of an individual model
    def recalibrate_model(self, model_index, cv=0):
        gamma_star = np.ones(self.num_classes)/self.num_classes
        model_m = self.models[model_index]
        if cv <= 1:
            x_mat = model_m.predict(self.X_train)
            y = self.y_train_cat
            gamma_star = self.compute_gamma(x_mat, y)
        else:
            kf = KFold(n_splits=cv, shuffle=True)
            gammas = []
            losses = []
            for train_idx, val_idx in kf.split(self.X):
                X_fold, y_fold = self.X[train_idx], self.y[train_idx]
                X_test, y_test = self.X[val_idx], self.y[val_idx]
                x_mat = model_m.predict(X_fold)
                y = y_fold
                y_cat = np.eye(self.num_classes)[y]
                gamma = self.compute_gamma(x_mat, y_cat)
                gammas.append(gamma)

                #Predict the labels for the test set using our gamma weights
                x_mat_test = model_m.predict(X_test)
                y_pred = np.argmax(x_mat_test*gamma.reshape(1, -1), axis=1)
                losses.append(self.ecological_loss(y_pred, y_test))
            gamma_star = gammas[np.argmin(losses)]
        self.gammas[model_index] = gamma_star

    #We use a stochastic global optimization routine that doesn't require any computation of a numerical gradient (since it doesn't exist in our case)
    def optimize_omega(self, cv=0):
        if cv <= 1:
            y_preds = self.get_predictions(self.X_train)
            result = dual_annealing(self.ecological_loss, bounds=[(-1, 1)] * len(y_preds), args=(y_preds, self.y_train))
            self.omega = result.x / np.sum(result.x)
        else:
            #We use k-fold CV to find the overall best gamma weights
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            omegas = []
            losses = []
            for train_idx, val_idx in kf.split(self.X):
                X_fold, y_fold = self.X[train_idx], self.y[train_idx]
                X_test, y_test = self.X[val_idx], self.y[val_idx]
                preds = self.get_predictions(X_fold)
                result = dual_annealing(self.eco_optimize, bounds=[(0, 1)]*self.m, args=(preds, y_fold))
                omega = result.x / np.sum(result.x)
                omegas.append(omega)

                #Out-of-sample loss
                y_pred = self.predict_proba(X_test, omega)
                y_pred_labels = np.argmax(y_pred, axis=1)
                losses.append(self.ecological_loss(y_pred_labels, y_test))
            self.omega = omegas[np.argmin(losses)]

    def save(self, path="models/beta"):
        os.makedirs(path, exist_ok=True)
        
        #Save each submodel
        for i, model in enumerate(self.models):
            model.save(os.path.join(path, f"cnn_model_{i}.keras"))

        #Save gamma weights
        np.save(os.path.join(path, "gamma.npy"), self.gammas)

        #Save omega weights
        np.save(os.path.join(path, "omega.npy"), self.omega)

        #Save configuration
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({
                "num_classes": self.num_classes,
                "loss_matrix": self.loss_matrix.tolist()
            }, f)

    def load(self, path="models/beta"):
        self.models = []
        for i in range(self.m):
            self.models.append(tf.keras.models.load_model(os.path.join(path, f"cnn_model_{i}.keras")))

        self.gamma = np.load(os.path.join(path, "gamma.npy"))
        self.omega = np.load(os.path.join(path, "omega.npy"))

        with open(os.path.join(path, "config.json"), "r") as f:
            cfg = json.load(f)
            self.num_classes = cfg["num_classes"]
            self.loss_matrix = np.array(cfg["loss_matrix"])

    def plot_confusion_matrix(self, y_true, y_pred, display_labs=None):
        labels = [0, 1, 2]  # Dead, Unhealthy, Healthy
        display_labels = ["Dead", "Unhealthy", "Healthy"]
        if display_labs is not None:
            labels = list(range(self.num_classes))
            display_labels = display_labs

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
    #Load and preprocess the dataset (Hugging Face Dataset)
    coral = CoralHealthDataset(data_dirs=["data/coral/dead", "data/coral/unhealthy", "data/coral/healthy"])
    preprocessor = CoralDataPreprocessor(dataset_dict=coral.dataset)
    coral_processed = preprocessor.process_dataset()
    X, y = preprocessor.get_data()

    model = CoralEnsembleTrainer(X, y, num_classes=3, epochs=100)
    model.load(path="models/beta")

    #Train and Calibrate (you don't need to do this if you are loading in a model)
    model.train_models()
    for m in range(model.m):
        model.recalibrate_model(m, cv=5)
    model.optimize_omega(cv=5)

    model.save(path="models/beta")
    ##############################################################################

    preds = model.predict(model.X_test)
    model.plot_confusion_matrix(model.y_test, preds)
