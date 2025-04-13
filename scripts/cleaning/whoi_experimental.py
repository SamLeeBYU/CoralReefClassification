from load import CoralHealthDataset
from preprocess import CoralDataPreprocessor
from cnn import CoralEnsembleTrainer
import numpy as np

loss_matrix_WHOI = np.array([
                [0, 1],  # True class 2 (Unhealthy)
                [0, 0]# True class 3 (Healthy)
            ])

if __name__ == "__main__":
    coral = CoralHealthDataset(data_dirs=["data/WHOI"], annotations="data/WHOI/annotations.json")
    preprocessor = CoralDataPreprocessor(dataset_dict=coral.dataset)
    coral_processed = preprocessor.process_dataset()
    X, y = preprocessor.get_data()

    model = CoralEnsembleTrainer(X, y-1, num_classes=2, epochs = 50, loss_matrix=loss_matrix_WHOI)

    #Train and Calibrate (you don't need to do this if you are loading in a model)
    model.train_models()
    for m in range(model.m):
        model.recalibrate_model(m, cv=5)
    model.optimize_omega(cv=5)

    model.save(path="models/whoi")
    ##############################################################################

    preds = model.predict(model.X_test)
    model.plot_confusion_matrix(model.y_test, preds, display_labs=["Unhealthy", "Healthy"])