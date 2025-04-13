import numpy as np

def ensemble_loss(preds, y, omega_init=None):
    if omega_init is None:
        omega_init = np.ones(preds.shape[0])/preds.shape[0]

    m, n, k = preds.shape

    #Compute Frobenius Squared Norm
    weights = np.ones((m, n, k))*omega_init.reshape(m, 1, 1)
    y_ensemble = np.sum(weights*preds, axis=0)
    loss = np.sum((y - y_ensemble)**2)
    return loss

if __name__ == "__main__":

    n = 5 #number of observations
    k = 2 #number of classes
    m = 3 #number of models

    y = np.array([0, 1, 0, 1, 0])
    y_cat = np.eye(k)[y]

    preds = np.random.rand(m, n, k)
    
    omega_init = np.array([0.5, 0.5])
    loss = ensemble_loss(preds, y_cat, omega_init)
    print(loss)