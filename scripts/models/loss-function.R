omega = 1:4
omega = log((omega+1)^2)

loss = matrix(
  c(0, omega[2], omega[4],
    omega[1], 0, omega[2],
    omega[3], omega[1], 0), nrow=3, ncol=3, byrow=T
)

#This loss matrix implies omega_6 > omega_5 > omega_4 > omega_3 > omega_2=omega_1 > 0

acc_loss = matrix(
  1, ncol=3, nrow=3
)
diag(acc_loss) = 0

ex = matrix(
  c(68, 16, 2,
    4, 97, 1,
    0, 9, 123), nrow=3, ncol=3, byrow=T
)
ex = ex/sum(ex)

1-sum(ex*acc_loss)/(sum(acc_loss)/6)
1-sum(ex*loss)/(sum(loss)/6)

row_normalize <- function(mat) {
  sweep(mat, 1, rowSums(mat), "/")
}

yhat_list <- replicate(3, row_normalize(matrix(runif(100 * 3), nrow = 100)), simplify = FALSE)

# Simulated true labels
ytrue <- max.col(matrix(runif(100 * 3), nrow = 100))

gamma.init = c(1/3, 1/3, 1/3)

y_ensemble <- Reduce(`+`, Map(`*`, gamma.init, yhat_list))
y_pred_label <- max.col(y_ensemble)
cm <- table(factor(ytrue, levels=1:3), factor(y_pred_label, levels=1:3)) / length(ytrue)
sum(cm * loss) / (sum(loss) / 6)

eco_weight <- function(gamma, ytrue, ypreds) {
  gamma <- gamma / sum(gamma)  # enforce simplex constraint
  
  # Ensemble prediction: convex combination
  y_ensemble <- Reduce(`+`, Map(`*`, gamma, ypreds))
  
  # Hard class decision
  y_pred_label <- max.col(y_ensemble)
  
  # Confusion matrix
  cm <- table(factor(ytrue, levels=1:3), factor(y_pred_label, levels=1:3)) / length(ytrue)
  
  # Ecological loss
  sum(cm * loss) / (sum(loss) / 6)
}

res <- optim(par = gamma.init,
             fn = eco_weight,
             method = "SANN",
             ytrue = ytrue,
             ypreds = yhat_list)

gamma <- res$par/sum(res$par)
y_ensemble <- Reduce(`+`, Map(`*`, gamma, yhat_list))
y_pred_label <- max.col(y_ensemble)
cm <- table(factor(ytrue, levels=1:3), factor(y_pred_label, levels=1:3)) / length(ytrue)
sum(cm * loss) / (sum(loss) / 6)
