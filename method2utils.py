# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np

def grad_U(Ui, Yij, Vj, ai, bj, mu, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), the user specific deviation ai, the movie
    specific deviation bj, the global bias mu, reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta * (reg * Ui - Vj * (Yij - mu - np.dot(Ui, Vj) - ai - bj))

def grad_V(Vj, Yij, Ui, ai, bj, mu, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), the user specific deviation ai, the movie
    specific deviation bj, the global bias mu, reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta * (reg * Vj - Ui * (Yij - mu - np.dot(Ui, Vj) - ai - bj))

def grad_a(ai, bj, Yij, Ui, Vj, mu, reg, eta):
    """
    Takes in as input the same thing that the previous functions take in

    Returns the gradient of the regularized loss function with
    respect to ai multiplied by eta.
    """
    return eta * (reg * ai - (Yij - mu - np.dot(Ui, Vj) - ai - bj))

def grad_b(ai, bj, Yij, Ui, Vj, mu, reg, eta):
    """
    Takes in as input the same thing that the previous functions take in

    Returns the gradient of the regularized loss function with
    respect to ai multiplied by eta.
    """
    return eta * (reg * bj - (Yij - mu - np.dot(Ui, Vj) - ai - bj))

def get_err(U, V, Y, a, b, mu, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    # Compute mean squared error on each data point in Y; include
    # regularization penalty in error calculations.
    term_1 = reg / 2 * (np.linalg.norm(U) ** 2 + np.linalg.norm(V) **2 \
             + np.linalg.norm(a) ** 2 + np.linalg.norm(b) ** 2)
    term_2 = 0
    for entry in Y:
        i = entry[0] - 1
        j = entry[1] - 1
        Y_ij = entry[2]
        term_2 += (Y_ij - mu - np.inner(U[i], V[j]) - a[i] - b[j]) ** 2

    # Return the mean of the regularized error
    return (term_1 + 0.5 * term_2) / float(len(Y))

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, a, b, mu, err) where err is the unregularized MSE
    of the model.
    """
    # Initialize U, V, a, b, mu
    U = np.random.random((M,K)) - 0.5
    V = np.random.random((N,K)) - 0.5
    a = np.random.random(M) - 0.5
    b = np.random.random(N) - 0.5
    mu = np.mean([y for _,_,y in Y])
    size = Y.shape[0]
    delta = None
    indices = np.arange(size)
    for epoch in range(max_epochs):
        # Run an epoch of SGD
        before_E_in = get_err(U, V, Y, a, b, mu, reg)
        np.random.shuffle(indices)
        for ind in indices:
            (i,j, Yij) = Y[ind]
            i -= 1
            j -= 1
            # Update based on gradient
            U[i] -= grad_U(U[i], Yij, V[j].T, a[i], b[j], mu, reg, eta)
            V[j] -= grad_V(V[j].T, Yij, U[i], a[i], b[j], mu, reg, eta)
            a[i] -= grad_a(a[i], b[j], Yij, U[i], V[j].T, mu, reg, eta)
            b[j] -= grad_b(a[i], b[j], Yij, U[i], V[j].T, mu, reg, eta)
        # At end of epoch, print E_in
        E_in = get_err(U, V, Y, a, b, mu, reg)
        print("Epoch %s, E_in (regularized MSE): %s"%(epoch + 1, E_in))

        # Compute change in E_in for first epoch
        if epoch == 0:
            delta = before_E_in - E_in

        # If E_in doesn't decrease by some fraction <eps>
        # of the initial decrease in E_in, stop early
        elif before_E_in - E_in < eps * delta:
            break
    return (U, V, a, b, mu, get_err(U, V, Y, a, b, mu))
