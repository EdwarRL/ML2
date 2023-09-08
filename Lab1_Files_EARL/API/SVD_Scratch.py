# Módulo para calcular la descomposición en valores singulares de una matriz

#Libraries
import numpy as np


def CalU(MA):
    cov_matrix = np.dot(MA, MA.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    ncols = np.argsort(eigenvalues)[::-1]
    # print('eigenvectors U')
    # print(eigenvectors)

    # Normalize each eigenvector by its eigenvalue
    # normalized_eigenvectors = []
    # for i in range(len(eigenvalues)):
    #     eigenvalue = eigenvalues[i]
    #     eigenvector = eigenvectors[:, i]
    #     normalized_eigenvector = eigenvector / eigenvalue
    #     normalized_eigenvectors.append(normalized_eigenvector)
    # normalized_eigenvectors = np.array(normalized_eigenvectors).T
    # return normalized_eigenvectors[:,ncols]
    return eigenvectors[:,ncols]

def CalSigma(MA):
    cov_matrix = np.dot(MA, MA.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    ncols = np.argsort(eigenvalues)[::-1]
    ordEing=eigenvalues[::-1]
    filterEig=ordEing[abs(ordEing)>=0.1]
    # print('eigenvalues U')
    # print(filterEig)
    # cov_matrix = np.dot(MA.T, MA)
    # eigenvalues, eigenvectors, ncols=calEingens(cov_matrix)
    # print('eigenvalues V')
    # print(eigenvalues)
    S = np.sqrt(filterEig)
    # print('S')
    # print(S)

    #Sorting in descending order as the svd function does
    return S

def CalVT(MA):
    cov_matrix=np.dot(MA.T, MA)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    ncols = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:,ncols].T

def CalcularSVD(MA):
    # column_means = np.mean(MA, axis=0)
    # column_std = np.std(MA, axis=0)
    # MAE=(MA - column_means) / column_std
    U=CalU(MA)
    S=CalSigma(MA)
    Vt=CalVT(MA)

    return U, S, Vt

def reconstruc(U,S,Vt):

    D=np.diag(S)

    nrow,ncol = D.shape
    nrowU, ncolU = U.shape

    # Number of rows to add
    num_rows_to_add = max(nrowU-nrow,0)
    # print(num_rows_to_add)

    if num_rows_to_add>0:
        # Create a matrix of zeros to add
        zeros_matrix = np.zeros((num_rows_to_add, D.shape[1]))

        # Concatenate the two matrices vertically
        result_matrix = np.vstack((D, zeros_matrix))
    else:
        result_matrix=D

    # Reconstructed Matriz
    reconstructed_matrix = np.dot(U, np.dot(result_matrix, Vt))
    return reconstructed_matrix