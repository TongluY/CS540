from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    return x - np.mean(x, axis = 0)

def get_covariance(dataset):
    # Your implementation goes here!
    return (1/(len(dataset)-1)) * np.dot(np.transpose(dataset), dataset)

def get_eig(S, m):
    # Your implementation goes here!
    # Return the largest m eigenvalues of S as a diagonal matrix, in descending order, and the corresponding eigenvectors as columns in a matrix.
    # argsort_S = eigh(S, eigvals_only=True).argsort()[::-1]
    n = len(S)
    # w, v = eigh(S, subset_by_index=[argsort_S[m-1],argsort_S[0]])
    w, v = eigh(S, subset_by_index=[n-m, n-1])
    w = np.diag(np.sort(w)[::-1])
    v = np.flip(v, axis=1)
    # v[:, [1,0]] = v[:, [0,1]]
    return w, v

def get_eig_prop(S, prop):
    # Your implementation goes here!
    # w, v = eigh(S, subset_by_value=[np.sum(eigh(S, eigvals_only=True))*prop, np.inf])
    # w, v = eigh(S, subset_by_value=[np.sum(np.trace(np.diag(eigh(S, eigvals_only=True))))*prop, np.inf])
    # rearrange the output of eigh to get the eigenvalues in decreasing order
    # w = np.diag(np.sort(w)[::-1])
    # w = np.diag(-np.sort(-w))
    # keep the eigenvectors in the corresponding columns after that rearrangement
    # v[:, [1,0]] = v[:, [0,1]]
    # v = np.flip(v, axis=1)
    w,v = eigh(S, subset_by_value=[np.sum(np.trace(np.diag(eigh(S, eigvals_only=True))))*prop, np.inf])
    # x = eigh(S, subset_by_value=[np.sum(eigh(S, eigvals_only=True))*prop, np.inf],eigvals_only=True)
    # w,v = get_eig(S, len(x))
    w = np.diag(np.sort(w)[::-1])
    v = np.flip(v, axis=1)
    return w, v

def project_image(image, U):
    # Your implementation goes here!
    sum = 0
    for i in range(len(U[0])):
        sum += np.dot(np.dot(U[:, i].T, image), U[:, i])
    return sum

def display_image(orig, proj):
    # Your implementation goes here!
    # reshape
    orig = np.rot90(np.reshape(orig, (32, 32)),3)
    proj = np.rot90(np.reshape(proj, (32, 32)),3)
    # create a figure with one row of two subplots
    fig, (axOg, axPj) = plt.subplots(1, 2)
    # Title
    axOg.set_title("Original")
    axPj.set_title("Projection")
    # imshow
    fig.colorbar(axOg.imshow(orig, aspect='equal'), ax = axOg)
    fig.colorbar(axPj.imshow(proj, aspect='equal'), ax = axPj)
    # render
    plt.show()

# x = load_and_center_dataset("YaleB_32x32.npy")
# print(len(x),len(x[0]), np.average(x))
# S = get_covariance(x)
# print(len(S),len(S[0]))
# Lambda, U = get_eig(S, 2)
# print(Lambda)
# print(U)
# Lambda, U = get_eig_prop(S, 0.02)
# print(Lambda)
# print(U)
# projection = project_image(x[0], U)
# print(projection)
# display_image(x[0], projection)