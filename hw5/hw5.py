import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Q2
    filename = sys.argv[1]
    # csv = "toy.csv"
    # csv = "hw5.csv"
    x, y = [], []
    # with open("toy.csv") as csvfile:
    # with open("hw5.csv") as csvfile:
    with open(filename) as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            x.append(row[0])
            y.append(row[1])
    years = [int(i) for i in x[1:]]
    days = [int(i) for i in y[1:]]
    plt.plot(years, days, color = "green")
    plt.xlabel('Year')
    plt.ylabel('Number of frozen days')
    plt.show()

    n = len(years)
    X = np.zeros((n, 2), dtype='int64')
    Y = np.empty([n, ], dtype='int64')
    for i in range(n):
        X[i][0] = 1
        X[i][1] = years[i]
        Y[i] = days[i]

    print("Q3a:")
    print(X)

    print("Q3b:")
    print(Y)

    XT = np.transpose(X)
    Z = np.dot(XT, X)
    print("Q3c:")
    print(Z)

    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)

    PI = np.dot(I, XT)
    print("Q3e:")
    print(PI)

    hat_beta = np.dot(PI, Y)
    print("Q3f:")
    print(hat_beta)

    b0 = hat_beta[0]
    b1 = hat_beta[1]

    y_test = b0 + b1 * 2021
    print("Q4: " + str(y_test))

    if b1 > 0: Symbol = ">"
    elif b1 == 0: Symbol = "="
    else: Symbol = "<"
    print("Q5a: " + Symbol)
    print("Q5b: " + "The sign '<' for Mendota ice means as the year increases, the number of frozen days decreases.")

    x_star = (0 - b0) / b1
    print("Q6a: " + str(x_star))
    print("Q6b: " + "Since the number of frozen days is predicted to decrease every year, it's a compelling prediction that around the year 2455, the number of frozen days will be zero.")