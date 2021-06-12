import csv
import matplotlib.pyplot as plt
import random
import numpy as np
import math


def RMSD(y0, y):
    sum = 0
    for i in range(len(y0)):
        sum += (y0[i] - y[i]) ** 2
    return math.sqrt(sum / len(y0))


def lagrangeFunction(x, y):
    def f(x0):
        if len(x) != len(y):
            return 1
        Ly = 0
        for k in range(len(x)):
            t = 1
            for j in range(len(x)):
                if j != k:
                    t = t * ((x0 - x[j]) / (x[k] - x[j]))
            Ly += t * y[k]
        return Ly
    return f


def LU(A, b, N):

    U = A.copy()
    L = np.eye(N, dtype=np.double)
    P = np.eye(N, dtype=np.double)

    for i in range(N):
        for k in range(i, N):
            if U[i, i] != 0:
                break
            U[[k, k + 1]] = U[[k + 1, k]]
            P[[k, k + 1]] = P[[k + 1, k]]
        fr = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = fr
        U[i + 1:] -= fr[:, np.newaxis] * U[i]

    # podstawianie wprzod
    y = np.zeros_like(b, dtype=np.double)
    y[0] = b[0] / L[0, 0]

    for i in range(1, L.shape[0]):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # podstawianie wstecz
    x = np.zeros_like(y, dtype=np.double)
    x[-1] = y[-1] / U[-1, -1]

    for i in range(U.shape[0] - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]

    return x


def parameters(x, y):

    N = 4 * (len(x) - 1)
    A = np.zeros((N, N))
    b = np.zeros((N, 1))

    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]

        # Sj(xj) = f(xj)
        A[4 * i][4 * i] = 1
        b[4 * i] = y[i]

        # Sj(xj+1) = f(xj+1)
        A[4 * i + 1][4 * i] = 1
        A[4 * i + 1][4 * i + 1] = h
        A[4 * i + 1][4 * i + 2] = h ** 2
        A[4 * i + 1][4 * i + 3] = h ** 3
        b[4 * i + 1] = y[i + 1]

        # Sj'(xj-1) = Sj'(xj)
        A[4 * i + 2][4 * (i - 1) + 1] = 1
        A[4 * i + 2][4 * (i - 1) + 2] = 2 * h
        A[4 * i + 2][4 * (i - 1) + 3] = 3 * (h ** 2)
        A[4 * i + 2][4 * i + 1] = -1
        b[4 * i + 2] = 0

        # Sj''(xj-1) = Sj''(xj)
        A[4 * i + 3][4 * (i - 1) + 2] = 2
        A[4 * i + 3][4 * (i - 1) + 3] = 6 * h
        A[4 * i + 3][4 * i + 2] = -2
        b[4 * i + 3] = 0

    # S0''(x0) = 0 and Sn-1''(xn) = 0
    A[2][2] = 1
    b[2] = 0

    h = x[len(x) - 1] - x[len(x) - 2]
    A[3][4 * (len(x) - 2) + 2] = 2
    A[3][4 * (len(x) - 2) + 3] = 6 * h
    b[3] = 0

    return LU(A, b, N)


def splineFunction(x, y):

    p = parameters(x, y)

    def fun(x0):
        for i in range(0, len(x) - 1):
            if x[i] <= x0 <= x[i + 1]:
                h = x0 - x[i]
                a = p[4 * i]
                b = p[4 * i + 1]
                c = p[4 * i + 2]
                d = p[4 * i + 3]
                return d * (h ** 3) + c * (h ** 2) + b * h + a
    return fun


# Spacerniak w Gdansku

height = []
distance = []
with open('SpacerniakGdansk.csv') as f:
    csvReader = csv.reader(f)
    n = 0
    for row in csvReader:
        if n != 0:
            distance.append(float(row[0]))
            height.append(float(row[1]))
        n += 1


# interpolacja Lagrange'a

# 5, 10, 15, 20 punktow Spacerniak Gdansk rozmieszczenie rownomierne
# 5, 10, 15 punktow Spacerniak Gdansk rozmieszczenie nierownomierne
ROWNOMIERNE = True
k = 0
# jesli range 7 to tylko raz generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
# jesli range 10 to dwa razy generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
for iter in range(10):
    if iter == 4 or iter == 7:
        k = 0
        ROWNOMIERNE = False
    n = 5 * (k + 1) - 1
    part = round(len(distance) / n)
    LX = []
    Y = []
    if ROWNOMIERNE == True:
        for i in range(n):
            LX.append(distance[i * part])
            Y.append(height[i * part])
    else:
        LX.append(distance[0])
        Y.append(height[0])
        # wybieranie losowych wezlow bez powtorzen
        data = [x for x in range(1, len(distance)-1)]
        random.shuffle(data)
        for i in range(n - 1):
            LX.append(distance[data[i]])
            Y.append(height[data[i]])
    LX.append(distance[len(distance) - 1])
    Y.append(height[len(height) - 1])

    # funkcja interpolacyjna
    f = lagrangeFunction(LX, Y)
    FY = []
    for i in range(len(distance)):
        FY.append(f(distance[i]))
    # punkty y wyliczone metoda lagrange'a
    LY = []
    for i in range(len(LX)):
        LY.append(f(LX[i]))

    # wykres
    plt.plot(distance, height, 'b.', label='dane')
    plt.plot(distance, FY, 'y-', label='funkcja interpolacyjna')
    plt.plot(LX, LY, 'r.', label='wezly')
    plt.legend()
    plt.xlabel("Dystans")
    plt.ylabel("Wysokosc")
    plt.title("LAGRANGE: Spacerniak Gdansk dla {} punktow".format(n + 1))
    plt.grid()
    plt.show()

    error = RMSD(FY, height)
    print("LAGRANGE: Spacerniak Gdansk dla {} punktow".format(n + 1))
    print("Blad interpolacji {}".format(error))
    print()

    k += 1


# interpolacja funkcje sklejane trzeciego stopnia

# 5, 10, 15, 20 punktow Spacerniak Gdansk rozmieszczenie rownomierne
# 5, 10, 15 punktow Spacerniak Gdansk rozmieszczenie nierownomierne
ROWNOMIERNE = True
k = 0
# jesli range 7 to tylko raz generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
# jesli range 10 to dwa razy generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
for iter in range(10):
    if iter == 4 or iter == 7:
        k = 0
        ROWNOMIERNE = False
    n = 5 * (k + 1) - 1
    part = round(len(distance) / n)
    SX = []
    Y = []
    if ROWNOMIERNE == True:
        for i in range(n):
            SX.append(distance[i * part])
            Y.append(height[i * part])
    else:
        SX.append(distance[0])
        Y.append(height[0])
        # wybieranie losowych wezlow bez powtorzen
        data = [x for x in range(1, len(distance)-1)]
        random.shuffle(data)
        for i in range(n - 1):
            SX.append(distance[data[i]])
            Y.append(height[data[i]])
    SX.append(distance[len(distance) - 1])
    Y.append(height[len(height) - 1])

    # funkcja interpolacyjna
    fun = splineFunction(SX, Y)
    FY = []
    for i in range(len(distance)):
        FY.append(fun(distance[i]))
    # punkty y wyliczone metoda splajnow
    SY = []
    for i in range(len(SX)):
        SY.append(fun(SX[i]))

    # wykres
    plt.plot(distance, height, 'b.', label='dane')
    plt.plot(distance, FY, 'y-', label='funkcja interpolacyjna')
    plt.plot(SX, SY, 'r.', label='wezly')
    plt.legend()
    plt.xlabel("Dystans")
    plt.ylabel("Wysokosc")
    plt.title("SPLAJNY: Spacerniak Gdansk dla {} punktow".format(n + 1))
    plt.grid()
    plt.show()

    error = RMSD(FY, height)
    print("SPLAJNY: Spacerniak Gdansk dla {} punktow".format(n + 1))
    print("Blad interpolacji {}".format(error))
    print()

    k += 1


# Mount Everest

height = []
distance = []
with open('MountEverest.csv') as f:
    csvReader = csv.reader(f)
    n = 0
    for row in csvReader:
        if n != 0:
            distance.append(float(row[0]))
            height.append(float(row[1]))
        n += 1


# interpolacja Lagrange'a

# 5, 10, 15, 20 punktow Mount Everest rozmieszczenie rownomierne
# 5, 10, 15 punktow Mount Everest rozmieszczenie nierownomierne
ROWNOMIERNE = True
k = 0
# jesli range 7 to tylko raz generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
# jesli range 10 to dwa razy generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
for iter in range(10):
    if iter == 4 or iter == 7:
        k = 0
        ROWNOMIERNE = False
    n = 5 * (k + 1) - 1
    part = round(len(distance) / n)
    LX = []
    Y = []
    if ROWNOMIERNE == True:
        for i in range(n):
            LX.append(distance[i * part])
            Y.append(height[i * part])
    else:
        LX.append(distance[0])
        Y.append(height[0])
        # wybieranie losowych wezlow bez powtorzen
        data = [x for x in range(1, len(distance)-1)]
        random.shuffle(data)
        for i in range(n - 1):
            LX.append(distance[data[i]])
            Y.append(height[data[i]])
    LX.append(distance[len(distance) - 1])
    Y.append(height[len(height) - 1])

    # funkcja interpolacyjna
    f = lagrangeFunction(LX, Y)
    FY = []
    for i in range(len(distance)):
        FY.append(f(distance[i]))
    # punkty y wyliczone metoda lagrange'a
    LY = []
    for i in range(len(LX)):
        LY.append(f(LX[i]))

    # wykres
    plt.plot(distance, height, 'b.', label='dane')
    plt.plot(distance, FY, 'y-', label='funkcja interpolacyjna')
    plt.plot(LX, LY, 'r.', label='wezly')
    plt.legend()
    plt.xlabel("Dystans")
    plt.ylabel("Wysokosc")
    plt.title("LAGRANGE: Mount Everest dla {} punktow".format(n + 1))
    plt.grid()
    plt.show()

    error = RMSD(FY, height)
    print("LAGRANGE: Mount Everest dla {} punktow".format(n + 1))
    print("Blad interpolacji {}".format(error))
    print()

    k += 1


# interpolacja funkcje sklejane trzeciego stopnia

# 5, 10, 15, 20 punktow Mount Everest rozmieszczenie rownomierne
# 5, 10, 15 punktow Mount Everest rozmieszczenie nierownomierne
ROWNOMIERNE = True
k = 0
# jesli range 7 to tylko raz generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
# jesli range 10 to dwa razy generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
for iter in range(10):
    if iter == 4 or iter == 7:
        k = 0
        ROWNOMIERNE = False
    n = 5 * (k + 1) - 1
    part = round(len(distance) / n)
    SX = []
    Y = []
    if ROWNOMIERNE == True:
        for i in range(n):
            SX.append(distance[i * part])
            Y.append(height[i * part])
    else:
        SX.append(distance[0])
        Y.append(height[0])
        # wybieranie losowych wezlow bez powtorzen
        data = [x for x in range(1, len(distance)-1)]
        random.shuffle(data)
        for i in range(n - 1):
            SX.append(distance[data[i]])
            Y.append(height[data[i]])
    SX.append(distance[len(distance) - 1])
    Y.append(height[len(height) - 1])

    # funkcja interpolacyjna
    fun = splineFunction(SX, Y)
    FY = []
    for i in range(len(distance)):
        FY.append(fun(distance[i]))
    # punkty y wyliczone metoda splajnow
    SY = []
    for i in range(len(SX)):
        SY.append(fun(SX[i]))

    # wykres
    plt.plot(distance, height, 'b.', label='dane')
    plt.plot(distance, FY, 'y-', label='funkcja interpolacyjna')
    plt.plot(SX, SY, 'r.', label='wezly')
    plt.legend()
    plt.xlabel("Dystans")
    plt.ylabel("Wysokosc")
    plt.title("SPLAJNY: Mount Everest dla {} punktow".format(n + 1))
    plt.grid()
    plt.show()

    error = RMSD(FY, height)
    print("SPLAJNY: Mount Everest dla {} punktow".format(n + 1))
    print("Blad interpolacji {}".format(error))
    print()

    k += 1


# Glebia Challengera
height = []
distance = []
with open('GlebiaChallengera.csv') as f:
    csvReader = csv.reader(f)
    n = 0
    for row in csvReader:
        if n != 0:
            distance.append(float(row[0]))
            height.append(float(row[1]))
        n += 1


# interpolacja Lagrange'a

# 5, 10, 15, 20 punktow Glebia Challengera rozmieszczenie rownomierne
# 5, 10, 15 punktow Glebia Challengera rozmieszczenie nierownomierne
ROWNOMIERNE = True
k = 0
# jesli range 7 to tylko raz generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
# jesli range 10 to dwa razy generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
for iter in range(10):
    if iter == 4 or iter == 7:
        k = 0
        ROWNOMIERNE = False
    n = 5 * (k + 1) - 1
    part = round(len(distance) / n)
    LX = []
    Y = []
    if ROWNOMIERNE == True:
        for i in range(n):
            LX.append(distance[i * part])
            Y.append(height[i * part])
    else:
        LX.append(distance[0])
        Y.append(height[0])
        # wybieranie losowych wezlow bez powtorzen
        data = [x for x in range(1, len(distance)-1)]
        random.shuffle(data)
        for i in range(n - 1):
            LX.append(distance[data[i]])
            Y.append(height[data[i]])
    LX.append(distance[len(distance) - 1])
    Y.append(height[len(height) - 1])

    # funkcja interpolacyjna
    f = lagrangeFunction(LX, Y)
    FY = []
    for i in range(len(distance)):
        FY.append(f(distance[i]))
    # punkty y wyliczone metoda lagrange'a
    LY = []
    for i in range(len(LX)):
        LY.append(f(LX[i]))

    # wykres
    plt.plot(distance, height, 'b.', label='dane')
    plt.plot(distance, FY, 'y-', label='funkcja interpolacyjna')
    plt.plot(LX, LY, 'r.', label='wezly')
    plt.legend()
    plt.xlabel("Dystans")
    plt.ylabel("Wysokosc")
    plt.title("LAGRANGE: Glebia Challengera dla {} punktow".format(n + 1))
    plt.grid()
    plt.show()

    error = RMSD(FY, height)
    print("LAGRANGE: Glebia Challengera dla {} punktow".format(n + 1))
    print("Blad interpolacji {}".format(error))
    print()

    k += 1


# interpolacja funkcje sklejane trzeciego stopnia

# 5, 10, 15, 20 punktow Glebia Challengera rozmieszczenie rownomierne
# 5, 10, 15 punktow Glebia Challengera rozmieszczenie nierownomierne
ROWNOMIERNE = True
k = 0
# jesli range 7 to tylko raz generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
# jesli range 10 to dwa razy generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
for iter in range(10):
    if iter == 4 or iter == 7:
        k = 0
        ROWNOMIERNE = False
    n = 5 * (k + 1) - 1
    part = round(len(distance) / n)
    SX = []
    Y = []
    if ROWNOMIERNE == True:
        for i in range(n):
            SX.append(distance[i * part])
            Y.append(height[i * part])
    else:
        SX.append(distance[0])
        Y.append(height[0])
        # wybieranie losowych wezlow bez powtorzen
        data = [x for x in range(1, len(distance)-1)]
        random.shuffle(data)
        for i in range(n - 1):
            SX.append(distance[data[i]])
            Y.append(height[data[i]])
    SX.append(distance[len(distance) - 1])
    Y.append(height[len(height) - 1])

    # funkcja interpolacyjna
    fun = splineFunction(SX, Y)
    FY = []
    for i in range(len(distance)):
        FY.append(fun(distance[i]))
    # punkty y wyliczone metoda splajnow
    SY = []
    for i in range(len(SX)):
        SY.append(fun(SX[i]))

    # wykres
    plt.plot(distance, height, 'b.', label='dane')
    plt.plot(distance, FY, 'y-', label='funkcja interpolacyjna')
    plt.plot(SX, SY, 'r.', label='wezly')
    plt.legend()
    plt.xlabel("Dystans")
    plt.ylabel("Wysokosc")
    plt.title("SPLAJNY: Glebia Challengera dla {} punktow".format(n + 1))
    plt.grid()
    plt.show()

    error = RMSD(FY, height)
    print("SPLAJNY: Glebia Challengera dla {} punktow".format(n + 1))
    print("Blad interpolacji {}".format(error))
    print()

    k += 1


# Wielki Kanion Kolorado
height = []
distance = []
with open('WielkiKanionKolorado.csv') as f:
    csvReader = csv.reader(f)
    n = 0
    for row in csvReader:
        if n != 0:
            distance.append(float(row[0]))
            height.append(float(row[1]))
        n += 1


# interpolacja Lagrange'a

# 5, 10, 15, 20 punktow Wielki Kanion Kolorado rozmieszczenie rownomierne
# 5, 10, 15 punktow Wielki Kanion Kolorado rozmieszczenie nierownomierne
ROWNOMIERNE = True
k = 0
# jesli range 7 to tylko raz generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
# jesli range 10 to dwa razy generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
for iter in range(10):
    if iter == 4 or iter == 7:
        k = 0
        ROWNOMIERNE = False
    n = 5 * (k + 1) - 1
    part = round(len(distance) / n)
    LX = []
    Y = []
    if ROWNOMIERNE == True:
        for i in range(n):
            LX.append(distance[i * part])
            Y.append(height[i * part])
    else:
        LX.append(distance[0])
        Y.append(height[0])
        # wybieranie losowych wezlow bez powtorzen
        data = [x for x in range(1, len(distance)-1)]
        random.shuffle(data)
        for i in range(n - 1):
            LX.append(distance[data[i]])
            Y.append(height[data[i]])
    LX.append(distance[len(distance) - 1])
    Y.append(height[len(height) - 1])

    # funkcja interpolacyjna
    f = lagrangeFunction(LX, Y)
    FY = []
    for i in range(len(distance)):
        FY.append(f(distance[i]))
    # punkty y wyliczone metoda lagrange'a
    LY = []
    for i in range(len(LX)):
        LY.append(f(LX[i]))

    # wykres
    plt.plot(distance, height, 'b.', label='dane')
    plt.plot(distance, FY, 'y-', label='funkcja interpolacyjna')
    plt.plot(LX, LY, 'r.', label='wezly')
    plt.legend()
    plt.xlabel("Dystans")
    plt.ylabel("Wysokosc")
    plt.title("LAGRANGE: Wielki Kanion Kolorado dla {} punktow".format(n + 1))
    plt.grid()
    plt.show()

    error = RMSD(FY, height)
    print("LAGRANGE: Wielki Kanion Kolorado dla {} punktow".format(n + 1))
    print("Blad interpolacji {}".format(error))
    print()

    k += 1


# interpolacja funkcje sklejane trzeciego stopnia

# 5, 10, 15, 20 punktow Wielki Kanion Kolorado rozmieszczenie rownomierne
# 5, 10, 15 punktow Wielki Kanion Kolorado rozmieszczenie nierownomierne
ROWNOMIERNE = True
k = 0
# jesli range 7 to tylko raz generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
# jesli range 10 to dwa razy generuja sie wykresy dla 5, 10 i 15 punktow rozmieszczenie nierownomierne
for iter in range(10):
    if iter == 4 or iter == 7:
        k = 0
        ROWNOMIERNE = False
    n = 5 * (k + 1) - 1
    part = round(len(distance) / n)
    SX = []
    Y = []
    if ROWNOMIERNE == True:
        for i in range(n):
            SX.append(distance[i * part])
            Y.append(height[i * part])
    else:
        SX.append(distance[0])
        Y.append(height[0])
        # wybieranie losowych wezlow bez powtorzen
        data = [x for x in range(1, len(distance)-1)]
        random.shuffle(data)
        for i in range(n - 1):
            SX.append(distance[data[i]])
            Y.append(height[data[i]])
    SX.append(distance[len(distance) - 1])
    Y.append(height[len(height) - 1])

    # funkcja interpolacyjna
    fun = splineFunction(SX, Y)
    FY = []
    for i in range(len(distance)):
        FY.append(fun(distance[i]))
    # punkty y wyliczone metoda splajnow
    SY = []
    for i in range(len(SX)):
        SY.append(fun(SX[i]))

    # wykres
    plt.plot(distance, height, 'b.', label='dane')
    plt.plot(distance, FY, 'y-', label='funkcja interpolacyjna')
    plt.plot(SX, SY, 'r.', label='wezly')
    plt.legend()
    plt.xlabel("Dystans")
    plt.ylabel("Wysokosc")
    plt.title("SPLAJNY: Wielki Kanion Kolorado dla {} punktow".format(n + 1))
    plt.grid()
    plt.show()

    error = RMSD(FY, height)
    print("SPLAJNY: Wielki Kanion Kolorado dla {} punktow".format(n + 1))
    print("Blad interpolacji {}".format(error))
    print()

    k += 1

