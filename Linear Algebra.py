import math
import copy
import numpy
import random

def dimensions(mat):
    assert not isJagged(mat)
    return [len(mat), len(mat[0])]

def isJagged(mat):
    size = len(mat[0])
    result = False
    for i in mat:
        if len(i) != size:
            result = True
    return result

def isSquare(mat):
    return dimensions(mat)[0] == dimensions(mat)[1]

def blank(dimensions):
    return [[0 for x in range(dimensions[1])] for y in range(dimensions[0])]

def getRow(mat, ind):
    return mat[ind]

def getCol(mat, ind):
    vec = []
    for i in mat:
        vec.append(i[ind])
    return vec

def setRow(mat, ind, vec):
    assert dimensions(mat)[1] == len(vec)
    mat[ind] = vec
    return mat

def setCol(mat, ind, vec):
    assert dimensions(mat)[0] == len(vec)
    for i in range(len(mat)):
        mat[i][ind] = vec[i]
    return mat

def delRow(mat, ind):
    del mat[ind]
    return mat

def delCol(mat, ind):
    for i in mat:
        del i[ind]
    return mat

def summable(mat0, mat1):
    return dimensions(mat0)[0] == dimensions(mat1)[0] and dimensions(mat0)[1] == dimensions(mat1)[1]

def add(mat0, mat1):
    assert summable(mat0, mat1)
    newMat = copy.deepcopy(mat0)
    for row in range(len(newMat)):
        for col in range(len(newMat[row])):
            newMat[row][col] += mat1[row][col]
    return newMat

def negate(mat):
    newMat = copy.deepcopy(mat)
    for row in range(len(newMat)):
        for col in range(len(newMat[row])):
            newMat[row][col] = -1 * newMat[row][col]
    return newMat

def subtract(mat0, mat1):
    return add(mat0, negate(mat1))

def scalar(sca, mat):
    return [[sca * col for col in row] for row in mat]

def dot(vec0, vec1):
    assert len(vec0) == len(vec1)
    a = 0
    for i in range(len(vec0)):
        a += vec0[i] * vec1[i]
    return a

def multipliable(mat0, mat1):
    return dimensions(mat0)[1] == dimensions(mat1)[0]

def multiply(mat0, mat1):
    assert multipliable(mat0, mat1)
    product = []
    for i in range(dimensions(mat0)[0]):
        row = []
        for j in range(dimensions(mat1)[1]):
            row.append(dot(getRow(mat0, i), getCol(mat1, j)))
        product.append(row)
    return product

def strassenCall(mat0, mat1):
    assert multipliable(mat0, mat1)
    return unpad(strassen(pad(mat0, max(dimensions(mat0) + dimensions(mat1))), pad(mat1, max(dimensions(mat0) + dimensions(mat1)))), [dimensions(mat0)[0], dimensions(mat1)[1]])

def strassen(mat0, mat1):
    if len(mat0) == 1:
        return [[mat0[0][0] * mat1[0][0]]]
    else:
        a00, a01, a10, a11 = split(mat0)
        b00, b01, b10, b11 = split(mat1)
        return combine(c00(a00, a01, a11, b00, b10, b11), c01(a00, a01, b01, b11), c10(a10, a11, b00, b10), c11(a00, a10, a11, b00, b01, b11))

def pad(mat, size = None):
    newMat = copy.deepcopy(mat)
    if size == None: size = max(dimensions(newMat))
    size = 2**math.ceil(math.log(size, 2))
    newMat.extend([[0 for i in range(dimensions(newMat)[1])] for j in range(size - dimensions(newMat)[0])])
    while not isSquare(newMat):
        for k in newMat:
            k.append(0)
    return newMat

def unpad(mat, targetDim):
    newMat = copy.deepcopy(mat)
    for row in range(targetDim[0], dimensions(newMat)[0]):
        delRow(newMat, dimensions(newMat)[0] - 1)
    for col in range(targetDim[1], dimensions(newMat)[1]):
        delCol(newMat, dimensions(newMat)[1] - 1)
    return newMat

def split(mat): # Written by Kristofer Heller
    newMat = copy.deepcopy(mat)
    C1,C2,C3,C4 = [],[],[],[]
    for i in range(dimensions(newMat)[0]//2):
        C1.append([])
        for j in range(dimensions(newMat)[1]//2):
            C1[i].append(newMat[i][j])
    for i in range(dimensions(newMat)[0]//2):
        C2.append([])
        for j in range(dimensions(newMat)[1]//2,dimensions(newMat)[1]):
            C2[i].append(newMat[i][j])
    for i in range(dimensions(newMat)[0]//2, dimensions(newMat)[0]):
        C3.append([])
        for j in range(dimensions(newMat)[1]//2):
            C3[i-(dimensions(newMat)[0]//2)].append(newMat[i][j])
    for i in range(dimensions(newMat)[0]//2, dimensions(newMat)[0]):
        C4.append([])
        for j in range(dimensions(newMat)[1]//2,dimensions(newMat)[1]):
            C4[i-(dimensions(newMat)[0]//2)].append(newMat[i][j])
    return C1,C2,C3,C4

def combine(A,B,C,D): # Written by Kristofer Heller
    newMat = []
    newMat.extend(A)
    for i in range(dimensions(newMat)[0]):
        newMat[i].extend(B[i])
    newMat.extend(C)
    for i in range(dimensions(A)[0],2*dimensions(A)[0]):
        newMat[i].extend(D[i-dimensions(A)[0]])
    return newMat

def c00(a00, a01, a11, b00, b10, b11):
    return add(subtract(add(m0(a00, a11, b00, b11), m3(a11, b00, b10)), m4(a00, a01, b11)), m6(a01, a11, b10, b11))

def c01(a00, a01, b01, b11):
    return add(m2(a00, b01, b11), m4(a00, a01, b11))

def c10(a10, a11, b00, b10):
    return add(m1(a10, a11, b00), m3(a11, b00, b10))

def c11(a00, a10, a11, b00, b01, b11):
    return add(add(subtract(m0(a00, a11, b00, b11), m1(a10, a11, b00)), m2(a00, b01, b11)), m5(a00, a10, b00, b01))

def m0(a00, a11, b00, b11):
    return strassen(add(a00, a11), add(b00, b11))

def m1(a10, a11, b00):
    return strassen(add(a10, a11), b00)

def m2(a00, b01, b11):
    return strassen(a00, subtract(b01, b11))

def m3(a11, b00, b10):
    return strassen(a11, subtract(b10, b00))

def m4(a00, a01, b11):
    return strassen(add(a00, a01), b11)

def m5(a00, a10, b00, b01):
    return strassen(subtract(a10, a00), add(b00, b01))

def m6(a01, a11, b10, b11):
    return strassen(subtract(a01, a11), add(b10, b11))

def determinant(mat):
    assert isSquare(mat)
    assert dimensions(mat)[0] > 1
    terms = []
    if dimensions(mat)[0] > 2:
        for j in range(dimensions(mat)[1]):
            newMat = copy.deepcopy(mat)
            delRow(newMat, 0)
            delCol(newMat, j)
            mult = mat[0][j] * math.pow(-1, j)
            det = determinant(newMat)
            terms.append(mult*det)
        return sum(terms)
    else:
        return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]

def normalize(vec):
    eigenvalue = max(vec)[0]
    eigenvector = [[row[0] / eigenvalue] for row in vec]
    return eigenvalue, eigenvector

def eigen(mat):
    assert isSquare(mat)
    eigenvector = [[random.random()] for i in mat]
    for i in range(30):
        eigenvalue, eigenvector = normalize(multiply(mat, eigenvector))
    return eigenvalue, eigenvector

def augment(mat, vec):
    for row in range(len(mat)):
        mat[row].append(vec[row][0])
    return mat

def gaussian(mat, vec): # Written by Martin Thoma
    assert isSquare(mat)
    A = augment(mat, vec)
    n = len(A)
    for i in range(0, n):
        maxEl = abs(A[i][i])
        maxRow = i
        for k in range(i+1, n):
            if abs(A[k][i]) > maxEl:
                maxEl = abs(A[k][i])
                maxRow = k
        for k in range(i, n+1):
            tmp = A[maxRow][k]
            A[maxRow][k] = A[i][k]
            A[i][k] = tmp
        for k in range(i+1, n):
            c = -A[k][i]/A[i][i]
            for j in range(i, n+1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]
    x = [0 for i in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = A[i][n]/A[i][i]
        for k in range(i-1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return [[y] for y in x]

def cramer(mat, vec):
    assert isSquare(mat)
    det = determinant(mat)
    assert det != 0
    result = []
    for i in range(dimensions(mat)[0]):
        result.append(determinant(setCol(copy.deepcopy(mat), i, getCol(vec, 0)))/det)
    return result

def jacobi(mat, vec):
    assert isSquare(mat)
    x1 = copy.deepcopy(x0)
    for i in range(30):
        for row in range(len(mat)):
            x0 = copy.deepcopy(x1)
            sig = sum([mat[row][col] * x1[col][0] for col in range(len(mat[row])) if col != row])
            x1[row][0] = (1 / mat[row][row]) * (vec[row][0] - sig)
    return x1

def sor(mat, vec, omega = random.uniform(1, 2)):
    assert isSquare(mat)
    x1 = [[random.random()] for i in mat]
    for i in range(30):
        for row in range(len(mat)):
            x0 = copy.deepcopy(x1)
            sigLess = sum([mat[row][col] * x1[col][0] for col in range(len(mat[row])) if col < row])
            sigGreater = sum([mat[row][col] * x0[col][0] for col in range(len(mat[row])) if col > row])
            x1[row][0] = ((1 - omega) * x0[row][0]) + ((omega / mat[row][row]) * (vec[row][0] - sigLess - sigGreater)) 
    return x1

def gaussSeidel(mat, vec):
    return sor(mat, vec, 1)

def invert(mat):
    assert isSquare(mat)
    return numpy.ndarray.tolist(numpy.linalg.inv(mat))

def interpolation(data):
    mat = []
    vec = []
    for point in data:
        row = []
        for col in range(len(data)):
            row.append(math.pow(point[0], col))
        mat.append(row)
        vec.append([point[1]])
    return multiply(invert(mat), vec)

def regression(data, order):
    mat = blank([order+1, order+1])
    vec = blank([order+1, 1])
    for row in range(order+1):
        for point in data:
            for col in range(order+1):
                mat[row][col] += math.pow(point[0], row + col)
            vec[row][0] += (math.pow(point[0], row) * point[1])
    return multiply(invert(mat), vec)
