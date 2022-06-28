import random
class Matrix:
    def __init__(self,name,rows,cols,nums=0) -> None:
        self.name = name
        self.rows = rows
        self.cols = cols
        self.nums = nums
    def create(self) -> None:
        self.matrix = []
        for i in range(self.rows):
            self.matrix.append([])
            for j in range(len(self.matrix[i])+self.cols):
                self.matrix[i].append(self.nums)
        return self.matrix
    def create_random(self,start=0,end=255) -> None:
        self.matrix = []
        for i in range(self.rows):
            self.matrix.append([])
            for j in range(len(self.matrix[i])+self.cols):
                randNum = random.randint(start,end)
                self.matrix[i].append(randNum)
        return self.matrix

# Max column
def maxcol(matrix):
    maxcol = max(len(matrix[i]) for i in range(len(matrix)))
    return maxcol

# Check maximum rows or columns of matrix.
def sizemax(matrix):
    size = max(len(matrix),maxcol(matrix))
    return size

# Check properties matrix.
def checkProps(matrix):
    rows = len(matrix)
    columns = maxcol(matrix)
    MatrixName = "Unnamed"
    variables = dict(globals())
    for name in variables:
        if variables[name] is matrix:
            MatrixName = name
            break
    return {'name': MatrixName, 'rows': rows, 'columns': columns}

# Check number of column in first rows of the first matrix should be equal to first columns of the second matrix.
def orderIsEqual(matrixA,matrixB,OP):
    # check size first row in 1st and size first column in 2nd
    col1st = maxcol(matrixA)
    row1st = len(matrixA)
    col2nd = maxcol(matrixB)
    row2nd = len(matrixB)

    if OP == "plusminus":
        if (col1st == col2nd) & (row1st == row2nd): return True
    elif OP == "multiply":
        if col1st == row2nd: return True
    else: return False

# A == B : A[n*m] == B[n*m] and a[i][j] == b[i][j]
def sameMatrix(matrixA,matrixB):
    # First check size by multiply row with column made it to one number
    size_Matrices_A = len(matrixA) * maxcol(matrixA)
    size_Matrices_B = len(matrixB) * maxcol(matrixB)

    if size_Matrices_A == size_Matrices_B:
        # # Loop check every members for compare matrix
        # for x in range(len(matrixA)):
        #     for y in range(len(matrixA[x])):
        #         if matrixA[x][y] == matrixB[x][y]:
        #             continue
        #         else: return False
        # return True
        # ------- Alternate Method --------
        z = zip(matrixA,matrixB)
        for i,j in z:
            inz = zip(i,j)
            for n,m in inz:
                if n == m:
                    continue
                else: return False
        return True
    else: return False


# Generate to N x M matrix by adding zeros make all columns are equal.
def addzeros(matrix):
    # check each rows that which columns are not equal to maxcol.
    for j in range(len(matrix)):
        for k in range(maxcol(matrix) - len(matrix[j])):
            matrix[j].append(0)
    return matrix

# Generate to N x N matrix by adding zeros following maximum rows or columns to makes rows and columns equal.
def n_matrix(matrix):
    size = sizemax(matrix)

    for i in range(size):
        if i > len(matrix)-1:
            matrix.append([])
        for j in range(size):
            if j > len(matrix[i])-1: matrix[i].append(0)
    return matrix

# # Generate matrix to identity matrix.
def identity(matrix):
    size = sizemax(n_matrix(matrix))

    identity_matrix = []
    for i in range(size):
        if i > len(identity_matrix)-1: identity_matrix.append([])
        for j in range(size):
            if len(identity_matrix[i]) == i: identity_matrix[i].append(1)
            if j > len(identity_matrix[i])-1: identity_matrix[i].append(0)
    return identity_matrix

def plus(matrixA,matrixB):
    # Check if plus is Possible
    if orderIsEqual(matrixA,matrixB,"plusminus") != True: return ["1st matrix is NOT EQUAL to 2nd matrix"]

    plus_matrix = [[0 for i in range(len(matrixA[0]))] for j in range(len(matrixA))]
    # plus_matrix[0][0] = matrixA[0][0] + matrixB[0][0]
    # plus_matrix[0][1] = matrixA[0][1] + matrixB[0][1]
    # plus_matrix[0][2] = matrixA[0][2] + matrixB[0][2]
    # plus_matrix[1][0] = matrixA[1][0] + matrixB[1][0]
    # plus_matrix[1][1] = matrixA[1][1] + matrixB[1][1]
    # plus_matrix[1][2] = matrixA[1][2] + matrixB[1][2]
    # plus_matrix[2][0] = matrixA[2][0] + matrixB[2][0]
    # plus_matrix[2][1] = matrixA[2][1] + matrixB[2][1]
    # plus_matrix[2][2] = matrixA[2][2] + matrixB[2][2]
    for i in range(len(matrixA)):
        for j in range(len(matrixA[i])):
            plus_matrix[i][j] = matrixA[i][j] + matrixB[i][j]
    return plus_matrix
        


def minus(matrixA,matrixB):
    # Check if minus is Possible
    if orderIsEqual(matrixA,matrixB,"plusminus") != True: return ["1st matrix is NOT EQUAL to 2nd matrix"]

    minus_matrix = [[0 for i in range(len(matrixA[0]))] for j in range(len(matrixA))]
    # minus_matrix[0][0] = matrixA[0][0] - matrixB[0][0]
    # minus_matrix[0][1] = matrixA[0][1] - matrixB[0][1]
    # minus_matrix[0][2] = matrixA[0][2] - matrixB[0][2]
    # minus_matrix[1][0] = matrixA[1][0] - matrixB[1][0]
    # minus_matrix[1][1] = matrixA[1][1] - matrixB[1][1]
    # minus_matrix[1][2] = matrixA[1][2] - matrixB[1][2]
    # minus_matrix[2][0] = matrixA[2][0] - matrixB[2][0]
    # minus_matrix[2][1] = matrixA[2][1] - matrixB[2][1]
    # minus_matrix[2][2] = matrixA[2][2] - matrixB[2][2]
    for i in range(len(matrixA)):
        for j in range(len(matrixA[i])):
            minus_matrix[i][j] = matrixA[i][j] - matrixB[i][j]
    return minus_matrix

def transpose(matrix):
    # Make equalize each row equal by adding zeros based on largest row
    matrix = addzeros(matrix)

    # Create zero matrix to store result matrix based on original row col
    transpose_matrix = [[0 for i in range(len(matrix))] for z in range(len(matrix[0]))]
    
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            transpose_matrix[i][j] = matrix[j][i]

    # ----------- ALTERNATE METHOD -------------

    # transpose_matrix[0][0] += matrix[0][0]
    # transpose_matrix[0][1] += matrix[1][0]
    # transpose_matrix[1][0] += matrix[0][1]
    # transpose_matrix[1][1] += matrix[1][1]
    # transpose_matrix[2][0] += matrix[0][2]
    # transpose_matrix[2][1] += matrix[1][2]

    # for i in range(len(matrix[0])):
    #     for j in range(len(matrix)):
    #         transpose_matrix[i][j] += matrix[j][i]
    return transpose_matrix

def multiplyScalar(matrix,scalar):
    # Create zero matrix to store result matrix based on number row of 1st matrix with number column of 2nd matrix
    scalar_matrix = [[0 for i in range(len(matrix[0]))] for z in range(len(matrix))]

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            scalar_matrix[i][j] = scalar * matrix[i][j]
    return scalar_matrix

def multiply(matrixA,matrixB):
    # Make equalize each row equal by adding zeros based on largest row
    matrixA = addzeros(matrixA)
    matrixB = addzeros(matrixB)

    # Check if multiplication is Possible.
    if orderIsEqual(matrixA,matrixB,"multiply") != True:
        return ["Rows of 1st matrix is NOT EQUAL to Columns of 2nd matrix"]

    # Create zero matrix to store result matrix based on number row of 1st matrix with number column of 2nd matrix
    multiply_matrix = [[0 for i in range(len(matrixB[0]))] for z in range(len(matrixA))]

    for i in range(len(matrixA)):
        for j in range(len(matrixB[0])):
            for k in range(len(matrixB)):
                multiply_matrix[i][j] += matrixA[i][k] * matrixB[k][j]
    return multiply_matrix

# def minormatrix(matrix):
#     (-1)**(1+1) * ((matrix[2][2] * matrix[3][3]) - (matrix[3][2] * matrix[2][3]))
#     return minor_matrix

# def determinant(matrix):
    
#     return det_matrix


# def triUp(matrix):
#     return triUp_matrix


# def triLow(matrix):
#     return triLow_matrix


# def diagonal(matrix):
#     return diagonal_matrix


# def inverse(matrix):
#     # First should be square matrix
#     square_matrix = n_matrix(matrix)
#     # if determinant equal 0 the matrix is not invertible
#     return inverse_matrix

def trace(matrix):
    trace_matrix = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i == j:
                trace_matrix += matrix[i][j]
    return trace_matrix



if __name__ == "__main__":

    list_v = [
        [0,5,0],
        [7,5+6,9],
        [0.5,3.33,1/2]
    ]
    list_vv = [
        [5-5,5,1-1],
        [4+3,11,3*3],
        [1/2,3+.33,0.5]
    ]

    list_x = [
        [2,3,4],
        [1,2,3]
    ]
    list_xx = [
        [6,5,0],
        [-1,12,-9]
    ]
    list_y = [
        [1,2],
        [1,2,3],
        [1,2],
        [1,2],
        [1]
    ]
    list_z = [
        [1,0]
    ]

    sm = [
        [4, 4, 2, 2, 4],
        [2, 4, 3, 2, 1],
        [3, 1, 1, 2, 1],
        [6, 5, 2, 1, 6],
        [4, 3, 5, 5, 3]
    ]
    # -120

    print('[', end='')
    print(*identity(sm), sep=',\n', end='')
    print(']')