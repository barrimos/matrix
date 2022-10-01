import random
class Matrix:
    def __init__(self, name, rows, cols, nums = 0) -> None:
        self.name = name
        self.rows = rows
        self.cols = cols
        self.nums = nums
    def create(self) -> None:
        self.matrix = []
        for i in range(self.rows):
            self.matrix.append([])
            for j in range(len(self.matrix[i]) + self.cols):
                self.matrix[i].append(self.nums)
        return self.matrix
    def create_random(self, start = 0, end = 255) -> None:
        self.matrix = []
        for i in range(self.rows):
            self.matrix.append([])
            for j in range(len(self.matrix[i]) + self.cols):
                randNum = random.randint(start, end)
                self.matrix[i].append(randNum)
        return self.matrix

# Error
def error():
    try:
        pass
    except:
        pass
    finally:
        pass

# Max column
def maxcol(matrix):
    maxcol = max([len(matrix[i]) for i in range(len(matrix))])
    return maxcol

# Check maximum rows or columns of matrix.
def sizemax(matrix):
    size = max(len(matrix), maxcol(matrix))
    return size

# Check properties matrix.
def checkProps(matrix):
    rows = len(matrix)
    columns = maxcol(matrix)
    matrixName = "Unnamed"
    variables = dict(globals())
    for name in variables:
        if variables[name] is matrix:
            matrixName = name
            break
    return {'name': matrixName, 'rows': rows, 'columns': columns}

# Check number of column in first rows of the first matrix should be equal to first columns of the second matrix.
def isSizeEqual(matrixA, matrixB, operator):
    """
    operator\n
    plusminus or multiply
    """
    # check size first row in 1st and size first column in 2nd
    col1st = maxcol(matrixA)
    row1st = len(matrixA)
    col2nd = maxcol(matrixB)
    row2nd = len(matrixB)

    if operator == "plusminus":
        if (col1st == col2nd) & (row1st == row2nd): return True
    elif operator == "multiply":
        if col1st == row2nd: return True
    else: return False

# A == B : A[i*j] == B[i*j] and A[i][j] == B[i][j]
def isSameMatrix(matrixA, matrixB):
    # First check size by multiply row with column made it to one number
    size_matrix_A = len(matrixA) * maxcol(matrixA)
    size_matrix_B = len(matrixB) * maxcol(matrixB)

    if size_matrix_A == size_matrix_B:
        # # Loop check every members for compare matrix
        # for x in range(len(matrixA)):
        #     for y in range(len(matrixA[x])):
        #         if matrixA[x][y] == matrixB[x][y]:
        #             continue
        #         else: return False
        # return True
        # ------- Alternate Method --------
        z = zip(matrixA, matrixB)
        for i, j in z:
            inz = zip(i, j)
            for n, m in inz:
                if n == m:
                    continue
                else: return False
        return True
    else: return False

def isSquareMatrix(matrix):
    max = sizemax(matrix)
    if len(matrix) == max:
        for i in range(len(matrix)):
            if len(matrix[i]) == max:
                continue
            else: return False
    else: return False
    return True

# Generate to N x M matrix by adding zeros make all columns are equal.
def addzeros(matrix):
    # check each rows that which columns are not equal to maxcol.
    for j in range(len(matrix)):
        for k in range(maxcol(matrix) - len(matrix[j])):
            matrix[j].append(0)
    return matrix

# Generate to N x N matrix by adding zeros following maximum rows or columns to makes rows and columns equal.
def squareMatrix(matrix):
    size = sizemax(matrix)

    for i in range(size):
        if i > len(matrix) - 1: matrix.append([])
        for j in range(size):
            if j > len(matrix[i]) - 1: matrix[i].append(0)
    return matrix

# # Generate matrix to identity matrix.
def identity(matrix):
    # Identity matrix shound be square matrix
    if not isSquareMatrix(matrix): return "The matrix is not square."
    size = sizemax(squareMatrix(matrix))

    identity_matrix = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    return identity_matrix

def plus_minus(matrixA, matrixB, operator):
    """
    operator:\n
    plus, (+), p or minus, (-), m
    """
    # Check the matrix size is equal
    if not isSizeEqual(matrixA, matrixB, "plusminus"): return "1st matrix is NOT EQUAL to 2nd matrix"

    res_matrix = [[0 for j in range(len(matrixA[0]))] for i in range(len(matrixA))]

    for i in range(len(matrixA)):
        for j in range(len(matrixA[i])):
            if operator == "plus" or operator == "+" or operator == "p":
                res_matrix[i][j] = matrixA[i][j] + matrixB[i][j]
            if operator == "minus" or operator == "-" or operator == "m":
                res_matrix[i][j] = matrixA[i][j] - matrixB[i][j]
    return res_matrix


def transpose(matrix):
    # Make equalize each row equal by adding zeros based on largest row
    matrix = addzeros(matrix)

    # Create zero matrix to store result matrix based on original row col
    transpose_matrix = [[0 for i in range(len(matrix))] for j in range(len(matrix[0]))]

    for i in range(len(transpose_matrix)):
        for j in range(len(transpose_matrix[i])):
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
    #         transpose_matrix[i][j] = matrix[j][i]

    return transpose_matrix

def multiplyScalar(matrix, scalar):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = round(scalar * matrix[i][j], 3)

    return matrix

def multiply(matrixA, matrixB):
    # Make equalize each row equal by adding zeros based on largest row
    matrixA = addzeros(matrixA)
    matrixB = addzeros(matrixB)

    # Check if multiplication is Possible.
    if not isSizeEqual(matrixA, matrixB, "multiply"): return "Rows of 1st matrix is NOT EQUAL to Columns of 2nd matrix"

    # Create zero matrix to store result matrix based on number row of 1st matrix with number column of 2nd matrix
    multiply_matrix = [[0 for i in range(len(matrixB[0]))] for z in range(len(matrixA))]

    for i in range(len(matrixA)):
        for j in range(len(matrixB[0])):
            for k in range(len(matrixB)):
                multiply_matrix[i][j] += matrixA[i][k] * matrixB[k][j]

    return multiply_matrix

def determinant(matrix, k = 1):
    # use Bareiss algorithm
    # determinant matrix should be square
    if not isSquareMatrix(matrix): return "The matrix is not square."
    p = 1
    pivot = 0
    prev_matrix = [x[:] for x in matrix]
    det_matrix = [[0 for i in range(len(prev_matrix[0]))] for j in range(len(prev_matrix))]

    for r in range(len(prev_matrix)):
        for i in range(len(det_matrix)):
            for j in range(len(det_matrix[i])):
                if i == r:
                    det_matrix[i][j] = prev_matrix[i][j]
                else:
                    det_matrix[i][j] = 0 if p == 0 else ((prev_matrix[pivot][pivot] * prev_matrix[i][j]) - (prev_matrix[i][pivot] * prev_matrix[pivot][j])) // p
        pivot += 1
        prev_matrix = [x[:] for x in det_matrix]
        p = prev_matrix[r][r]

    # det_matrix[0][0] = prev_matrix[0][0]
    # det_matrix[0][1] = prev_matrix[0][1]
    # det_matrix[0][2] = prev_matrix[0][2]
    # det_matrix[1][0] = ((prev_matrix[0][0] * prev_matrix[1][0]) - (prev_matrix[1][0] * prev_matrix[0][0])) // p
    # det_matrix[1][1] = ((prev_matrix[0][0] * prev_matrix[1][1]) - (prev_matrix[1][0] * prev_matrix[0][1])) // p
    # det_matrix[1][2] = ((prev_matrix[0][0] * prev_matrix[1][2]) - (prev_matrix[1][0] * prev_matrix[0][2])) // p
    # det_matrix[2][0] = ((prev_matrix[0][0] * prev_matrix[2][0]) - (prev_matrix[2][0] * prev_matrix[0][0])) // p
    # det_matrix[2][1] = ((prev_matrix[0][0] * prev_matrix[2][1]) - (prev_matrix[2][0] * prev_matrix[0][1])) // p
    # det_matrix[2][2] = ((prev_matrix[0][0] * prev_matrix[2][2]) - (prev_matrix[2][0] * prev_matrix[0][2])) // p

    # prev_matrix = [x[:] for x in det_matrix]
    # p = prev_matrix[0][0]

    # det_matrix[0][0] = ((prev_matrix[1][1] * prev_matrix[0][0]) - (prev_matrix[0][1] * prev_matrix[1][0])) // p
    # det_matrix[0][1] = ((prev_matrix[1][1] * prev_matrix[0][1]) - (prev_matrix[0][1] * prev_matrix[1][1])) // p
    # det_matrix[0][2] = ((prev_matrix[1][1] * prev_matrix[0][2]) - (prev_matrix[0][1] * prev_matrix[1][2])) // p
    # det_matrix[1][0] = prev_matrix[1][0]
    # det_matrix[1][1] = prev_matrix[1][1]
    # det_matrix[1][2] = prev_matrix[1][2]
    # det_matrix[2][0] = ((prev_matrix[1][1] * prev_matrix[2][0]) - (prev_matrix[2][1] * prev_matrix[1][0])) // p
    # det_matrix[2][1] = ((prev_matrix[1][1] * prev_matrix[2][1]) - (prev_matrix[2][1] * prev_matrix[1][1])) // p
    # det_matrix[2][2] = ((prev_matrix[1][1] * prev_matrix[2][2]) - (prev_matrix[2][1] * prev_matrix[1][2])) // p

    # prev_matrix = [x[:] for x in det_matrix]
    # p = prev_matrix[1][1]

    # det_matrix[0][0] = ((prev_matrix[2][2] * prev_matrix[0][0]) - (prev_matrix[0][2] * prev_matrix[2][0])) // p
    # det_matrix[0][1] = ((prev_matrix[2][2] * prev_matrix[0][1]) - (prev_matrix[0][2] * prev_matrix[2][1])) // p
    # det_matrix[0][2] = ((prev_matrix[2][2] * prev_matrix[0][2]) - (prev_matrix[0][2] * prev_matrix[2][2])) // p
    # det_matrix[1][0] = ((prev_matrix[2][2] * prev_matrix[1][0]) - (prev_matrix[1][2] * prev_matrix[2][0])) // p
    # det_matrix[1][1] = ((prev_matrix[2][2] * prev_matrix[1][1]) - (prev_matrix[1][2] * prev_matrix[2][1])) // p
    # det_matrix[1][2] = ((prev_matrix[2][2] * prev_matrix[1][2]) - (prev_matrix[1][2] * prev_matrix[2][2])) // p
    # det_matrix[2][0] = prev_matrix[2][0]
    # det_matrix[2][1] = prev_matrix[2][1]
    # det_matrix[2][2] = prev_matrix[2][2]

    size = len(det_matrix)

    return (k**size) * det_matrix[r][r]

def minor_cofactor(matrix, posI, posJ, sel = 'm'):
    """
    sel - type m or c\n
    m = Determinant of matrix(i, j)\n
    c = Cofactor of matrix(i, j)
    """
    rows = len(matrix)
    cols = maxcol(matrix)
    select = ['m', 'c']

    if sel not in select: return "m or c"
    if not isSquareMatrix(matrix): return "The matrix is not square."
    if posI not in range(rows) or posJ not in range(cols): return f"Out of range of rows or cols.\nProperties of the matrix is\n{checkProps(matrix)}"
    
    # Copy the matrix to avoid damaging the original matrix.
    dummy_matrix = [x[:] for x in matrix]

    # Delete row and column
    for i in range(len(dummy_matrix)):
        if i == posI:
            del dummy_matrix[i]
    for j in dummy_matrix:
        del j[posJ]

    result = determinant(dummy_matrix) if sel == 'm' else (-1)**(posI + posJ) * determinant(dummy_matrix)
    return result


def triangular(matrix, dir):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if dir == 'up':
                if j >= i:
                    matrix[i][j] = matrix[i][j]
                else: matrix[i][j] = 0
            if dir == 'low':
                if j > i:
                    matrix[i][j] = 0
                else: matrix[i][j] = matrix[i][j]
    return matrix


# def diagonal(matrix):
#     return diagonal_matrix


def inverse(matrix):
    if not isSquareMatrix(matrix): return "The matrix is not square."

    rows = len(matrix)
    cols = maxcol(matrix)
    det_matrix = determinant(matrix)

    # if determinant equal 0 the matrix is not invertible
    if det_matrix == 0: return "The determinant is 0 the matrix is not invertible."

    # Copy matrix for storing new value
    cof_matrix = [x[:] for x in matrix]

    for i in range(rows):
        for j in range(cols):
            cof_matrix[i][j] = minor_cofactor(matrix, i, j, 'c')

    inverse_matrix = multiplyScalar(transpose(cof_matrix), 1 / det_matrix)
    return inverse_matrix

def trace(matrix):
    trace_matrix = 0
    if not isSquareMatrix(matrix): return "The matrix is not square."
    for i in range(len(matrix)):
        trace_matrix += matrix[i][i]
    return trace_matrix

if __name__ == "__main__":

    list_v = [
        [0, 5, 0],
        [7, 5 + 6, 9],
        [0.5, 3.33, 1 / 2]
    ]
    list_vv = [
        [5 - 2, 5, 1 - 1],
        [4 + 3, 11, 3 * 3],
        [1 / 2, 3 + .33, 0.5]
    ]

    list_x = [
        [1, 2, 1],
        [3, 4, 1],
        [1, 1, 1]
    ]
    list_xx = [
        [1, 2, 3],
        [2, 1, 4],
        [2, 1, 1]
    ]
    list_y = [
        [1, 2],
        [1, 2, 3],
        [1, 2],
        [1, 2],
        [1]
    ]
    list_z = [
        [1, 0]
    ]

    sm = [
        [4, 4, 2, 2, 4],
        [2, 4, 3, 2, 1],
        [3, 1, 1, 2, 1],
        [6, 5, 2, 1, 6],
        [4, 3, 5, 5, 3]
    ]

    print(inverse(list_x))
    # newX = Matrix('newX', 3, 3)
    # list_newX = newX.create()

    # print('[', end = '\n')
    # print(*determinant(list_x), sep = '\n', end = '\n')
    # print(']')