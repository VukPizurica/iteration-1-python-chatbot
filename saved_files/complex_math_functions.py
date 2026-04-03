import math

def calculate_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2.0 * h)

def calculate_integral(f, a, b, num_points=1000):
    x = np.linspace(a, b, num_points)
    y = f(x)
    return np.trapz(y, x)

def solve_quadratic_equation(a, b, c):
    d = b**2 - 4*a*c
    if d < 0:
        return "No real solutions"
    elif d == 0:
        x = -b / (2*a)
        return x
    else:
        x1 = (-b + math.sqrt(d)) / (2*a)
        x2 = (-b - math.sqrt(d)) / (2*a)
        return x1, x2

def calculate_matrix_determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    else:
        det = 0
        for i in range(len(matrix)):
            minor = [[matrix[j][k] for k in range(len(matrix)) if k != i] for j in range(1, len(matrix))]
            det += ((-1)**i)*matrix[0][i]*calculate_matrix_determinant(minor)
        return det

def calculate_eigenvalues(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return eigenvalues

def calculate_eigenvectors(matrix):
    eigenvectors = np.linalg.eig(matrix)[1]
    return eigenvectors