import numpy as np
import networkx as nx
from scipy.linalg import solve_triangular


def leer_archivo(input_file_path):

    f = open(input_file_path, 'r')
    n = int(f.readline())
    m = int(f.readline())
    W = np.zeros(shape=(n,n))
    for _ in range(m):
    	line = f.readline()
    	i = int(line.split()[0]) - 1
    	j = int(line.split()[1]) - 1
    	W[j,i] = 1.0
    f.close()
    
    return W

def dibujarGrafo(W, print_ejes=True):
    
    options = {
    'node_color': 'yellow',
    'node_size': 200,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    'with_labels' : True}
    
    N = W.shape[0]
    G = nx.DiGraph(W.T)
    
    #renombro nodos de 1 a N
    G = nx.relabel_nodes(G, {i:i+1 for i in range(N)})
    if print_ejes:
        print('Ejes: ', [e for e in G.edges])
    
    nx.draw(G, pos=nx.spring_layout(G), **options)

# Calcula el ranking (rnk) y los scores (scr) utilizando el algoritmo PageRank
def calcularRanking(M, p):
    npages = M.shape[0]
    rnk = np.arange(0, npages) # ind{k] = i, la pagina k tienen el iesimo orden en la lista.
    scr = np.zeros(npages) # scr[k] = alpha, la pagina k tiene un score de alpha 
    # COIDGO

    # Función para calcular el grado de una página
    def grado(W, pag):
        cj = 0  # Inicializar la suma en 0
        
        for i in range(len(W)): # Sumar el valor de la columna "pag" en cada fila
            
            cj += W[i-1][pag-1] # Restar 1 a los indices ya que python utiliza el 0 para la pagina 1, el 1 para la pagina 2 y asi sucesivamente
            
        return cj

    # Función para construir la matriz diagonal D
    def construir_matriz_D(W):
        n = len(W)
        D = [[0] * n for _ in range(n)] # Inicializar D con la misma dimension que W
        
        for j in range(len(W)):
            
            if grado(W, j) != 0: # Descartar casos en que el grado es igual a 0 para no dividir por 0
                D[j-1][j-1] = 1 / grado(W, j) # Construir la matriz diagonal D con los coeficientes 1/Cj
        
        return D

    # Función para crear una matriz identidad de tamaño n x n
    def matriz_identidad(n):
        identidad = []
        for i in range(n):
            fila = []
            for j in range(n):
                if i == j:
                    fila.append(1)
                else:
                    fila.append(0)
            identidad.append(fila)
        return identidad

    # Función para crear un vector columna de unos de tamaño n
    def construir_e(n):
        return np.ones(n)
   
    # Función para la descomposición LU de una matriz A
    def descomposicion_LU(A):
        n = len(A)
        L = np.eye(n)  # Matriz identidad inicial para L
        U = np.zeros((n, n))  # Inicializar matriz U con ceros

        for k in range(n):
        
            for j in range(k, n): # Actualizar elementos de U
                U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])

        
            for i in range(k+1, n): # Encontrar los elementos de L
                L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

        return L, U
   
    # Resuelve el sistema (I - pWD) x = e (suponiendo gamma = 1)
    def resolucion_sistema_lineal(p, W):
        n = len(W)
        
        D = construir_matriz_D(W) # Matriz diagonal D
        
        e = construir_e(n) # Vector columna e
        
        I = matriz_identidad(n) # Matriz identidad
        
        WD = np.dot(W,D) # Producto de matrices WD
        
        A = I - (p*WD) # Defino A = I - (pWD)
        
        L, U = descomposicion_LU(A) # Descomposicion LU de A
        
        y = solve_triangular(L, e, lower=True) # Resolver el sistema L y = e
        
        x = solve_triangular(U, y) # Resolver el sistema U x = y
        
        return x

     # Normaliza un vector de tal manera que la suma de sus componentes sea igual a 1
    def normalizar(v):
        t = 0
        
        for i in range(len(v)): # Sumar todas las componentes de v
            t += v[i]
            
        for i in range(len(v)): # Normalizar el vector para que la sumatoria de sus componentes den 1 
            v[i] /= t 
        
        return v
            
    
    # Ordena el ranking de manera descendente
    def ranking(scr):
    
        sorted_indices = np.argsort(-np.array(scr))  # Ordenar los puntajes en orden descendente y obtener los índices  
        
        rnk = sorted_indices + 1  # Añadir 1 a cada índice para que los rankings comiencen desde 1
    
        return rnk
    
    
    # Calcula los scores sin normalizar
    scr = resolucion_sistema_lineal(p, M)
    
    # Calcula el ranking en base a los scores
    rnk = ranking(scr) # Asignar a la variable rnk el ranking de forma ascendente sobre los puntajes de las páginas
    
    # Devuelve el ranking y los scores normalizados
    return rnk, normalizar(scr)

# Obtiene el máximo score de un grafo de conectividad M con un valor de p
def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)
    
    return output


