import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Variables para almacenar los resultados del entrenamiento
norma_error_epocas = []
evolucion_pesos = []
pesos_iniciales = None
pesos_finales = None
num_epocas = 0
error_permisible = 0

def entrenar_perceptron(eta, epocas, ruta_del_archivo, progreso_barra):
    global norma_error_epocas, evolucion_pesos, pesos_iniciales, pesos_finales

    # Limpiando las listas en caso de entrenamientos previos
    norma_error_epocas.clear()
    evolucion_pesos.clear()

    delimitador = ';'  # Asumiendo que el archivo usa ';' como delimitador
    nombres_columnas = ['X1', 'X2', 'X3', 'Y']
    
    data_frame = pd.read_csv(ruta_del_archivo, delimiter=delimitador, header=None, names=nombres_columnas)
    num_caracteristicas = len(data_frame.columns) - 1

    pesos = np.random.uniform(low=0, high=1, size=(num_caracteristicas + 1, 1)).round(2)
    columnas_x = np.hstack([data_frame.iloc[:, :-1].values, np.ones((data_frame.shape[0], 1))])
    columna_y = np.array(data_frame.iloc[:, -1])

    pesos_iniciales = pesos.copy()
    num_epocas = epocas

    print(f"estos son los pesos iniciales {pesos}")

    for i in range(num_caracteristicas + 1):
        evolucion_pesos.append([])

    for epoch in range(epocas):
        u = np.dot(columnas_x, pesos)
        ycalculada = np.where(u >= 0, 1, 0).reshape(-1, 1)
        errores = columna_y.reshape(-1, 1) - ycalculada

        norma_error = np.linalg.norm(errores)
        norma_error_epocas.append(norma_error)

        for i in range(num_caracteristicas + 1):
            evolucion_pesos[i].append(pesos[i, 0])

        producto_errores = np.dot(columnas_x.T, errores)
        deltaW = eta * producto_errores
        pesos += np.round(deltaW, 4)

        # Actualiza la barra de progreso
        progreso_barra['value'] = (epoch + 1) / epocas * 100
        progreso_barra.update()
    pesos_finales = pesos 

def mostrar_resultados():
    global norma_error_epocas, evolucion_pesos
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(norma_error_epocas) + 1), norma_error_epocas)
    plt.title('Evolución de la norma del error (|e|)')
    plt.xlabel('Época')
    plt.ylabel('Norma del Error')

    plt.subplot(1, 2, 2)
    for i, pesos_epoca in enumerate(evolucion_pesos):
        plt.plot(range(1, len(pesos_epoca) + 1), pesos_epoca, label=f'Peso {i + 1}')
    plt.title('Evolución del valor de los pesos (W)')
    plt.xlabel('Época')
    plt.ylabel('Valor del Peso')
    plt.legend()

    plt.tight_layout()
    plt.show()

def obtener_pesos():
    return pesos_iniciales, pesos_finales, num_epocas, error_permisible
