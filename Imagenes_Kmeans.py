import cv2
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

class KMeans:
    def __init__(self, ruta_rgb, ruta_hu, ruta_db_imagenes="DB_Imagenes", k=4, max_iter=5):
        self.ruta_rgb = ruta_rgb
        self.ruta_hu = ruta_hu
        self.ruta_db_imagenes = ruta_db_imagenes
        self.k = k
        self.max_iter = max_iter
        self.datos = None
        self.centroides = None
        self.ruta_centroides = os.path.join(self.ruta_db_imagenes, "centroides.csv")
        self.ruta_clusters = os.path.join(self.ruta_db_imagenes, "clusters.csv")
        self.ruta_nombres = os.path.join(self.ruta_db_imagenes, "nombres_grupos.csv")
        self.nombres_grupos = [f"Grupo {i+1}" for i in range(k)]  # Inicialmente genéricos

    def cargar_datos(self):
        """
        Carga los datos de valores RGB promedio y momentos de Hu escalados desde los archivos CSV.
        """
        try:
            # Leer archivos CSV
            df_rgb = pd.read_csv(self.ruta_rgb)
            df_hu = pd.read_csv(self.ruta_hu)

            # Verificar que los archivos contengan datos
            if df_rgb.empty or df_hu.empty:
                print("Error: Los archivos CSV están vacíos.")
                self.datos = None
                return

            # Extraer columnas necesarias
            valores_rgb = df_rgb[['Promedio R', 'Promedio G', 'Promedio B']].to_numpy()
            momentos_hu = df_hu.iloc[:, 1:].to_numpy()  # Excluir la columna de nombres

            # Preparar dataset utilizando Hu_1, Promedio G y Promedio B
            self.datos = np.array([
                momentos_hu[:, 0],  # Hu_1
                valores_rgb[:, 1] / 255,  # Promedio G normalizado
                valores_rgb[:, 2] / 255   # Promedio B normalizado
            ]).T

            print("Datos cargados correctamente.")
        except FileNotFoundError as e:
            print(f"Error: No se encontró uno de los archivos necesarios. {e}")
            self.datos = None
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            self.datos = None
            
    @staticmethod
    def calcular_distancia(punto, centroides):
        return np.linalg.norm(punto - centroides, axis=1)

    def inicializar_centroides(self):
        indices = random.sample(range(len(self.datos)), self.k)
        self.centroides = self.datos[indices]

    def asignar_clusters(self):
        clusters = []
        for punto in self.datos:
            distancias = self.calcular_distancia(punto, self.centroides)
            clusters.append(np.argmin(distancias))
        return np.array(clusters)

    def actualizar_centroides(self, clusters):
        nuevos_centroides = []
        for i in range(self.k):
            puntos_cluster = self.datos[clusters == i]
            if len(puntos_cluster) > 0:
                nuevos_centroides.append(np.mean(puntos_cluster, axis=0))
            else:
                nuevos_centroides.append(self.datos[random.randint(0, len(self.datos) - 1)])
        self.centroides = np.array(nuevos_centroides)

    def verificar_balance(self, clusters):
        tamanos = [np.sum(clusters == i) for i in range(self.k)]
        print(f"Tamaños de los grupos: {tamanos}")
        return len(set(tamanos)) == 1  # Verifica que todos los tamaños sean iguales


    def graficar_kmeans(self, clusters, iteracion, usar_nombres_genericos=True):

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        colores_clusters = ['r', 'g', 'b', 'y']
        nombres_leyenda = [f"Grupo {i+1}" for i in range(len(self.centroides))] if usar_nombres_genericos else self.nombres_grupos

        for i in range(len(self.centroides)):
            puntos = self.datos[clusters == i]
            ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], c=colores_clusters[i], label=nombres_leyenda[i])

        ax.scatter(self.centroides[:, 0], self.centroides[:, 1], self.centroides[:, 2], c='k', s=100, marker='X', label='Centroides')

        ax.set_xlabel('Hu_1')
        ax.set_ylabel('Promedio G')
        ax.set_zlabel('Promedio B')
        ax.set_title(f'Iteración {iteracion}')
        ax.legend()
        plt.show()

    def ejecutar_kmeans(self, ruta_centroides, ruta_clusters, ruta_nombres):
        for intento in range(10):  # Intentar hasta 10 veces
            self.inicializar_centroides()
            for iteracion in range(self.max_iter):
                clusters = self.asignar_clusters()
                # Usar nombres genéricos durante las iteraciones
                self.graficar_kmeans(clusters, iteracion + 1, usar_nombres_genericos=True)
                self.actualizar_centroides(clusters)

            if self.verificar_balance(clusters):
                print("Clasificación balanceada completada con éxito.")
                self.guardar_clasificacion(ruta_centroides, ruta_clusters)
                
                # Solicitar nombres a los grupos si es balanceado
                nombres_grupos = self.solicitar_nombres_grupos(ruta_nombres)
                self.nombres_grupos = nombres_grupos
                return clusters

            print(f"Intento {intento + 1}: Clasificación no balanceada. Reiniciando...")

        print("No se logró una clasificación balanceada.")
        return None



    def asignar_nombres_grupos(self):
        print("Asignación de nombres a los grupos:")
        for i in range(self.k):
            nombre = input(f"Ingrese el nombre para el Grupo {i+1} (actual: {self.nombres_grupos[i]}): ")
            if nombre:
                self.nombres_grupos[i] = nombre

    def clasificar_imagen_candidata(self, carpeta_candidato):
        if self.centroides is None:
            print("Error: Los centroides no están inicializados. Ejecute primero la clasificación completa (opción 3).")
            return

        if self.datos is None:
            print("Error: Los datos no han sido cargados. Asegúrese de cargar los datos antes de clasificar.")
            return

        ruta_rgb_candidato = os.path.join(carpeta_candidato, 'valores_rgb_candidata.csv')
        ruta_hu_candidato = os.path.join(carpeta_candidato, 'momentos_hu_candidata.csv')
        ruta_nombres = os.path.join(os.getcwd(), "DB_Imagenes", "nombres_grupos.csv")

        # Cargar los nombres de los grupos
        self.cargar_nombres_grupos(ruta_nombres)

        if not os.path.exists(ruta_rgb_candidato) or not os.path.exists(ruta_hu_candidato):
            print("No se encontraron los archivos CSV del candidato en la carpeta especificada.")
            return

        # Cargar los datos del candidato
        df_rgb_candidato = pd.read_csv(ruta_rgb_candidato)
        df_hu_candidato = pd.read_csv(ruta_hu_candidato)

        # Extraer valores de Hu_1, Promedio G y Promedio B
        hu1 = df_hu_candidato.iloc[0, 1]  # Primera fila, columna Hu_1
        g = df_rgb_candidato.iloc[0, 2]  # Primera fila, columna Promedio G
        b = df_rgb_candidato.iloc[0, 3]  # Primera fila, columna Promedio B

        punto_candidato = np.array([hu1, g / 255, b / 255])
        distancias = self.calcular_distancia(punto_candidato, self.centroides)
        grupo = np.argmin(distancias)

        print(f"\n\n\nLa imagen candidata pertenece a: {self.nombres_grupos[grupo]}.")

        # Graficar el espacio 3D con el candidato
        clusters = self.asignar_clusters()
        self.graficar_candidato(punto_candidato, clusters)

    def guardar_clasificacion(self, ruta_centroides, ruta_clusters):
        # Guardar los centroides
        pd.DataFrame(self.centroides, columns=['Hu_1', 'Promedio G', 'Promedio B']).to_csv(ruta_centroides, index=False)
        print(f"Centroides guardados en {ruta_centroides}")

        # Guardar la clasificación (grupo asignado a cada punto)
        clusters = self.asignar_clusters()
        df_clusters = pd.DataFrame({
            'Grupo': clusters,
            'Hu_1': self.datos[:, 0],
            'Promedio G': self.datos[:, 1],
            'Promedio B': self.datos[:, 2]
        })
        df_clusters.to_csv(ruta_clusters, index=False)
        print(f"Clasificación guardada en {ruta_clusters}")

    def cargar_clasificacion(self):
        # Cargar los centroides
        if os.path.exists(self.ruta_centroides):
            self.centroides = pd.read_csv(self.ruta_centroides).to_numpy()
            print(f"Centroides cargados desde {self.ruta_centroides}")
        else:
            print(f"No se encontró el archivo de centroides en {self.ruta_centroides}")

        # Cargar la clasificación (opcional)
        if os.path.exists(self.ruta_clusters):
            df_clusters = pd.read_csv(self.ruta_clusters)
            print(f"Clasificación cargada desde {self.ruta_clusters}")
        else:
            print(f"No se encontró el archivo de clusters en {self.ruta_clusters}")

        # Cargar los nombres de los grupos
        self.cargar_nombres_grupos()

    def solicitar_nombres_grupos(self, ruta_nombres):
        nombres_grupos = []
        print("Asignación de nombres a los grupos:")
        for i in range(self.k):
            nombre = input(f"Ingrese el nombre para el Grupo {i+1}: ")
            nombres_grupos.append(nombre)

        # Guardar nombres en archivo CSV
        with open(ruta_nombres, mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow(['Grupo', 'Nombre'])
            for i, nombre in enumerate(nombres_grupos):
                escritor_csv.writerow([i, nombre])

        print(f"Nombres de grupos guardados en {ruta_nombres}")
        return nombres_grupos
    
    def cargar_nombres_grupos(self):
        if os.path.exists(self.ruta_nombres):
            with open(self.ruta_nombres, mode='r') as archivo_csv:
                lector_csv = csv.reader(archivo_csv)
                next(lector_csv)  # Saltar la cabecera
                self.nombres_grupos = [fila[1] for fila in lector_csv]
            print(f"Nombres de grupos cargados desde {self.ruta_nombres}")
        else:
            print(f"No se encontró el archivo de nombres de grupos en {self.ruta_nombres}")

    def graficar_candidato(self, punto_candidato, clusters):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        colores_clusters = ['r', 'g', 'b', 'y']

        # Graficar los puntos de cada cluster
        for i in range(len(self.centroides)):
            puntos = self.datos[clusters == i]
            ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], c=colores_clusters[i], label=self.nombres_grupos[i])

        # Graficar el punto candidato
        ax.scatter(punto_candidato[0], punto_candidato[1], punto_candidato[2], c='k', s=200, marker='*', label='Candidato')

        # Graficar los centroides
        ax.scatter(self.centroides[:, 0], self.centroides[:, 1], self.centroides[:, 2], c='k', s=100, marker='X', label='Centroides')

        ax.set_xlabel('Hu_1')
        ax.set_ylabel('Promedio G')
        ax.set_zlabel('Promedio B')
        ax.set_title('Clasificación del Candidato')
        ax.legend()
        plt.show()

