import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.preprocessing import MinMaxScaler

class ClasificadorAudios:
    def __init__(self, archivo_db="DB_Audios/parametros_DB.csv", archivo_candidato="DB_Audios/parametros_candidato.csv", n_componentes=3, k=4):
        self.archivo_db = archivo_db
        self.archivo_candidato = archivo_candidato
        self.n_componentes = n_componentes
        self.k = k

    def cargar_parametros(self, archivo, es_base_datos=False):
        """
        Carga parámetros y etiquetas desde un archivo CSV.
        :param archivo: Ruta del archivo CSV.
        :param es_base_datos: Indica si el archivo pertenece a la base de datos (True) o al audio candidato (False).
        :return: Tupla (X, y). En el caso del audio candidato, y será None.
        """
        X, y = [], []
        with open(archivo, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Saltar encabezado
            for row in reader:
                if es_base_datos:
                    # Última columna es la etiqueta
                    X.append([float(value) for value in row[:-1]])
                    y.append(row[-1])
                else:
                    # Todo el contenido es de características (sin etiqueta)
                    X.append([float(value) for value in row])
        return np.array(X), np.array(y) if es_base_datos else None

    def clasificar_candidato(self):
        """
        Clasifica el audio candidato y genera una visualización en 3D.
        """
        # Cargar parámetros de la base de datos (incluye etiquetas)
        X_db, y_db = self.cargar_parametros(self.archivo_db, es_base_datos=True)
        
        # Cargar parámetros del audio candidato (sin etiquetas)
        X_candidato, _ = self.cargar_parametros(self.archivo_candidato, es_base_datos=False)

        # Validar dimensiones
        if X_candidato.shape[1] != X_db.shape[1]:
            print(f"Error: Las dimensiones del audio candidato ({X_candidato.shape[1]}) no coinciden con la base de datos ({X_db.shape[1]}).")
            return

        # Calcular PCA considerando todos los parámetros
        pca = PCA(n_components=self.n_componentes)
        X_db_pca = pca.fit_transform(X_db)
        X_candidato_pca = pca.transform(X_candidato)

        # Clasificar con KNN utilizando los PCA
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(X_db_pca, y_db)
        etiqueta_candidato = knn.predict(X_candidato_pca)

        # Llamar a la visualización en 3D
        self.visualizar_3D(X_db_pca, y_db, X_candidato_pca, etiqueta_candidato[0])

        # Devolver la etiqueta clasificada
        return etiqueta_candidato[0]
        
    def visualizar_3D(self, X_db_pca, y_db, X_candidato_pca, etiqueta_candidato):
        """
        Genera un gráfico en 3D con los puntos escalados, conectando el candidato con sus k vecinos más cercanos.
        """

        # Escalar los datos entre -1 y 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_db_pca_scaled = scaler.fit_transform(X_db_pca)
        X_candidato_pca_scaled = scaler.transform(X_candidato_pca)

        # Obtener los k vecinos más cercanos
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(X_db_pca_scaled, y_db)
        distancias, indices_vecinos = knn.kneighbors(X_candidato_pca_scaled)

        # Colores para las clases
        color_map = {
            "zanahoria": "orange",
            "camote": "violet",
            "berenjena": "darkviolet",
            "papa": "saddlebrown"
        }
        saturados = {k: to_rgba(v, alpha=1.0) for k, v in color_map.items()}

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar los puntos de la base de datos por etiqueta
        for etiqueta, color in color_map.items():
            indices = (y_db == etiqueta)
            ax.scatter(
                X_db_pca_scaled[indices, 0], 
                X_db_pca_scaled[indices, 1], 
                X_db_pca_scaled[indices, 2], 
                color=color, label=etiqueta, s=30, alpha=0.5
            )

        # Resaltar los vecinos más cercanos del candidato
        for idx in indices_vecinos[0]:
            ax.scatter(
                X_db_pca_scaled[idx, 0], 
                X_db_pca_scaled[idx, 1], 
                X_db_pca_scaled[idx, 2], 
                color=saturados[y_db[idx]], s=50, alpha=0.8
            )
            ax.plot(
                [X_candidato_pca_scaled[0, 0], X_db_pca_scaled[idx, 0]],
                [X_candidato_pca_scaled[0, 1], X_db_pca_scaled[idx, 1]],
                [X_candidato_pca_scaled[0, 2], X_db_pca_scaled[idx, 2]],
                'k--', alpha=0.6
            )

        # Graficar el candidato
        ax.scatter(
            X_candidato_pca_scaled[0, 0], 
            X_candidato_pca_scaled[0, 1], 
            X_candidato_pca_scaled[0, 2], 
            color=saturados[etiqueta_candidato], s=100, label="Candidato", alpha=1.0
        )

        # Configuración del gráfico
        ax.set_title("Clasificación de audio candidato")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
        plt.show()

if __name__ == "__main__":
    clasificador = ClasificadorAudios()

    # Clasificar el audio candidato
    etiqueta_predicha = clasificador.clasificar_candidato()
    print(f"El audio candidato fue clasificado como: {etiqueta_predicha}")
