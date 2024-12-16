import os
from Audios_Grabadora import GrabadoraAudios
from Audios_Preprocesador import PreprocesadorAudios
from Audios_Parametrizador import ParametrizadorAudios
from Audios_Clasificador import ClasificadorAudios
from Imagenes_Procesador import ProcesadorImagenes
from Imagenes_Parametrizador import Parametrizador
from Imagenes_Kmeans import KMeans
import matplotlib
matplotlib.use('TkAgg')  # Fuerza un backend interactivo que abre ventanas

class ClasificacionVerduras:
    def __init__(self):
        # Directorios base
        self.ruta_base = os.getcwd()
        self.ruta_cinta = os.path.join(self.ruta_base, "Cinta")
        self.ruta_masks = os.path.join(self.ruta_cinta, "Masks")
        self.ruta_processed = os.path.join(self.ruta_cinta, "Processed")
        self.ruta_csv_rgb = os.path.join(self.ruta_cinta, "valores_rgb_promedio.csv")
        self.ruta_csv_hu = os.path.join(self.ruta_cinta, "momentos_hu_escalados.csv")
        
        # Rutas relacionadas al audio candidato
        self.ruta_candidato_audio = os.path.join(self.ruta_base, "Candidato")
        self.ruta_audio_crudo = os.path.join(self.ruta_candidato_audio, "audio_candidato.wav")
        self.ruta_audio_procesado = os.path.join(self.ruta_candidato_audio, "processed_audio_candidato.wav")
        self.ruta_audio_csv = os.path.join(self.ruta_candidato_audio, "parametros_candidato.csv")
        
        # Crear carpetas si no existen
        os.makedirs(self.ruta_candidato_audio, exist_ok=True)
        os.makedirs(self.ruta_masks, exist_ok=True)
        os.makedirs(self.ruta_processed, exist_ok=True)
        
        self.ruta_centroides = os.path.join(self.ruta_cinta, "centroides.csv")
        self.ruta_clusters = os.path.join(self.ruta_cinta, "clusters.csv")
        self.ruta_nombres = os.path.join(self.ruta_cinta, "nombres_grupos.csv")

        # Instancias de clases
        self.grabadora = GrabadoraAudios(
            ruta_candidato=self.ruta_candidato_audio,
            archivo_candidato=self.ruta_audio_crudo,
            archivo_procesado=self.ruta_audio_procesado
        )
        self.preprocesador_audio = PreprocesadorAudios()
        self.parametrizador_audio = ParametrizadorAudios()
        self.procesador_imagenes = ProcesadorImagenes(
            self.ruta_cinta,
            self.ruta_masks,
            self.ruta_processed
        )
        self.parametrizador_imagenes = Parametrizador(
            self.ruta_masks,
            self.ruta_processed,
            self.ruta_cinta
        )
        self.kmeans = KMeans(self.ruta_csv_rgb, self.ruta_csv_hu)

    def menu_principal(self):
        while True:
            print("\n===== MENÚ PRINCIPAL =====")
            print("1. Iniciar clasificación de verduras")
            print("2. Salir")
            opcion = input("Selecciona una opción: ")

            if opcion == "1":
                self.iniciar_clasificacion()
            elif opcion == "2":
                print("Saliendo del programa. ¡Hasta luego!")
                break
            else:
                print("Opción no válida. Intenta nuevamente.")

    def iniciar_clasificacion(self):
        print("\nGrabando y procesando el audio candidato...")
        self.grabadora.grabar_audio(duracion=3)
        if self.grabadora.procesar_audio_candidato():
            if not self.grabadora.extraer_parametros_candidato():
                print("Error al extraer parámetros del audio candidato.")
                return

        # Clasificar audio candidato para obtener la verdura
        print("\nClasificando audio candidato...")
        clasificador_audio = ClasificadorAudios()
        etiqueta_audio = clasificador_audio.clasificar_candidato().strip().lower()
        print(f"El usuario mencionó: {etiqueta_audio}")

        # Procesar imágenes en la carpeta "Cinta"
        print("\nProcesando imágenes en la cinta...")
        self.procesador_imagenes.procesar_carpeta()

        # Validar carpetas
        if not os.listdir(self.ruta_masks):
            print("Error: No se encontraron máscaras en la carpeta Masks.")
            return
        if not os.listdir(self.ruta_processed):
            print("Error: No se encontraron imágenes enmascaradas en la carpeta Processed.")
            return

        # Parametrizar imágenes procesadas
        print("\nParametrizando imágenes de la cinta...")
        self.parametrizador_imagenes.procesar_parametrizacion()

        # Cargar datos y clasificación preexistente
        print("\nCargando datos y clasificación preexistente de imágenes...")
        self.kmeans.cargar_datos()
        self.kmeans.cargar_clasificacion()

        # Verificar que los datos y centroides estén cargados
        if self.kmeans.datos is None or self.kmeans.centroides is None:
            print("Error: No se pueden clasificar las imágenes. Datos o centroides faltantes.")
            return

        # Clasificar imágenes en la cinta
        clusters = self.kmeans.asignar_clusters()
        print("Clasificación completada.")

        # Buscar y mostrar la imagen original correspondiente
        print("\nBuscando la imagen clasificada que coincide con el audio...")
        archivos_imagenes_originales = sorted(
            [img for img in os.listdir(self.ruta_cinta) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        )
        print(f"Imágenes originales: {archivos_imagenes_originales}")
        print(f"Nombres de grupos: {self.kmeans.nombres_grupos}")
        print(f"Clusters asignados: {clusters}")

        for idx, cluster in enumerate(clusters):
            nombre_grupo = self.kmeans.nombres_grupos[cluster].strip().lower()
            if nombre_grupo == etiqueta_audio:
                imagen_original = archivos_imagenes_originales[idx]
                ruta_imagen_original = os.path.join(self.ruta_cinta, imagen_original)
                print(f"Imagen original clasificada: {ruta_imagen_original}")
                self.mostrar_imagen(ruta_imagen_original)
                return

        print("No se encontró ninguna imagen que coincida con la clasificación del audio.")

    @staticmethod
    def mostrar_imagen(ruta_imagen):
        import cv2
        import os

        # Leer la imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"No se pudo cargar la imagen desde: {ruta_imagen}")
            return

        # Guardar la imagen clasificada en una carpeta específica
        ruta_guardado = os.path.join(os.getcwd(), "Imagenes_Clasificadas")
        os.makedirs(ruta_guardado, exist_ok=True)
        nombre_archivo = os.path.basename(ruta_imagen)
        ruta_imagen_guardada = os.path.join(ruta_guardado, f"clasificada_{nombre_archivo}")
        cv2.imwrite(ruta_imagen_guardada, imagen)
        print(f"Imagen clasificada guardada en: {ruta_imagen_guardada}")

        # Redimensionar la imagen para mostrarla sin zoom excesivo
        altura_original, ancho_original = imagen.shape[:2]
        ancho_nuevo = 600  # Ancho deseado para la ventana
        altura_nueva = 600

        # Redimensionar solo si la imagen es más grande que el tamaño deseado
        if ancho_original > ancho_nuevo:
            imagen = cv2.resize(imagen, (ancho_nuevo, altura_nueva), interpolation=cv2.INTER_AREA)

        # Mostrar la imagen en una ventana
        print("Mostrando imagen clasificada...")
        cv2.imshow("Imagen Clasificada", imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    clasificador_verduras = ClasificacionVerduras()
    clasificador_verduras.menu_principal()
