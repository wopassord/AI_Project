import cv2
import numpy as np
import os
import csv

class Parametrizador:
    def __init__(self, ruta_masks, ruta_processed, ruta_salida):
        self.ruta_masks = ruta_masks
        self.ruta_processed = ruta_processed
        self.ruta_salida = ruta_salida
        os.makedirs(self.ruta_salida, exist_ok=True)

    @staticmethod
    def escalar_momentos_hu(momentos_hu):
        return [-np.sign(m) * np.log10(abs(m) + 1e-10) if m != 0 else 0 for m in momentos_hu]

    def calcular_rgb_promedio(self, ruta_csv_salida):
        resultados_rgb = []

        for nombre_archivo in os.listdir(self.ruta_processed):
            ruta_imagen = os.path.join(self.ruta_processed, nombre_archivo)
            if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Saltando archivo no válido: {nombre_archivo}")
                continue

            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f"No se pudo cargar la imagen: {nombre_archivo}")
                continue

            promedio_b = np.mean(imagen[:, :, 0])
            promedio_g = np.mean(imagen[:, :, 1])
            promedio_r = np.mean(imagen[:, :, 2])

            max_valor = max(promedio_r, promedio_g, promedio_b)
            promedio_r = (promedio_r / max_valor) * 255 if max_valor > 0 else 0
            promedio_g = (promedio_g / max_valor) * 255 if max_valor > 0 else 0
            promedio_b = (promedio_b / max_valor) * 255 if max_valor > 0 else 0

            resultados_rgb.append([nombre_archivo, promedio_r, promedio_g, promedio_b])

        # Guardar los resultados en el CSV
        with open(ruta_csv_salida, mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow(['Archivo', 'Promedio R', 'Promedio G', 'Promedio B'])
            escritor_csv.writerows(resultados_rgb)

        print(f"Valores RGB promedio guardados en {ruta_csv_salida}")

    def calcular_momentos_hu(self, ruta_csv_salida):
        resultados_hu = []

        for nombre_archivo in os.listdir(self.ruta_masks):
            ruta_imagen = os.path.join(self.ruta_masks, nombre_archivo)
            if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Saltando archivo no válido: {nombre_archivo}")
                continue

            mascara = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            if mascara is None:
                print(f"No se pudo cargar la máscara: {nombre_archivo}")
                continue

            momentos = cv2.moments(mascara)
            momentos_hu = cv2.HuMoments(momentos).flatten()
            momentos_hu_escalados = self.escalar_momentos_hu(momentos_hu)

            resultados_hu.append([nombre_archivo] + momentos_hu_escalados)

        # Guardar los resultados en el CSV
        with open(ruta_csv_salida, mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            encabezado = ['Archivo'] + [f'Hu_{i+1}' for i in range(7)]
            escritor_csv.writerow(encabezado)
            escritor_csv.writerows(resultados_hu)

        print(f"Momentos de Hu escalados guardados en {ruta_csv_salida}")

    def procesar_parametrizacion(self):
        ruta_csv_rgb = os.path.join(self.ruta_salida, 'valores_rgb_promedio.csv')
        ruta_csv_hu = os.path.join(self.ruta_salida, 'momentos_hu_escalados.csv')
        self.calcular_rgb_promedio(ruta_csv_rgb)
        self.calcular_momentos_hu(ruta_csv_hu)

    def procesar_candidata(self, ruta_imagen_candidata):
        ruta_candidato = os.path.join(os.getcwd(), "Candidato")
        ruta_mascara = os.path.join(ruta_candidato, "Mask", "mask_candidata.png")
        ruta_procesada = os.path.join(ruta_candidato, "Processed", "processed_candidata.png")
        ruta_csv_rgb = os.path.join(ruta_candidato, 'valores_rgb_candidata.csv')
        ruta_csv_hu = os.path.join(ruta_candidato, 'momentos_hu_candidata.csv')

        if not os.path.exists(ruta_mascara) or not os.path.exists(ruta_procesada):
            print("No se encontraron máscara o imagen procesada para la candidata.")
            return

        self.calcular_rgb_promedio_imagen_candidata(ruta_procesada, ruta_csv_rgb)
        self.calcular_momentos_hu_imagen_candidata(ruta_mascara, ruta_csv_hu)


    def calcular_rgb_promedio_imagen_candidata(self, ruta_imagen_procesada, ruta_csv_salida):
        imagen = cv2.imread(ruta_imagen_procesada)
        if imagen is None:
            print(f"No se pudo cargar la imagen procesada: {ruta_imagen_procesada}")
            return

        promedio_b = np.mean(imagen[:, :, 0])
        promedio_g = np.mean(imagen[:, :, 1])
        promedio_r = np.mean(imagen[:, :, 2])

        max_valor = max(promedio_r, promedio_g, promedio_b)
        promedio_r = (promedio_r / max_valor) * 255 if max_valor > 0 else 0
        promedio_g = (promedio_g / max_valor) * 255 if max_valor > 0 else 0
        promedio_b = (promedio_b / max_valor) * 255 if max_valor > 0 else 0

        with open(ruta_csv_salida, mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow(['Archivo', 'Promedio R', 'Promedio G', 'Promedio B'])
            escritor_csv.writerow([os.path.basename(ruta_imagen_procesada), promedio_r, promedio_g, promedio_b])

        print(f"Valores RGB promedio guardados en {ruta_csv_salida}")

    def calcular_momentos_hu_imagen_candidata(self, ruta_mascara, ruta_csv_salida):
        mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)
        if mascara is None:
            print(f"No se pudo cargar la máscara: {ruta_mascara}")
            return

        momentos = cv2.moments(mascara)
        momentos_hu = cv2.HuMoments(momentos).flatten()
        momentos_hu_escalados = self.escalar_momentos_hu(momentos_hu)

        with open(ruta_csv_salida, mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            encabezado = ['Archivo'] + [f'Hu_{i+1}' for i in range(7)]
            escritor_csv.writerow(encabezado)
            escritor_csv.writerow([os.path.basename(ruta_mascara)] + momentos_hu_escalados)

        print(f"Momentos de Hu escalados guardados en {ruta_csv_salida}")
