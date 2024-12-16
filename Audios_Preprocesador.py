import os
import librosa
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf

class PreprocesadorAudios:
    def __init__(self, ruta_db="DB_Audios", carpeta_crudos="Crudos", carpeta_procesados="Processed", umbral_silencio=25, sr=48000, frec_corte_baja=250, frec_corte_alta=5000):
        """
        Inicializa el preprocesador con las configuraciones necesarias.

        :param ruta_db: Ruta principal de la base de datos.
        :param carpeta_crudos: Carpeta que contiene los audios crudos.
        :param carpeta_procesados: Carpeta donde se guardarán los audios procesados.
        :param umbral_silencio: Umbral en dB para eliminar silencios.
        :param sr: Frecuencia de muestreo deseada.
        :param frec_corte_baja: Frecuencia de corte baja para el filtro pasabanda.
        :param frec_corte_alta: Frecuencia de corte alta para el filtro pasabanda.
        """
        self.ruta_db = ruta_db
        self.carpeta_crudos = os.path.join(ruta_db, carpeta_crudos)
        self.carpeta_procesados = os.path.join(ruta_db, carpeta_procesados)
        self.umbral_silencio = umbral_silencio
        self.sr = sr
        self.frec_corte_baja = frec_corte_baja
        self.frec_corte_alta = frec_corte_alta

        # Crear carpeta de procesados si no existe
        os.makedirs(self.carpeta_procesados, exist_ok=True)

    def eliminar_silencios(self, audio):
        """
        Elimina los silencios del audio según el umbral configurado.

        :param audio: Señal de audio a procesar.
        :return: Audio sin silencios.
        """
        intervalos = librosa.effects.split(audio, top_db=self.umbral_silencio)
        return np.concatenate([audio[inicio:fin] for inicio, fin in intervalos])

    def filtro_pasabanda(self, audio):
        """
        Aplica un filtro pasabanda a la señal de audio.

        :param audio: Señal de audio a procesar.
        :return: Señal filtrada.
        """
        nyquist = 0.5 * self.sr
        baja = self.frec_corte_baja / nyquist
        alta = self.frec_corte_alta / nyquist
        b, a = butter(1, [baja, alta], btype='band')
        return lfilter(b, a, audio)

    def procesar_audio(self, ruta_audio):
        """
        Procesa un audio completo: elimina silencios, normaliza y aplica filtro pasabanda.

        :param ruta_audio: Ruta del archivo de audio.
        :return: Señal de audio procesada y su frecuencia de muestreo.
        """
        try:
            # Cargar el audio
            audio, sr_original = librosa.load(ruta_audio, sr=self.sr)

            # Eliminar silencios
            audio_sin_silencio = self.eliminar_silencios(audio)

            # Normalizar amplitudes
            audio_normalizado = librosa.util.normalize(audio_sin_silencio)

            # Aplicar filtro pasabanda
            audio_filtrado = self.filtro_pasabanda(audio_normalizado)

            return audio_filtrado
        except Exception as e:
            print(f"Error al procesar el audio {ruta_audio}: {e}")
            return None

    def procesar_base_datos(self):
        """
        Procesa todos los audios en la carpeta de "Crudos" y guarda los resultados en "Processed".
        """
        for archivo in os.listdir(self.carpeta_crudos):
            if archivo.endswith(".wav"):
                ruta_entrada = os.path.join(self.carpeta_crudos, archivo)
                ruta_salida = os.path.join(self.carpeta_procesados, archivo)

                print(f"Procesando {archivo}...")
                audio_procesado = self.procesar_audio(ruta_entrada)

                if audio_procesado is not None:
                    # Guardar audio procesado
                    sf.write(ruta_salida, audio_procesado, self.sr)
                    print(f"Guardado: {ruta_salida}")
                else:
                    print(f"No se pudo procesar {archivo}.")

if __name__ == "__main__":
    preprocesador = PreprocesadorAudios()
    preprocesador.procesar_base_datos()