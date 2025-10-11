"""
audio.py
--------
Este módulo se encarga de la generación y reproducción de audio para acordes musicales.
Se provee la clase ChordPlayer, la cual permite:

  - Generar una onda sinusoidal para una frecuencia y duración específicas.
  - Generar el audio de un acorde combinando las ondas de cada nota.
  - Generar una secuencia de audio que incluya notas individuales (ascendentes y descendentes)
    y la reproducción del acorde completo.
  - Devolver un objeto IPython.display.Audio para reproducir el audio generado en el notebook.

Inventario Detallado:

1. Clase ChordPlayer:
   - **__init__(sample_rate: int = 44100, amplitude: float = 0.3)**
     - Inicializa el reproductor de audio con la frecuencia de muestreo y la amplitud.
     - Dependencias: Se usa para configurar parámetros iniciales.
   
   - **generate_sine_wave(frequency, duration) -> np.ndarray**
     - Genera una onda sinusoidal para una frecuencia y duración específicas.
     - Uso: Es la función base para la generación de notas.
     - Dependencias: numpy, matemáticas trigonométricas.
   
   - **generate_chord_audio(frequencies, duration) -> np.ndarray**
     - Genera el audio de un acorde sumando las ondas sinusoidales correspondientes a cada nota.
     - Uso: Se utiliza para reproducir el acorde completo.
     - Dependencias: generate_sine_wave, numpy.
   
   - **generate_sequence_audio(frequencies, individual_duration: float = 0.5, chord_duration: float = 1.0, delay: float = 0.2) -> np.ndarray**
     - Genera una secuencia completa de audio que incluye:
         • Notas individuales en orden ascendente, con un breve silencio entre ellas.
         • El acorde completo.
         • Notas individuales en orden descendente.
         • El acorde completo final.
     - Uso: Permite reproducir una secuencia completa con transiciones suaves.
     - Dependencias: generate_sine_wave, generate_chord_audio, numpy.
   
   - **play_sequence(frequencies, individual_duration: float = 0.6, chord_duration: float = 3.0, delay: float = 0.2) -> IPython.display.Audio**
     - Genera la secuencia de audio y devuelve un objeto de audio reproducible en el notebook.
     - Uso: Se llama para reproducir el audio generado en el entorno interactivo.
     - Dependencias: generate_sequence_audio, IPython.display.

Dependencias Generales:
  - numpy: Para operaciones numéricas y generación de secuencias.
  - IPython.display: Para devolver objetos de audio reproducibles en el notebook.
  - time (opcional, si se requiere medir tiempos, aunque no se usa directamente en esta versión).

"""

import numpy as np
import IPython.display as ipd

class ChordPlayer:
    def __init__(self, sample_rate: int = 44100, amplitude: float = 0.3):
        """
        Inicializa el reproductor de acordes con parámetros de audio.

        Args:
            sample_rate (int, opcional): Frecuencia de muestreo en Hz. Por defecto 44100.
            amplitude (float, opcional): Amplitud del audio. Por defecto 0.3.
        """
        self.sample_rate = sample_rate
        self.amplitude = amplitude

    def generate_sine_wave(self, frequency: float, duration: float) -> np.ndarray:
        """
        Genera una onda sinusoidal para una frecuencia y duración específicas.

        Args:
            frequency (float): Frecuencia de la nota en Hz.
            duration (float): Duración de la nota en segundos.

        Returns:
            np.ndarray: Array de datos de audio (int16).
        
        Uso:
          - Es la función base para la generación de audio de una nota.
        """
        frequency = float(frequency)  # Asegurar que la frecuencia sea float
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        wave = np.sin(2 * np.pi * frequency * t)
        # Convertir a formato de audio int16
        audio = (wave * (2**15 - 1) * self.amplitude).astype(np.int16)
        return audio

    def generate_chord_audio(self, frequencies: list, duration: float) -> np.ndarray:
        """
        Genera el audio de un acorde combinando las ondas sinusoidales de cada nota.

        Args:
            frequencies (list): Lista de frecuencias (en Hz) de las notas del acorde.
            duration (float): Duración del acorde en segundos.

        Returns:
            np.ndarray: Array de audio representando el acorde.
        
        Uso:
          - Se utiliza para reproducir el acorde completo.
        """
        audios = [self.generate_sine_wave(freq, duration) for freq in frequencies]
        combined = np.sum(audios, axis=0)
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = (combined * ((2**15 - 1) / max_val)).astype(np.int16)
        return combined

    def generate_sequence_audio(self, frequencies: list, individual_duration: float = 0.5,
                                chord_duration: float = 1.0, delay: float = 0.2) -> np.ndarray:
        """
        Genera una secuencia completa de audio que incluye:
          - Notas individuales en orden ascendente con un breve silencio entre ellas.
          - El acorde completo.
          - Notas individuales en orden descendente.
          - El acorde completo final.

        Args:
            frequencies (list): Lista de frecuencias del acorde.
            individual_duration (float, opcional): Duración de cada nota individual. Por defecto 0.5 s.
            chord_duration (float, opcional): Duración del acorde completo. Por defecto 1.0 s.
            delay (float, opcional): Duración del silencio entre sonidos en segundos. Por defecto 0.2 s.

        Returns:
            np.ndarray: Array de audio que representa la secuencia completa.
        
        Uso:
          - Se utiliza para generar secuencias de audio que permiten apreciar la transición
            entre notas y acordes.
        """
        sequence = []
        silence = np.zeros(int(self.sample_rate * delay), dtype=np.int16)

        # Notas individuales en orden ascendente
        for freq in frequencies:
            note = self.generate_sine_wave(freq, individual_duration)
            sequence.append(note)
            sequence.append(silence)

        # Acorde completo
        chord = self.generate_chord_audio(frequencies, chord_duration)
        sequence.append(chord)
        sequence.append(silence)

        # Notas individuales en orden descendente
        for freq in reversed(frequencies):
            note = self.generate_sine_wave(freq, individual_duration)
            sequence.append(note)
            sequence.append(silence)

        # Acorde final
        sequence.append(chord)

        return np.concatenate(sequence)

    def play_sequence(self, frequencies: list, individual_duration: float = 0.6,
                      chord_duration: float = 3.0, delay: float = 0.2) -> ipd.Audio:
        """
        Genera y devuelve un objeto de audio para reproducir la secuencia definida.

        Args:
            frequencies (list): Lista de frecuencias para el acorde.
            individual_duration (float, opcional): Duración de cada nota individual. Por defecto 0.6 s.
            chord_duration (float, opcional): Duración del acorde completo. Por defecto 3.0 s.
            delay (float, opcional): Duración del silencio entre sonidos en segundos. Por defecto 0.2 s.
        
        Returns:
            IPython.display.Audio: Objeto de audio reproducible en el notebook.
        
        Uso:
          - Se llama para reproducir el audio generado en el entorno interactivo.
        """
        audio_data = self.generate_sequence_audio(frequencies, individual_duration, chord_duration, delay)
        return ipd.Audio(audio_data, rate=self.sample_rate)

if __name__ == "__main__":
    # Ejemplo de uso para pruebas locales (se puede ejecutar en el entorno interactivo)
    test_frequencies = [261.63, 329.63, 392.00]  # Frecuencias de C, E, G
    player = ChordPlayer()
    audio_obj = player.play_sequence(test_frequencies)
    # En un entorno interactivo, se podría usar: display(audio_obj)
    print("ChordPlayer test: Audio object generado.")
