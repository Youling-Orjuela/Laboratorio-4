# Laboratorio-4 FATIGA MUSCULAR
## Introducción
El electromiograma (EMG) es una técnica utilizada para registrar la actividad eléctrica de los músculos, conocida también como actividad mioeléctrica. Este proceso es esencial para el diagnóstico de diversas condiciones musculares y neuromusculares. Existen dos tipos principales de EMG: el de superficie y el intramuscular o de aguja, cada uno con sus aplicaciones específicas [1].

Para llevar a cabo la medición de las señales mioeléctricas, se emplean dos electrodos activos y un electrodo de tierra. En el caso de los electrodos de superficie, estos deben colocarse en la piel, directamente sobre el músculo que se desea estudiar. El electrodo de tierra, por otro lado, se conecta a una zona del cuerpo que esté eléctricamente activa, funcionando como referencia. La señal registrada por el EMG se obtiene al medir la diferencia de potencial entre las señales de los electrodos activos, lo que permite analizar la actividad eléctrica del músculo en cuestión [2].

El objetivo de esta práctica es aplicar el filtrado de señales continuas para procesar una señal electromigráfica y detectar la fatiga muscular a través del análisis espectral de la misma.

## Procedimiento
Primero se colocaron los electrodos en el músculo a estudiar que en nuestro caso fue el biceps, después se conectaron al sensor de electrocardiograma y este a su vez se conecto al DAQ NI USB6001/6002/6003
## Configuración inicial
En esta parte se prepara el codigo para la captura de la señal a traves del DAQ. Se configura la frecuencia de muestreo, la duracion de la señal y el archivo de salida deseado.
```matlab
% ======= CONFIGURACIÓN =======
device = 'Dev3';     % Nombre de tu DAQ
channel = 'ai0';     % Canal de entrada
sampleRate = 1000;   % Frecuencia de muestreo (Hz)
duration = 120;      % Duración total (segundos)
outputFile = 'emg_signal_filtered.csv';  % Archivo de salida

% ======= CREAR SESIÓN =======
d = daq("ni");  % Crear sesión para DAQ NI
addinput(d, device, channel, "Voltage");  % Agregar canal de entrada
d.Rate = sampleRate;

% ======= VARIABLES =======
timeVec = [];   % Vector de tiempo
signalVec = []; % Vector de señal

% ======= CONFIGURAR GRÁFICA =======
figure('Name', 'Señal en Tiempo Real', 'NumberTitle', 'off');
h = plot(NaN, NaN);
xlabel('Tiempo (s)');
ylabel('Voltaje (V)');
title('Señal EMG en Tiempo Real');
xlim([0, duration]);
ylim([-1, 5]);  % Ajuste de voltaje según rango esperado
grid on;
```
## Adquisición y guardado
En esta parte previa de la adquisicion de la señal, se realiza una etapa de guardado de datos por vectores en un archivo.csv
```matlab
% ======= ADQUISICIÓN Y GUARDADO =======
disp('Iniciando adquisición...');
startTime = datetime('now');

while seconds(datetime('now') - startTime) < duration
    % Leer una muestra
    [data, timestamp] = read(d, "OutputFormat", "Matrix");
    
    % Guardar datos en vectores
    t = seconds(datetime('now') - startTime);
    timeVec = [timeVec; t];
    signalVec = [signalVec; data];
    
    % Actualizar gráfica
    set(h, 'XData', timeVec, 'YData', signalVec);
    drawnow;
end
```
## Filtrado de la señal
En esta etapa de codigo se filtra la señal capturada, teniendo como banda de trabajo desde los 20 Hz hasta los 450 Hz.
```matlab
% ======= FILTRADO DE SEÑAL =======
disp('Aplicando filtros...');

% Definir frecuencias de corte
lowCut = 20;  % Filtro pasa altas (elimina ruido de movimiento)
highCut = 450; % Filtro pasa bajas (elimina ruido de alta frecuencia)

% Diseño de filtros
[bHigh, aHigh] = butter(2, lowCut/(sampleRate/2), 'high'); % Pasa altas
[bLow, aLow] = butter(2, highCut/(sampleRate/2), 'low');   % Pasa bajas

% Aplicar filtros en cascada
filteredSignal = filtfilt(bHigh, aHigh, signalVec);
filteredSignal = filtfilt(bLow, aLow, filteredSignal);
```
## Guardar datos
Como etapa final del codigo en MATLAB, se realiza el guardado de los datos ya filtrados en el archivo de salida
```matlab
% ======= GUARDAR LOS DATOS FILTRADOS =======
disp('Adquisición finalizada. Guardando archivo...');
T = table(timeVec, filteredSignal, 'VariableNames', {'Tiempo (s)', 'Voltaje Filtrado (V)'});
writetable(T, outputFile);
disp(['Datos guardados en: ', outputFile]);

% ======= GRÁFICA DE SEÑAL FILTRADA =======
figure;
plot(timeVec, filteredSignal);
xlabel('Tiempo (s)');
ylabel('Voltaje Filtrado (V)');
title('Señal EMG Filtrada');
grid on;

% ======= CERRAR SESIÓN =======
clear d;
```
## Librerias
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import t
```
numpy (np): Se utiliza para realizar operaciones matemáticas avanzadas y manejo de matrices.
pandas (pd): Sirve para manipular y analizar datos tabulares.
matplotlib.pyplot (plt): Permite generar gráficos para visualizar las señales EMG.
scipy.signal.hilbert: Usada para aplicar la Transformada de Hilbert y extraer la envolvente de la señal EMG.
scipy.stats.t: Calcula valores críticos del estadístico t y permite realizar la prueba de hipótesis.


## Cargar archivo CSV
```python
archivo2 = "emg_signal_filtered1.csv"
datos2 = pd.read_csv(archivo2)
```
archivo2: Especifica el nombre del archivo CSV que contiene la señal EMG filtrada.
pd.read_csv(archivo2): Carga los datos desde el archivo CSV en un DataFrame de pandas, que permite manipular columnas fácilmente.
## Establecer ejes
```python
# Extraer columnas
tiempo2 = datos2["Tiempo (s)"]
voltaje2 = datos2["Voltaje Filtrado (V)"]
```
Tiempo (s): Representa el eje temporal de la señal EMG.
Voltaje Filtrado (V): Contiene los valores de la señal EMG filtrada en función del tiempo.
## Graficar la señal
```python
# ======= GRAFICAR SEÑAL =======
plt.figure(figsize=(10, 5))
plt.plot(tiempo2, voltaje2, label="Señal EMG Filtrada", color='b')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal EMG Filtrada")
plt.legend()
plt.grid()
plt.show()
```
##Transformada de Hilbert y Envolvente de la Señal
```python
voltaje2 = np.array(voltaje2).flatten()
analytic_signal = hilbert(voltaje2)
envelope = np.abs(analytic_signal)
```
Transformada de Hilbert: Se aplica a la señal EMG filtrada para obtener la señal analítica, que incluye información de fase y amplitud.
Envolvente: La envolvente se calcula tomando el valor absoluto de la señal analítica. Esto nos permite extraer el "contorno" de la señal EMG, que es útil para detectar patrones y analizar la actividad muscular.
##Aplicar una Ventana de Hanning y Suavizar la Envolvente
```python
window_size = 200  
hanning_window = np.hanning(window_size)
envelope_smoothed = np.convolve(envelope, hanning_window, mode='same') / sum(hanning_window)
```
Ventana de Hanning: Se crea una ventana de Hanning de tamaño 200. Esta ventana suaviza la envolvente de la señal, reduciendo el ruido.
Suavizado de la Envolvente: La convolución entre la envolvente y la ventana de Hanning suaviza la señal resultante, reduciendo fluctuaciones bruscas.
##Graficar la Envolvente Original y Suavizada
```python
plt.figure(figsize=(10, 5))
plt.plot(tiempo2, envelope, label="Envolvente EMG (Original)", alpha=0.5)
plt.plot(tiempo2, envelope_smoothed[:len(tiempo2)], label="Envolvente con Hanning", color="orange", linewidth=2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Envolvente Suavizada con Ventana de Hanning")
plt.legend()
plt.grid()
plt.show()

```
##Extraer y Graficar la Primera y Última Ventana
```python
# ======= Extraer Primera y Última Ventana de Hanning =======
first_window = envelope[:window_size] * hanning_window
last_window = envelope[-window_size:] * hanning_window
first_window /= np.max(first_window)
last_window /= np.max(last_window)

# Mostrar ventanas seleccionadas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(tiempo2[:window_size], first_window, label="Primera Ventana de Hanning", color="blue")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud Normalizada")
plt.title("Primera Ventana de Hanning")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(tiempo2.iloc[-window_size:], last_window, label="Última Ventana de Hanning", color="red")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud Normalizada")
plt.title("Última Ventana de Hanning")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
```
Primera y Última Ventana: Se seleccionan las primeras y últimas ventanas de 200 puntos de la envolvente y se aplican las ventanas de Hanning correspondientes.

##Calcular Frecuencia Mediana

```python
def calcular_frecuencia_mediana(ventana, sampling_rate=1000):
    fft_vals = np.abs(np.fft.fft(ventana))
    fft_freqs = np.fft.fftfreq(len(ventana), 1 / sampling_rate)
    positive_freqs = fft_freqs[fft_freqs > 0]
    positive_fft_vals = fft_vals[fft_freqs > 0]
    cumulative_sum = np.cumsum(positive_fft_vals)
    median_freq_index = np.searchsorted(cumulative_sum, cumulative_sum[-1] / 2)
    return positive_freqs[median_freq_index]
```
Esta función calcula la frecuencia mediana de la señal usando la Transformada Rápida de Fourier (FFT). La frecuencia mediana es la frecuencia que divide la energía de la señal en dos partes iguales.

##Calcular Varianza y Estadístico t para Prueba de Hipótesis

```python
var_first = np.var(first_window, ddof=1)
var_last = np.var(last_window, ddof=1)
n1, n2 = len(first_window), len(last_window)
t_calculado = (freq_mediana_first - freq_mediana_last) / np.sqrt((var_first / n1) + (var_last / n2))

```
##Calcular grados de libertad{

```python
df = ((var_first / n1) + (var_last / n2))**2 / (((var_first / n1)**2 / (n1 - 1)) + ((var_last / n2)**2 / (n2 - 1)))
t_critico = t.ppf(0.975, df)

```

##Conclusión del Test de Hipótesis

```python
if abs(t_calculado) > t_critico:
    print("\nConclusión: Se rechaza la hipótesis nula (H₀). Hay una diferencia significativa.")
else:
    print("\nConclusión: No se rechaza la hipótesis nula (H₀). No hay evidencia significativa de fatiga muscular.")
```
## Referencias
[1] “Electromiografía - Mayo Clinic.” https://www.mayoclinic.org/es/tests-procedures/emg/about/pac-20393913
[2] H. Tankisi et al., “Standards of instrumentation of EMG,” Clinical Neurophysiology, vol. 131, no. 1, pp. 243–258, Nov. 2019, doi: 10.1016/j.clinph.2019.07.025.
