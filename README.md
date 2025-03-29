# Laboratorio-4 FATIGA MUSCULAR
## Configuración inicial
### En esta parte se prepara el codigo para la captura de la señal a traves del DAQ. Se configura la frecuencia de muestreo, la duracion de la señal y el archivo de salida deseado.
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
### En esta parte previa de la adquisicion de la señal, se realiza una etapa de guardado de datos por vectores en un archivo.csv
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
### En esta etapa de codigo se filtra la señal capturada, teniendo como banda de trabajo desde los 20 Hz hasta los 450 Hz.
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
### Como etapa final del codigo en MATLAB, se realiza el guardado de los datos ya filtrados en el archivo de salida
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
from scipy.signal import get_window, spectrogram, hilbert
from scipy.stats import ttest_ind
```
## Cargar datos donde se guardo la señal
```python
# ======= CARGAR DATOS =======
archivo = "emg_signal_filtered.csv"  # Nombre del archivo CSV
datos = pd.read_csv(archivo)  # Leer archivo CSV
archivo2= "emg_signal_filtered1.csv"  
datos2 = pd.read_csv(archivo)
```
## Establecer ejes
```python
# Extraer columnas
tiempo = datos["Tiempo (s)"]
voltaje = datos["Voltaje Filtrado (V)"]

tiempo2 = datos2["Tiempo (s)"]
voltaje2 = datos2["Voltaje Filtrado (V)"]
```
## Graficar la señal
```python
# ======= GRAFICAR SEÑAL =======
plt.figure(figsize=(10, 5))
plt.plot(tiempo, voltaje, label="Señal EMG Filtrada", color='b')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal EMG Filtrada")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(tiempo2, voltaje2, label="Señal EMG Filtrada", color='b')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal EMG Filtrada")
plt.legend()
plt.grid()
plt.show()
```
