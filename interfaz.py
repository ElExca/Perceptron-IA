import tkinter as tk
from tkinter import filedialog, ttk
import perceptron as pl
import threading

def seleccionar_archivo():
    filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    archivo_label.config(text=filename)
    return filename

def iniciar_entrenamiento():
    eta = float(eta_entry.get())
    epocas = int(epocas_entry.get())
    ruta_del_archivo = archivo_label.cget("text")
    threading.Thread(target=lambda: pl.entrenar_perceptron(eta, epocas, ruta_del_archivo, progreso_barra)).start()

def mostrar_graficas():
    pl.mostrar_resultados()

def generar_reporte():
    pesos_iniciales, pesos_finales, epocas, error = pl.obtener_pesos()
    reporte = (f"Número de Épocas: {epocas}\n"
               f"Error permisible: {error}\n\n"
               f"Pesos iniciales:\n{pesos_iniciales}\n\n"
               f"Pesos finales:\n{pesos_finales}")
    tk.messagebox.showinfo("Reporte de Pesos", reporte)

root = tk.Tk()
root.title("Entrenamiento del Perceptrón")
root.geometry('600x400')

style = ttk.Style()
style.theme_use('clam')

frame = ttk.Frame(root, padding="10 10 10 10")
frame.pack(fill=tk.BOTH, expand=True)

ttk.Label(frame, text="Tasa de aprendizaje (eta):").pack(pady=5)
eta_entry = ttk.Entry(frame)
eta_entry.pack()

ttk.Label(frame, text="Número de épocas:").pack(pady=5)
epocas_entry = ttk.Entry(frame)
epocas_entry.pack()

ttk.Button(frame, text="Seleccionar archivo CSV", command=seleccionar_archivo).pack(pady=10)
archivo_label = ttk.Label(frame, text="")
archivo_label.pack(pady=5)

ttk.Button(frame, text="Iniciar entrenamiento", command=iniciar_entrenamiento).pack(pady=10)

# Barra de progreso
progreso_barra = ttk.Progressbar(frame, orient="horizontal", length=200, mode="determinate")
progreso_barra.pack(pady=10)

ttk.Button(frame, text="Mostrar Gráficas", command=mostrar_graficas).pack(pady=10)

ttk.Button(frame, text="Generar Reporte", command=generar_reporte).pack(pady=10)

root.mainloop()
