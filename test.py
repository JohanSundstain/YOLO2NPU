import onnx

# Загружаем модель
model = onnx.load("onnx_model/yolov8n_sim.onnx")

# Выводим имена всех входов
print("Input names:")
for inp in model.graph.input:
    print(f"  - {inp.name}")

# Выводим имена всех выходов (тоже полезно)
print("Output names:")
for out in model.graph.output:
    print(f"  - {out.name}")