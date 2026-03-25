from ultralytics import YOLO


class Stage:
    def __init__(self, func, comment):
        self.func = func
        self.comment = comment

    def __iter__(self):
        yield self.func
        yield self.comment


class Pipeline:
	def __init__(self, model_path):
		self.model = YOLO(model_path)
		self.stages = []

	def add_stage(self, stage):
		self.stages.append(stage)

	def run(self):
		for func, comment in self.stages:
			print(comment)
			func()	

	def pth2onnx(self):
		self.model.export(format='onnx', opset=12, simplify=False, dynamic=False)

	def onnx2rknn(self):
		

		

if __name__ == "__main__":
	pipeline = Pipeline('yolov8n.pt')
	pipeline.run()