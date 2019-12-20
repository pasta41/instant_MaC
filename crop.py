from imageai.Detection import ObjectDetection
from PIL import Image
import os
from os.path import isfile, join


execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()


catfiles = [f for f in os.listdir("./cats") if isfile(join("./cats", f))]
for f in catfiles:
	detections = detector.detectObjectsFromImage(input_image=os.path.join("./cats" , f), output_image_path=os.path.join(execution_path , "tempcat.jpg"), minimum_percentage_probability=50)

	for eachObject in detections:
		if(eachObject["name"] == "cat" or eachObject["name"] == "dog"):
			print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
			image = Image.open(join("./cats",f))
			cropped = image.crop(eachObject["box_points"])
			cropped.save(os.path.join("cropped-cats",f))
		print("--------------------------------")


