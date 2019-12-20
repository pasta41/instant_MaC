from imageai.Detection import ObjectDetection
import os
from os.path import isfile, join


execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()


catfiles = [f for f in os.listdir("./cats") if isfile(join("./cats", f))]
for f in catfiles:
	detections, objects_path = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , f), 
															   output_image_path=os.path.join(execution_path , "tempcat.jpg"), 
															   minimum_percentage_probability=60,  extract_detected_objects=True)

	for eachObject, eachObjectPath in zip(detections, objects_path):
		print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
		print("Object's image saved in " + eachObjectPath)
		print("--------------------------------")

