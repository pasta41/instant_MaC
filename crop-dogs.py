from imageai.Detection import ObjectDetection
from PIL import Image
import os
from os.path import isfile, join


execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

for dir in os.listdir("./data/stanford-dogs/images/Images"):
    if not os.path.exists(os.path.join("./data/stanford-dogs/cropped-images",dir)):
      os.mkdir(os.path.join("./data/stanford-dogs/cropped-images",dir))
      print("Making dir: " + dir)
      for f in os.listdir(os.path.join("./data/stanford-dogs/images/Images",dir)):
        detections = detector.detectObjectsFromImage(input_image=os.path.join("./data/stanford-dogs/images/Images", dir, f), output_image_path=os.path.join(execution_path , "tempdog.jpg"), minimum_percentage_probability=50)

        for eachObject in detections:
            if(eachObject["name"] == "cat" or eachObject["name"] == "dog"):
                print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
                image = Image.open(join("./data/stanford-dogs/images/Images",dir,f))
                # some of the images are rgba and we fail on saving a jpg
                image_rgb = image.convert('RGB')
                cropped = image_rgb.crop(eachObject["box_points"])
                cropped.save(os.path.join("./data/stanford-dogs/cropped-images",dir,f))
                print("--------------------------------")
        print("Processed: " + f)

