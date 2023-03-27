# Achira_Answer

## Part I -  
Generation file

File - gen_images.py can generate images and labels for them.<br>
- It randomly selects a point and draws a sqaure of size 64x64 around it, it checks also if it intersects with a previous square and if so removes it.
- A shape is selected randomly from the given list and rotates and resizes it and fixes it in the generated sqaure.
- It uses the points of sqaure and selected shape to generated label in YOLO format - [class, x, y, w, h]

### usage - 
python3 gen_images.py --input input_images/ --out-dims 1024 --nout 1000 --labels(if labels are required)

## Part II -
DNN

YOLOv5 small has been used, cloned from https://github.com/ultralytics/yolov5 <br>
the YOLO_DNN.ipynb - google colab note contains - <br>
- generation of data (train, test, validation)
- training them using a custom data file - custom.yaml, trainng was done for 715 epochs - 600 trainng images, 40 validation images.
 <img src="./images/results.png" alt="training loss curves" title="training loss curves">
- function for prediction, generates a result image with bounding boxes over shapes, lists the predicted number of shapes.
![traininf loss curves](./images/result.jpg "result of predict.jpg")
<img src="./images/result.jpg" alt="result" title="result of predict.jpg">
- Inference cell to get accuracy plots. accuracy metrics

 ![](./images/P_curve.png) | ![](./images/R_curve.png) 
:-------------------------:|:-------------------------:
      Precision curve      |         Recall curve      
      
<img src="./images/F1_curve.png" alt="f1" title="F1 curve">
 <img src="./images/mAP.png" alt="map" title="mAP Scores">
 <img src="./images/confusion_matrix.png" alt="cm" title="Confusion Matrix">
