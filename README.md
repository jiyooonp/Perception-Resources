# Perception-Resources
Perception Algorithm Tutorial Implementations 

## Faster-RCNN
https://debuggercafe.com/faster-rcnn-object-detection-with-pytorch/
- basic frame to use the model on images and videos

for error:
```angular2html
cv2.error: OpenCV(4.7.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1272: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'
```
https://github.com/opencv/opencv-python/issues/18

## Custom Object Detection using PyTorch Faster RCNN

https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
- training Faster RCNN to identify between 5 microcontrollers
- Takes 7+ min for one epoch (RTX 3050)