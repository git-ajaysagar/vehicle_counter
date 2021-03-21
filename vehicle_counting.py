def vehicle_counter(path_to_video,left_coords_of_line,right_coords_of_line):
    #importing required libraries
    import cv2 as cv
    import numpy as np

    #loading pre-trained yolov3 model

    labels=[]  #loading classes on which yolo is pre-trained
    with open('coco.names.txt','r') as f:
        labels=[i.strip() for i in f.readlines()]

    #loading config file and weights file in opencv dnn module for vehicle detection
    network=cv.dnn.readNet('yolov3.weights','yolov3.cfg.txt')

    #extracting layers which will be used to classify vehicles
    layer_names=network.getLayerNames()
    out_layers = list([layer_names[i[0] - 1] for i in network.getUnconnectedOutLayers()])
    # print(out_layers)

    #assigning different colors to different classes
    label_box_colors=np.random.uniform(0,255,(len(labels),3))

    #loading video
    frames=cv.VideoCapture(path_to_video)
    n=0       #counter

    #looping through frames of video
    while (frames.isOpened()):
            _,test_img=frames.read()
            if _==True:
                cv.putText(test_img, 'No. of vehicles crossed: '+str(n), (50,100), cv.FONT_HERSHEY_PLAIN, 4, (0, 70, 255), 3)
                test_img=cv.resize(test_img1,None,None,0.5,0.5)
                # test_img=cv.fastNlMeansDenoisingColored(test_img,None,10,10,7,21)         #can de-noise if required (at the cost of reduced speed)
                img_height,img_width,channels=test_img.shape
                #converting image to blob for better feature detection
                img_blob= cv.dnn.blobFromImage(test_img,0.00392,(160,160),(0,0,0),True,False)
                #setting blob as input to the network
                network.setInput(img_blob)
                #this output vector contains all the necessary information of detected object
                output_vectors=network.forward(out_layers)

                #extracting detected objects, their confidence and also co-ordinates of their location in the image from output_vectors
                class_labels=[]
                confidences=[]
                rec_coords=[]
                for iterator1 in output_vectors:
                    for detections in iterator1:
                        det_scores=detections[5:]
                        # print(det_scores)
                        class_label=np.argmax(det_scores)
                        confidence=det_scores[class_label]
                        if confidence>0.5:
                            center_coordx=int(detections[0]*img_width)
                            center_coordy=int(detections[1]*img_height)
                            w=int(detections[2]*img_width)
                            h=int(detections[3]*img_height)

                            x=int(center_coordx-w/2)
                            y=int(center_coordy-h/2)
                            rec_coords.append([x,y,w,h])
                            confidences.append(float(confidence))
                            class_labels.append(class_label)

                #removing multiple detection of same object by using non-max supression, basically removing multiple rectangles on single object
                suppressed=cv.dnn.NMSBoxes(rec_coords,confidences,0.5,0.5)

                #drawing a line on the image
                line=cv.line(test_img,(left_coords_of_line),(right_coords_of_line),(0,255,0),2)
                #drawing bounding box around vehicles and labeling them
                for i,j,k in zip(class_labels,confidences,rec_coords):
                    if labels[i]=='car' or labels[i]=='bus' or labels[i]=='motorbike' or labels[i]=='truck':
                        if rec_coords.index(k) in suppressed:
                            x,y,w,h=k
                            cv.rectangle(test_img,(x,y),(x+w,y+h),label_box_colors[i],2)
                            if int((y+h/2)+3)>int((img_height/1.5)-2) and int((y+h/2)-3)<int((img_height/1.5)+2) :    #condition to count a vehicle
                                n+=1
                            class_label=labels[i]
                            cv.putText(test_img,class_label,(x-5,y-8),cv.FONT_HERSHEY_PLAIN,1,(0,255,9),1)
                cv.namedWindow('s',cv.WINDOW_KEEPRATIO)
                cv.imshow('s',test_img)
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
    cv.destroyAllWindows()
    frames.release()

def vehicle_counter(path_to_video,left_coords_of_line,right_coords_of_line)


