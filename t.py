import cv2 
  
  
# define a video capture object 
# vid = cv2.VideoCapture('rtsp://192.168.30.125:554/mainstream1') 
vid = cv2.VideoCapture(0) 

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

while(True): 

    ret, frame = vid.read() 
    if True:
        # frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        text = "Not Blurry"
        if fm < 100:
            text = "Blurry"
        h, w, c = frame.shape
        print(f'height: {h}, width: {w}')
        frame = cv2.putText(frame, f"{text}: {fm}", (int(h*0.1), int(w*0.1)),
			cv2.FONT_ITALIC, h*w/300000, (0, 0, 255), int(h*w/100000))
    	# Display the resulting frame 
        cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 