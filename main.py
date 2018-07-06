import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import glob
import time
from sklearn.svm import NuSVC,LinearSVC
from sklearn import svm, tree
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from car_detection_func import *

cars = []
not_cars = []

all_cars = glob.glob('/Users/Yifan/Desktop/CarND/CarND-Vehicle-Detection/vehicles/*/*.png')

all_not_cars = glob.glob('/Users/Yifan/Desktop/CarND/CarND-Vehicle-Detection/non-vehicles/*/*.png')

for i, car in enumerate(all_cars):
	if i % 1 == 0:
		cars.append(car)

for i, not_car in enumerate(all_not_cars):
	if i % 1 == 0:
		not_cars.append(not_car)

data_info = data_look(cars,not_cars)

print("n_cars:" ,data_info["n_cars"] , "    n_notcars   :" , data_info["n_notcars"])

##---------------------------------------------------------------------------------------

spatial = 16
histbin = 16
colorspace = 'YUV'
orient = 11
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
heatmap_threshold = 1

t = time.time()
car_features = extract_features(cars, cspace=colorspace, spatial_size=(spatial, spatial),
						hist_bins=histbin, orient=orient, pix_per_cell=pix_per_cell,
						cell_per_block=cell_per_block,hog_channel=hog_channel)
notcar_features = extract_features(not_cars, cspace=colorspace, spatial_size=(spatial, spatial),
						hist_bins=histbin, orient=orient, pix_per_cell=pix_per_cell,
						cell_per_block=cell_per_block,hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t),'Seconds to extract spatial, hist and HOG features....')

X = np.vstack((car_features, notcar_features)).astype(np.float64)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

X_scaler = StandardScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using: ', orient, 'orientations', pix_per_cell, 'pixels_per_cell and', cell_per_block,
		'cells_per_block')
print('Feature vector length: ', len(X_train[0]))

##svc = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20,10,5,2))
svc = LinearSVC()
#svc = SGDClassifier(loss="hinge", penalty="l2")
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t), 'Seconds to train SVC.....')
print('Test Accuracy of SVC= ', round(svc.score(X_test, y_test), 4))

t = time.time()
n_predict = 10
print('My SVC predicts', svc.predict(X_test[0:n_predict]))
print('For that ', n_predict, 'labels:', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t), 'Seconds to predict', n_predict, 'labels with SVC')

##----------------------------------------------------------------------------------------
'''
test_imgs = glob.glob('/Users/Yifan/Desktop/CarND/CarND-Vehicle-Detection/test_images/*.jpg')


test_img = mpimg.imread(test_imgs[0])
test_img1 = mpimg.imread(test_imgs[1])
test_img2 = mpimg.imread(test_imgs[2])	
test_img3 = mpimg.imread(test_imgs[3])
test_img4 = mpimg.imread(test_imgs[4])
test_img5 = mpimg.imread(test_imgs[5])
	

out_img = apply_sliding_window(test_img, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block, (spatial, spatial), histbin)
draw_img = detect_cars_with_heat(test_img, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, (spatial, spatial), histbin, 
						heatmap_threshold)

out_img1 = apply_sliding_window(test_img1, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block, (spatial, spatial), histbin)
draw_img1 = detect_cars_with_heat(test_img1, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, (spatial, spatial), histbin, 
						heatmap_threshold)

out_img2 = apply_sliding_window(test_img2, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block, (spatial, spatial), histbin)
draw_img2 = detect_cars_with_heat(test_img2, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, (spatial, spatial), histbin, 
						heatmap_threshold)

out_img3 = apply_sliding_window(test_img3, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block, (spatial, spatial), histbin)
draw_img3 = detect_cars_with_heat(test_img3, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, (spatial, spatial), histbin, 
						heatmap_threshold)

out_img4 = apply_sliding_window(test_img4, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block, (spatial, spatial), histbin)
draw_img4 = detect_cars_with_heat(test_img4, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, (spatial, spatial), histbin, 
						heatmap_threshold)

out_img5 = apply_sliding_window(test_img5, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block, (spatial, spatial), histbin)
draw_img5 = detect_cars_with_heat(test_img5, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, (spatial, spatial), histbin, 
						heatmap_threshold)

ffig1 = plt.figure(1)
plt.subplot(121)
plt.imshow(out_img[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img)
plt.title(' After Heatmap Car Positions ')

fig2 = plt.figure(2)
plt.subplot(121)
plt.imshow(out_img1[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img1)
plt.title(' After Heatmap Car Positions ')

fig3 = plt.figure(3)
plt.subplot(121)
plt.imshow(out_img2[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img2)
plt.title(' After Heatmap Car Positions ')

fig4 = plt.figure(4)
plt.subplot(121)
plt.imshow(out_img3[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img3)
plt.title(' After Heatmap Car Positions ')

fig5 = plt.figure(5)
plt.subplot(121)
plt.imshow(out_img4[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img4)
plt.title(' After Heatmap Car Positions ')

fig6 = plt.figure(6)
plt.subplot(121)
plt.imshow(out_img5[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img5)
plt.title(' After Heatmap Car Positions ')

plt.show()

'''

##-----------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------
'''
colorspace = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'

t = time.time()
car_features = hog_extract_features(cars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell,
						cell_per_block=cell_per_block,hog_channel=hog_channel)
notcar_features = hog_extract_features(not_cars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell,
						cell_per_block=cell_per_block,hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t),'Seconds to extract HOG features....')

X = np.vstack((car_features, notcar_features)).astype(np.float64)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

X_scaler = StandardScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using: ', orient, 'orientations', pix_per_cell, 'pixels_per_cell and', cell_per_block,
		'cells_per_block')
print('Feature vector length: ', len(X_train[0]))

##svc = MLPClassifier(solver='adam', hidden_layer_sizes=(32,8,2))
svc = LinearSVC()
#svc = SGDClassifier(loss="hinge", penalty="l2")
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t), 'Seconds to train SVC.....')
print('Test Accuracy of SVC= ', round(svc.score(X_test, y_test), 4))

t = time.time()
n_predict = 10
print('My SVC predicts', svc.predict(X_test[0:n_predict]))
print('For that ', n_predict, 'labels:', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t), 'Seconds to predict', n_predict, 'labels with SVC')

##----------------------------------------------------------------------------------------

test_imgs = glob.glob('/Users/Yifan/Desktop/CarND/CarND-Vehicle-Detection/test_images/*.jpg')


test_img = mpimg.imread(test_imgs[0])
test_img1 = mpimg.imread(test_imgs[1])
test_img2 = mpimg.imread(test_imgs[2])	
test_img3 = mpimg.imread(test_imgs[3])
test_img4 = mpimg.imread(test_imgs[4])
test_img5 = mpimg.imread(test_imgs[5])
	

heatmap_threshold = 3


out_img = hog_apply_sliding_window(test_img, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block)
draw_img = hog_detect_cars_with_heat(test_img, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, heatmap_threshold)

out_img1 = hog_apply_sliding_window(test_img1, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block)
draw_img1 = hog_detect_cars_with_heat(test_img1, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, heatmap_threshold)

out_img2 = hog_apply_sliding_window(test_img2, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block)
draw_img2 = hog_detect_cars_with_heat(test_img2, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, heatmap_threshold)

out_img3 = hog_apply_sliding_window(test_img3, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block)
draw_img3 = hog_detect_cars_with_heat(test_img3, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, heatmap_threshold)

out_img4 = hog_apply_sliding_window(test_img4, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block)
draw_img4 = hog_detect_cars_with_heat(test_img4, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, heatmap_threshold)

out_img5 = hog_apply_sliding_window(test_img5, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block)
draw_img5 = hog_detect_cars_with_heat(test_img5, svc, X_scaler, orient, 
						pix_per_cell, cell_per_block, heatmap_threshold)

ffig1 = plt.figure(1)
plt.subplot(121)
plt.imshow(out_img[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img)
plt.title(' After Heatmap Car Positions ')

fig2 = plt.figure(2)
plt.subplot(121)
plt.imshow(out_img1[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img1)
plt.title(' After Heatmap Car Positions ')

fig3 = plt.figure(3)
plt.subplot(121)
plt.imshow(out_img2[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img2)
plt.title(' After Heatmap Car Positions ')

fig4 = plt.figure(4)
plt.subplot(121)
plt.imshow(out_img3[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img3)
plt.title(' After Heatmap Car Positions ')

fig5 = plt.figure(5)
plt.subplot(121)
plt.imshow(out_img4[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img4)
plt.title(' After Heatmap Car Positions ')

fig6 = plt.figure(6)
plt.subplot(121)
plt.imshow(out_img5[0])
plt.title(' Car Positions')
plt.subplot(122)
plt.imshow(draw_img5)
plt.title(' After Heatmap Car Positions ')

plt.show()

'''
##---------------------------------------------------------------------------------------

from moviepy.editor import VideoFileClip 


def process_video(img):

	out_img = detect_cars_with_heat(img, svc, X_scaler, orient, pix_per_cell, 
					cell_per_block, (spatial,spatial), histbin, heatmap_threshold)

	return out_img

white_output = '/Users/Yifan/Desktop/CarND/CarND-Vehicle-Detection/out_test_video.mp4'

clip1 = VideoFileClip('/Users/Yifan/Desktop/CarND/CarND-Vehicle-Detection/test_video.mp4')
white_clip = clip1.fl_image(process_video)
white_clip.write_videofile(white_output, audio=False)























