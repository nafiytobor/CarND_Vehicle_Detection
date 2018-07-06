import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label






def data_look(car_list,not_car_list):
	data_dict = {}
	data_dict["n_cars"] = len(car_list)
	data_dict["n_notcars"] = len(not_car_list)
	example_img = mpimg.imread(car_list[0])
	data_dict["image_shape"] = example_img.shape
	data_dict["data_type"] = example_img.dtype 
	return data_dict


def color_hist(img, nbins = 32):
	rhist = np.histogram(img[:,:,0], bins=nbins)
	ghist = np.histogram(img[:,:,1], bins=nbins)
	bhist = np.histogram(img[:,:,2], bins=nbins)

	hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

	return hist_features

def bin_spatial(img, size=(32,32)):
	
	bin1 = cv2.resize(img[:,:,0], size).ravel()
	bin2 = cv2.resize(img[:,:,1], size).ravel()
	bin3 = cv2.resize(img[:,:,2], size).ravel()

	return np.hstack((bin1, bin2, bin3))

def get_hot_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	
	if vis:
		return_list = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
				cells_per_block=(cell_per_block,cell_per_block), visualise=vis, 
				feature_vector=feature_vec, transform_sqrt=False)

		hog_features = return_list[0]
		hog_image = return_list[1]

		return hog_features, hog_image
	else:
		hog_features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
				cells_per_block=(cell_per_block,cell_per_block), visualise=vis, 
				feature_vector=feature_vec, transform_sqrt=False)

		return hog_features


def extract_features(imgs, cspace='RGB', spatial_size=(32,32), hist_bins=32, 
					orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, vis=False, 
					feature_vec=True):
	features = []

	for file in imgs:
		img = mpimg.imread(file)
		

		if cspace != 'RGB':
			if cspace == 'HSV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
			elif cspace == 'LUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
			elif cspace == 'HLS':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
			elif cspace == 'YUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
			elif cspace == 'YCrCb':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
		else: feature_image = np.copy(img)

		spatial_features = bin_spatial(feature_image, size=spatial_size)
		##print(spatial_features.shape)

		hist_features = color_hist(feature_image, nbins=hist_bins)
		##print(hist_features.shape)

		if hog_channel == 'ALL':
			hog_features = []

			for channel in range(feature_image.shape[2]):
				hog_features.append(get_hot_features(feature_image[:,:,channel], orient, 
								pix_per_cell, cell_per_block, vis, feature_vec))

			hog_features = np.ravel(hog_features)
		else:
			hog_features = get_hot_features(feature_image[:,:,hog_channel], orient, 
								pix_per_cell, cell_per_block, vis, feature_vec)

		features.append(np.concatenate((spatial_features, hist_features, hog_features)))

	return features

def convert_color(img, conv='RGB2YCrCb'):
	if conv == 'RGB2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	elif conv == 'RGB2HLS':
		return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	elif conv == 'RGB2LUV':
		return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
	elif conv == 'RGB2YUV':
		return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
			spatial_size, hist_bins):
	draw_img = np.copy(img)
	img = img.astype(np.float32)/255
	box_list = []

	img_tosearch = img[ystart:ystop, :, :]
	ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')

	if scale!= 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1	
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
	nfeat_per_block = orient * cell_per_block ** 2

	window = 64 
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1 
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

	hog1 = get_hot_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = get_hot_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = get_hot_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb * cells_per_step
			xpos = xb * cells_per_step

			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos * pix_per_cell
			ytop = ypos * pix_per_cell

			subimg = cv2.resize(ctrans_tosearch[ytop:ystop+window, xleft:xleft+window], (64,64))
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

			

			test_features = X_scaler.transform(np.hstack((spatial_features, hist_features,
											hog_features)).reshape(1,-1))
			test_prediction = svc.predict(test_features)

			if test_prediction == 1:
				xbox_left = np.int(xleft * scale)
				ytop_draw = np.int(ytop * scale)
				win_draw = np.int(window * scale)

				cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, 
							ytop_draw + win_draw + ystart), (0,0,255), 6)
				box_list.append(( (int(xbox_left), int(ytop_draw + ystart)), 
						(int(xbox_left + win_draw), int(ytop_draw + win_draw + ystart)) ))

	return draw_img, box_list


def add_heat(heatmap, bbox_list):
	for box in bbox_list:
		
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	return heatmap


def apply_threshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0

	return heatmap

def draw_labeled_bboxes(img, labels):
	for car_number in range(1, labels[1] + 1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

	return img


def detect_cars_with_heat(img, svc, X_scaler, orient, pix_per_cell,
				cell_per_block, spatial_size, hist_bins, heatmap_threshold):
	
	out_img, bbox_list = apply_sliding_window(img, svc, X_scaler, orient, 
							pix_per_cell, cell_per_block, spatial_size, hist_bins)

	heat_zero = np.zeros_like(img[:,:,0].astype(np.float))
	heat_add_box = add_heat(heat_zero, bbox_list)
	heat_add_box_thres = apply_threshold(heat_add_box, heatmap_threshold)

	heatmap = np.clip(heat_add_box_thres, 0, 255)
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(img), labels)


	return draw_img	



def apply_sliding_window(img, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, 
						hist_bins):
	bbox_list = []
	ystart = 400
	ystop = 480
	out_img, box1 = find_cars(img, ystart, ystop, 1.0, svc, X_scaler, orient, pix_per_cell, 
							cell_per_block, spatial_size, hist_bins)
	
	ystart = 400
	ystop = 560
	out_img, box2 = find_cars(out_img, ystart, ystop, 1.2, svc, X_scaler, orient, pix_per_cell, 
							cell_per_block, spatial_size, hist_bins)
	
	ystart = 400
	ystop = 620
	out_img, box3 = find_cars(out_img, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, 
							cell_per_block, spatial_size, hist_bins)
	
	ystart = 400
	ystop = 656
	out_img, box4 = find_cars(out_img, ystart, ystop, 2, svc, X_scaler, orient, pix_per_cell, 
							cell_per_block, spatial_size, hist_bins)
	
	bbox_list.extend(box1)
	bbox_list.extend(box2)
	bbox_list.extend(box3)
	bbox_list.extend(box4)
	

	
	return out_img, bbox_list



##------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------


def hog_extract_features(imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, 
					hog_channel=0, vis=False, feature_vec=True):
	features = []

	for file in imgs:
		img = mpimg.imread(file)
		

		if cspace != 'RGB':
			if cspace == 'HSV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
			elif cspace == 'LUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
			elif cspace == 'HLS':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
			elif cspace == 'YUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
			elif cspace == 'YCrCb':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
		else: feature_image = np.copy(img)

		
		if hog_channel == 'ALL':
			hog_features = []

			for channel in range(feature_image.shape[2]):
				hog_features.append(get_hot_features(feature_image[:,:,channel], orient, 
								pix_per_cell, cell_per_block, vis, feature_vec))

			hog_features = np.ravel(hog_features)
		else:
			hog_features = get_hot_features(feature_image[:,:,hog_channel], orient, 
								pix_per_cell, cell_per_block, vis, feature_vec)

		features.append(hog_features)

	return features




def hog_find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block):
	draw_img = np.copy(img)
	img = img.astype(np.float32)/255
	box_list = []

	img_tosearch = img[ystart:ystop, :, :]
	ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')

	if scale!= 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1	
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
	nfeat_per_block = orient * cell_per_block ** 2

	window = 64 
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1 
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

	hog1 = get_hot_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = get_hot_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = get_hot_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb * cells_per_step
			xpos = xb * cells_per_step

			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos * pix_per_cell
			ytop = ypos * pix_per_cell


			test_features = X_scaler.transform((hog_features).reshape(1,-1))
			test_prediction = svc.predict(test_features)

			if test_prediction == 1:
				xbox_left = np.int(xleft * scale)
				ytop_draw = np.int(ytop * scale)
				win_draw = np.int(window * scale)

				cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, 
							ytop_draw + win_draw + ystart), (0,0,255), 6)
				box_list.append(( (int(xbox_left), int(ytop_draw + ystart)), 
						(int(xbox_left + win_draw), int(ytop_draw + win_draw + ystart)) ))

	return draw_img, box_list



def hog_detect_cars_with_heat(img, svc, X_scaler, orient, pix_per_cell,
				cell_per_block, heatmap_threshold):
	
	out_img, bbox_list = hog_apply_sliding_window(img, svc, X_scaler, orient, 
							pix_per_cell, cell_per_block)

	heat_zero = np.zeros_like(img[:,:,0].astype(np.float))
	heat_add_box = add_heat(heat_zero, bbox_list)
	heat_add_box_thres = apply_threshold(heat_add_box, heatmap_threshold)

	heatmap = np.clip(heat_add_box_thres, 0, 255)
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(img), labels)


	return draw_img	



def hog_apply_sliding_window(img, svc, X_scaler, orient, pix_per_cell, cell_per_block):
	bbox_list = []
	ystart = 400
	ystop = 480
	out_img, box1 = hog_find_cars(img, ystart, ystop, 1.0, svc, X_scaler, orient, pix_per_cell, 
							cell_per_block)
	
	ystart = 400
	ystop = 560
	out_img, box2 = hog_find_cars(out_img, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, 
							cell_per_block)
	
	ystart = 400
	ystop = 620
	out_img, box3 = hog_find_cars(out_img, ystart, ystop, 1.7, svc, X_scaler, orient, pix_per_cell, 
							cell_per_block)
	
	ystart = 400
	ystop = 656
	out_img, box4 = hog_find_cars(out_img, ystart, ystop, 2, svc, X_scaler, orient, pix_per_cell, 
							cell_per_block)
	
	bbox_list.extend(box1)
	bbox_list.extend(box2)
	bbox_list.extend(box3)
	bbox_list.extend(box4)
	

	
	return out_img, bbox_list





























