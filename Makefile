all:
	cd training/ && python train_rects.py

rotate_rects:
	cd training/distort_training_data && clang++ rotate_images.cpp dir_utils.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -o rotate_rects
	cd training/distort_training_data && ./rotate_rects
	
