# actionRecog


# improved_trajectory-master
## already debug

# for ariginal version:
>>makedir build
>>cd build
>>cmake ..
>>make

# to execute the program
>>cd build
>>./DenseTrackStab -f videofile.avi


# A modified version:improved trjectory_v2
## 1.compile similar to iDT program
## 2.fixed some code that make the trajectory show successfully.

# compile:
>>make

# to execute the program
>>cd release
>>./DenseTrackStab -f videofile.avi


##
f  | video_file     | test.avi | filename of video 
o  | idt_file   | test.bin | filename of idt features }"
			
// @zz
//r  | tra_file   | tra.bin  | filename of track files  

L  | track_length   | 15 | the length of trajectory
S  | start_frame     | 0 | start frame of tracking 
E  | end_frame | 1000000 | end frame of tracking 
W  | min_distance | 5 | min distance 
N  | patch_size   | 32  | patch size
s  | nxy_cell  | 2 | descriptor parameter 
t  | nt_cell  | 3 | discriptor parameter
A  | scale_num  | 8 | num of scales
I  | init_gap  | 1 | gap
T  | show_track | 0 | whether show tracks
###

# dir TDD
extract TDD_FV feature


## ppipeline
(install matlab with caffe,opencv,ffmpeg)
1.make iDT_v2:
cd improved_trajectory_v2
make


2.extract iTDD:
MATLAB: ADDPATH iTDD
script_extraTra_ucf11.m


