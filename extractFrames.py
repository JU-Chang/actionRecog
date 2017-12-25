import cv2
import os
import rarfile
import shutil
from tqdm import tqdm

rarfile_path = '/home/chang/data/HMDB51/hmdb51_org'
source_path = '/home/chang/data/HMDB51/hmdb51_org/unrar'
destination_path = '/home/chang/data/HMDB51/hmdb51_jpg'

if os.path.exists(source_path):
    shutil.rmtree(source_path,ignore_errors=True)

if os.path.exists(destination_path):
    shutil.rmtree(destination_path,ignore_errors=True)
os.makedirs(source_path)
os.makedirs(destination_path)

# Unrar video file
for rarfiles in os.listdir(rarfile_path):
    if rarfiles.endswith('.rar'):
        rarf = rarfile.RarFile(os.path.join(rarfile_path,rarfiles))
        print('unrar:%s' % rarfiles)
        rarf.extractall(source_path)
        rarf.close()

# Extract each frame of video and store as jpg
# tqdm is used to show progress meter.install tqdm package before running the code
for files in tqdm(os.listdir(source_path)):
    # print('processing:%s' % files)
    dest_dir1 = os.path.join(destination_path,files)
    if not os.path.exists(dest_dir1):
        os.makedirs(dest_dir1)
    for filename in os.listdir(os.path.join(source_path, files)):
        if filename.endswith('.avi'):
            # print('>>>%s' % filename)
            frame_count = 0
            cap = cv2.VideoCapture(os.path.join(source_path, files, filename))
            if cap.isOpened():
                dest_dir = os.path.join(destination_path, files, str(filename.split('.', 1)[0]))
                while True:
                    rval, frame = cap.read()
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)
                    jpg_filename = os.path.join(dest_dir, str(frame_count)+'.jpg')
                    if (rval is True) and (frame is not None):
                        cv2.imwrite(jpg_filename, frame)
                        frame_count += 1
                    else:
                        break
                    cv2.waitKey(1)
            cap.release()