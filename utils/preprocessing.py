from facenet_pytorch import MTCNN
import cv2
import os
import pandas as pd
import numpy as np
import tqdm

detector = MTCNN()
img_size = 256

def get_frame(cap, sec, current_count, name, label, dataset, save_dir):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames,image = cap.read()

    dataset_temp = dataset
    if dataset == 'train':
        # Put 10% into the validation set
        if np.random.randint(0,10) == 0:
            dataset_temp = 'validation'

    if hasFrames:
        faces = detector.detect(image)
        if type(faces[0]) == np.ndarray:
            for i in range(len(faces[0])):
                try:
                    accuracy = faces[1][i]
                    if(accuracy > 0.99):
                        x1 = int(faces[0][i][0])
                        y1 = int(faces[0][i][1])
                        x2 = int(faces[0][i][2])
                        y2 = int(faces[0][i][3])

                        # Add 20 % padding to each side of the face
                        padding_x = int((x2 - x1) * 0.2)
                        padding_y = int((y2 - y1) * 0.2)
                        
                        x1 = x1 - padding_x
                        y1 = y1 - padding_y
                        x2 = x2 + padding_x
                        y2 = y2 + padding_y

                        # Ensure we're inside 
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(image.shape[1], x2)
                        y2 = min(image.shape[0], y2)

                        current_count += 1
                        face_image = image[y1-padding_y:y2+padding_y, x1-padding_x:x2+padding_x]

                        # Resize image
                        face_image = cv2.resize(face_image, (img_size, img_size))

                        # Save image 
                        cv2.imwrite(f"./{save_dir}/{dataset_temp}/{label}/{name}{current_count}.jpg", face_image)
                except:
                    # Silent catch
                    pass
    return hasFrames, current_count

def find_faces_in_videos(video_directory, dataset, video_name, label, save_dir):
    cap = cv2.VideoCapture(f'{video_directory}{video_name}')

    sec = 0
    frame_rate = 0.5 # Clips is about 10 seconds giving us 20 frame for each video
    count = 0
    name = video_name.replace(".mp4", "")
    success, count = get_frame(cap, sec, 0, name, label, dataset, save_dir)
    while success:
        success, count = get_frame(cap, sec, count, name, label, dataset, save_dir)
        sec += frame_rate


def preprocess_videos(datasets, save_dir = "data", video_dir="videos", skip_sampling=False):
    classes = ["REAL", "FAKE"]
    datasets_names = ["train", "test", "validation"]
    # Ensure all folders is created
    for dataset in datasets_names:
        for class_name in classes:
            if not os.path.exists(f"./{save_dir}/{dataset}/{class_name}"):
                os.makedirs(f"./{save_dir}/{dataset}/{class_name}")

    for dataset in datasets:
        folders = datasets[dataset]
        print(f"Processing {dataset} datasets")
        # Ensure directory exists
        for dir in folders:
            print(f"\t Processing {dir}")
            path = f"./{video_dir}/{dir}/"

            # Only keep .mp4 files
            data = pd.read_json(f"./{video_dir}/{dir}/metadata.json").T

            # Find number of real videos
            real_videos = data[data["label"] == "REAL"]
            fake_videos = data[data["label"] == "FAKE"]

            # Sample n from fake videos
            sampled_fake_videos = fake_videos.index
            if not skip_sampling:
                sampled_fake_videos = np.random.choice(fake_videos.index, size=len(real_videos), replace=False)

            # Combine real and fake videos
            videos = np.concatenate((real_videos.index, sampled_fake_videos))

            # Get the rows in data
            videos = data.loc[videos]

            for video in tqdm.tqdm(videos.iterrows()):
                label = video[1][0]
                filename = video[0]
                find_faces_in_videos(path, dataset, filename, label, save_dir)