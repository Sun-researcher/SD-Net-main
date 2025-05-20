from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import argparse
from retinaface.pre_trained_models import get_model
import torch
import random
import dlib
from imutils import face_utils

def facecrop(model, org_path, save_path, face_predictor, face_detector, num_frames=10):
    cap_org = cv2.VideoCapture(org_path)
    cap_mask = cv2.VideoCapture(org_path.replace('/videos/','/masks/'))
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_org.get(cv2.CAP_PROP_FPS)
    max_start_frame = min(int(5 * fps), frame_count_org - 1)

    start_frame = random.randint(0, max_start_frame)

    period_frames = int(0.1 * fps)

    frame_idxs = [start_frame + i * period_frames for i in range(num_frames) if (start_frame + i * period_frames) < frame_count_org]

    frames_org = []
    frames_mask = []

    for cnt_frame in range(frame_count_org):
        ret_org, frame_org = cap_org.read()
        ret_mask, frame_mask = cap_mask.read()
        if not ret_org:
            tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(org_path)))
            continue

        if cnt_frame in frame_idxs:
            frames_org.append(frame_org)
            frames_mask.append(frame_mask)

        if len(frames_org) == num_frames:
            break

    while len(frames_org) < num_frames:
        frames_org.append(frames_org[-1])
        frames_mask.append(frames_mask[-1])

    for idx, frame_org in enumerate(frames_org):
        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
        faces_retina = model.predict_jsons(frame)
        faces_dlib = face_detector(frame, 1)

        try:
            if len(faces_retina) == 0:
                tqdm.write('No faces in {}:{}'.format(idx, os.path.basename(org_path)))
                continue
            landmarks_retina = []
            size_list_retina = []
            for face_idx in range(len(faces_retina)):
                x0, y0, x1, y1 = faces_retina[face_idx]['bbox']
                landmark_retina = np.array([[x0, y0], [x1, y1]] + faces_retina[face_idx]['landmarks'])
                face_s_retina = (x1 - x0) * (y1 - y0)
                size_list_retina.append(face_s_retina)
                landmarks_retina.append(landmark_retina)
        except Exception as e:
            print(f'Error in {idx}:{org_path}')
            print(e)
            continue
        landmarks_retina = np.concatenate(landmarks_retina).reshape((len(size_list_retina),) + landmark_retina.shape)
        landmarks_retina = landmarks_retina[np.argsort(np.array(size_list_retina))[::-1]]  # 按照面部大小排序

        if len(faces_dlib) == 0:
            tqdm.write('No faces in {}:{}'.format(idx, os.path.basename(org_path)))
            continue
        landmarks_dlib = []
        size_list_dlib = []
        for face_idx in range(len(faces_dlib)):
            landmark_dlib = face_predictor(frame, faces_dlib[face_idx])
            landmark_dlib = face_utils.shape_to_np(landmark_dlib)
            x0, y0 = landmark_dlib[:, 0].min(), landmark_dlib[:, 1].min()
            x1, y1 = landmark_dlib[:, 0].max(), landmark_dlib[:, 1].max()
            face_s = (x1 - x0) * (y1 - y0)
            size_list_dlib.append(face_s)
            landmarks_dlib.append(landmark_dlib)
        landmarks_dlib = np.concatenate(landmarks_dlib).reshape((len(size_list_dlib),) + landmark_dlib.shape)
        landmarks_dlib = landmarks_dlib[np.argsort(np.array(size_list_dlib))[::-1]]

        save_path_ = save_path + 'frames/' + os.path.basename(org_path).replace('.mp4', '/')
        os.makedirs(save_path_, exist_ok=True)
        image_path = save_path_ + str(idx).zfill(3) + '.png'
        land_path = save_path_ + str(idx).zfill(3)

        land_path_retina = land_path.replace('/frames', '/retina')
        os.makedirs(os.path.dirname(land_path_retina), exist_ok=True)
        np.save(land_path_retina, landmarks_retina)

        land_path_dlib = land_path.replace('/frames', '/landmarks')
        os.makedirs(os.path.dirname(land_path_dlib), exist_ok=True)
        np.save(land_path_dlib, landmarks_dlib)

        if not os.path.isfile(image_path):
            cv2.imwrite(image_path, frame_org)
        if not os.path.isfile(image_path.replace('/videos/','/masks/')):
            cv2.imwrite(image_path.replace('/videos/','/masks/'), frames_mask[idx])


    cap_org.release()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset',
                        choices=['DeepFakeDetection_original', 'DeepFakeDetection', 'FaceShifter', 'Face2Face',
                                 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Original', 'manipulate', 'Celeb-real', 'Celeb-synthesis',
                                 'YouTube-real', 'DFDC', 'DFDCP', 'Celeb-DF-v1'])
    parser.add_argument('-c', dest='comp', choices=['raw', 'c23', 'c40'], default='c23')
    parser.add_argument('-n', dest='num_frames', type=int, default=32)
    args = parser.parse_args()
    dataset_list = []
    if args.dataset == 'Original':
        dataset_list = ['data/FaceForensics++/original_sequences/youtube/{}/'.format(args.comp)]
    elif args.dataset == 'DeepFakeDetection_original':
        dataset_list = ['data/FaceForensics++/original_sequences/actors/{}/'.format(args.comp)]
    elif args.dataset in ['DeepFakeDetection', 'FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
        dataset_list = ['data/FaceForensics++/manipulated_sequences/{}/{}/'.format(args.dataset, args.comp)]
    elif args.dataset == 'manipulate':
        dataset_list = ['data/FaceForensics++/manipulated_sequences/{}/{}/'.format(i, args.comp) for i in
                        ['Face2Face','Deepfakes', 'FaceSwap', 'NeuralTextures']]
    elif args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
        dataset_list = ['data/Celeb-DF-v2/{}/'.format(args.dataset)]
    elif args.dataset == 'Celeb-DF-v1':
        dataset_list = ['data/Celeb-DF-v1/{}/'.format(args.dataset)]
    elif args.dataset in ['DFDC']:
        dataset_list = ['data/{}/'.format(args.dataset)]
    else:
        raise NotImplementedError

    device = torch.device('cuda')

    model = get_model("resnet50_2020-07-20", max_size=2048, device=str(device))
    model.eval()

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'src/preprocess/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    for dataset_path in dataset_list:
        movies_path = dataset_path + 'videos/'

        movies_path_list = sorted(glob(movies_path + '*.mp4'))
        n_sample = len(movies_path_list)
        print("{} : videos are exist in {}".format(n_sample, dataset_path))

        for i in tqdm(range(n_sample)):
            folder_path = movies_path_list[i].replace('videos/', 'frames/').replace('.mp4', '/')
            facecrop(model, movies_path_list[i], save_path=dataset_path, face_predictor=face_predictor, face_detector=face_detector, num_frames=args.num_frames)