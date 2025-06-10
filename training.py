import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
import random
from collections import defaultdict

class Trainer:
    def __init__(self, app, config):
        self.app = app
        self.config = config

    def augment_image(self, img):
        angle = random.uniform(-10, 10)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        
        alpha = random.uniform(0.9, 1.1)
        img = cv2.convertScaleAbs(img, alpha=alpha)
        
        scale = random.uniform(0.8, 1.2)
        img = cv2.resize(img, None, fx=scale, fy=scale)
     #motion blur

        if random.random()<0.4:
            kernel_size=random.choice([3,5,7])
            kernel=np.zeros([kernel_size,kernel_size])
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel=kernel/kernel_size
            img=cv2.filter2D(img, -1,kernel=kernel)
        if random.random()<0.5:
            noise=np.random.normal(0,random.uniform(5,15),img.shape)
            img=np.clip(img.astype(np.float32)+noise,0,255).astype(np.uint8)
        
        if random.random() < 0.3:  # 30% chance
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            # Small random perspective changes
            offset = random.uniform(5, 15)
            pts2 = np.float32([
                [random.uniform(0, offset), random.uniform(0, offset)],
                [w - random.uniform(0, offset), random.uniform(0, offset)],
                [random.uniform(0, offset), h - random.uniform(0, offset)],
                [w - random.uniform(0, offset), h - random.uniform(0, offset)]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M, (w, h))        
        if random.random() < 0.4:  # 40% chance
            quality = random.randint(60, 85)  # JPEG compression quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(encimg, 1)
        if random.random() < 0.5:  # 50% chance
            img = cv2.flip(img, 1)
        if random.random() < 0.2:  # 20% chance
            img = img.astype(np.float32)
            # Randomly adjust individual color channels
            for c in range(3):
                img[:, :, c] *= random.uniform(0.9, 1.1)
            img = np.clip(img, 0, 255).astype(np.uint8)
            

        
        return img

    def train_model(self):
        print("Starting optimized training...")
        label_embeddings = defaultdict(list)
        
        for person in os.listdir(self.config["training_data_path"]):
            person_path = os.path.join(self.config["training_data_path"], person)
            if not os.path.isdir(person_path):
                continue
            
            print(f"Processing training images for {person}")
            image_count = 0
            face_count = 0
            
            for img_file in os.listdir(person_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                image_count += 1
                
                for scale in self.config["detection_scales"]:
                    scaled_img = cv2.resize(img, None, fx=scale, fy=scale)
                    faces = self.app.get(scaled_img)
                    for face in faces:
                        if face.det_score < self.config["recognition_confidence"]:
                            continue
                        embedding = normalize(face.embedding.reshape(1, -1)).flatten()
                        label_embeddings[person].append(embedding)
                        face_count += 1
                
                for _ in range(self.config["augmentation_factor"]):
                    aug_img = self.augment_image(img)
                    faces = self.app.get(aug_img)
                    for face in faces:
                        if face.det_score < self.config["recognition_confidence"]:
                            continue
                        embedding = normalize(face.embedding.reshape(1, -1)).flatten()
                        label_embeddings[person].append(embedding)
                        face_count += 1
            
            print(f"  - Processed {image_count} images, found {face_count} faces for {person}")
        
        known_labels = []
        known_embeddings = []
        
        for label, embeddings in label_embeddings.items():
            if len(embeddings) < 3:
                print(f"Skipping {label} - too few valid embeddings")
                continue
            
            embeddings = np.array(embeddings)
            clustering = DBSCAN(eps=0.3, min_samples=3, metric='cosine').fit(embeddings)
            labels = clustering.labels_
            valid_embeddings = embeddings[labels != -1]
            
            if len(valid_embeddings) < 3:
                print(f"Skipping {label} - too few valid embeddings after outlier removal")
                continue
            
            unique_labels = set(labels) - {-1}
            if len(unique_labels) > self.config["max_clusters_per_person"]:
                cluster_sizes = [(lbl, np.sum(labels == lbl)) for lbl in unique_labels]
                cluster_sizes.sort(key=lambda x: x[1], reverse=True)
                selected_labels = [lbl for lbl, _ in cluster_sizes[:self.config["max_clusters_per_person"]]]
                valid_embeddings = np.concatenate([embeddings[labels == lbl] for lbl in selected_labels])
            
            for cluster_label in set(labels) - {-1}:
                cluster_embeddings = embeddings[labels == cluster_label]
                if len(cluster_embeddings) >= 3:
                    centroid = np.mean(cluster_embeddings, axis=0)
                    normalized_centroid = normalize(centroid.reshape(1, -1)).flatten()
                    known_embeddings.append(normalized_centroid)
                    known_labels.append(label)
        
        return known_embeddings, known_labels