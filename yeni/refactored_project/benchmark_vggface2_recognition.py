import os
import time
import glob
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d

# InsightFace
import cv2
import insightface
from insightface.app import FaceAnalysis

class VGGFace2Benchmark:
    def __init__(self, dataset_root, model_name='buffalo_s', ctx_id=0, n_identities=100, max_images=20):
        """
        Benchmark class for Face Recognition models on VGGFace2 dataset.
        
        Args:
            dataset_root (str): Path to VGGFace2 dataset (identity_id/image.jpg)
            model_name (str): InsightFace model provider name (e.g., 'buffalo_s')
            ctx_id (int): CPU/GPU context (negative for CPU)
            n_identities (int): Number of random identities to select for test
            max_images (int): Max images per identity
        """
        self.dataset_root = dataset_root
        self.model_name = model_name
        self.n_identities = n_identities
        self.max_images = max_images
        
        # Initialize InsightFace
        print(f"[INFO] Initializing InsightFace model: {model_name}")
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider']) # Default to CPU for stability
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        self.results = {}
        self.embeddings_cache = {} # {path: embedding}
        self.identity_map = {} # {identity_id: [paths]}

    def load_dataset(self):
        """Scan dataset directory and pick N random identities."""
        print(f"[INFO] Scanning dataset at {self.dataset_root}")
        
        # Check if identities are in subfolders like 'train' or 'val' (common in Kaggle)
        scan_path = self.dataset_root
        if not any(os.path.isdir(os.path.join(scan_path, d)) and d.startswith('n') for d in os.listdir(scan_path)):
            for sub in ['train', 'val']:
                test_path = os.path.join(self.dataset_root, sub)
                if os.path.exists(test_path) and any(d.startswith('n') for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))):
                    scan_path = test_path
                    print(f"[INFO] Found identities in subfolder: {sub}")
                    break

        all_identities = [d for d in os.listdir(scan_path) if os.path.isdir(os.path.join(scan_path, d))]
        
        if len(all_identities) < self.n_identities:
            print(f"[WARNING] Requested {self.n_identities} identities, but only {len(all_identities)} found.")
            selected_ids = all_identities
        else:
            selected_ids = random.sample(all_identities, self.n_identities)
            
        for identity in selected_ids:
            img_paths = glob.glob(os.path.join(scan_path, identity, "*.jpg"))
            if len(img_paths) >= 2: # Need at least 2 images for verification/identification split
                # Limit images per identity
                if len(img_paths) > self.max_images:
                    img_paths = random.sample(img_paths, self.max_images)
                self.identity_map[identity] = img_paths
        
        print(f"[INFO] Selected {len(self.identity_map)} identities with max {self.max_images} images each.")

    def get_embedding(self, img_path):
        """Extract 512-d embedding from a single image."""
        if img_path in self.embeddings_cache:
            return self.embeddings_cache[img_path]
        
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        start_time = time.time()
        faces = self.app.get(img)
        extraction_time = (time.time() - start_time) * 1000 # ms
        
        if not faces:
            return None
        
        # Take the largest face if multiple detected
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        embedding = face.normed_embedding
        
        return embedding, extraction_time

    def run_benchmark(self):
        """Main execution flow."""
        self.load_dataset()
        
        all_image_paths = []
        for paths in self.identity_map.values():
            all_image_paths.extend(paths)
            
        print(f"[INFO] Extracting embeddings for {len(all_image_paths)} images...")
        times = []
        valid_embeddings = {} # path -> embedding
        
        for path in tqdm(all_image_paths):
            res = self.get_embedding(path)
            if res:
                emb, t = res
                valid_embeddings[path] = emb
                times.append(t)
        
        avg_extraction_time = np.mean(times) if times else 0
        print(f"[SUCCESS] Avg Extraction Time: {avg_extraction_time:.2f} ms")
        
        # Filter identity map to only include images with valid embeddings
        filtered_identity_map = {}
        for id_val, paths in self.identity_map.items():
            valid_paths = [p for p in paths if p in valid_embeddings]
            if len(valid_paths) >= 2:
                filtered_identity_map[id_val] = valid_paths
        
        # 1. Evaluation: Verification (1:1)
        print("[INFO] Evaluating Verification (1:1)...")
        verif_results = self.evaluate_verification(filtered_identity_map, valid_embeddings)
        
        # 2. Evaluation: Identification (1:N)
        print("[INFO] Evaluating Identification (1:N)...")
        ident_results = self.evaluate_identification(filtered_identity_map, valid_embeddings)
        
        # Combine Results
        final_results = {
            "Model": self.model_name,
            "Avg_Extraction_Time_ms": avg_extraction_time,
            "EER": verif_results['EER'],
            "TAR@FAR=1e-3": verif_results['TAR@1e-3'],
            "Rank-1_Accuracy": ident_results['Rank-1'],
            "Rank-5_Accuracy": ident_results['Rank-5'],
            "Total_Identities": len(filtered_identity_map),
            "Total_Images": len(valid_embeddings)
        }
        
        # Save to CSV
        df = pd.DataFrame([final_results])
        df.to_csv("Phase2_VGGFace2_Results.csv", index=False)
        print("\n[RESULT TABLE]")
        print(df.to_string())
        
        # Plot ROC
        self.plot_roc(verif_results['fpr'], verif_results['tpr'], verif_results['EER'])
        
    def evaluate_verification(self, identity_map, embeddings):
        """Calculate FAR, FRR, EER, and TAR@FAR=1e-3."""
        matches = []
        mismatches = []
        
        ids = list(identity_map.keys())
        
        # Positive pairs (Matches)
        for identity in ids:
            paths = identity_map[identity]
            if len(paths) >= 2:
                # Create random pairs within identity
                # For simplicity, we can do sequential pairs or random
                random.shuffle(paths)
                for i in range(0, len(paths)-1, 2):
                    sim = 1 - cosine(embeddings[paths[i]], embeddings[paths[i+1]])
                    matches.append(sim)
        
        # Negative pairs (Mismatches)
        # Select random images from different identities
        n_neg = len(matches)
        for _ in range(n_neg):
            id1, id2 = random.sample(ids, 2)
            p1 = random.choice(identity_map[id1])
            p2 = random.choice(identity_map[id2])
            sim = 1 - cosine(embeddings[p1], embeddings[p2])
            mismatches.append(sim)
            
        y_true = [1] * len(matches) + [0] * len(mismatches)
        y_score = matches + mismatches
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # Calculate EER
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        # Calculate TAR @ FAR = 1e-3
        # Interpolate TPR at target FPR
        f = interp1d(fpr, tpr)
        try:
            tar_at_far_1e3 = f(0.001)
        except:
            # If 0.001 is out of bounds or not enough data, find closest
            idx = np.argmin(np.abs(fpr - 0.001))
            tar_at_far_1e3 = tpr[idx]
            
        return {
            'EER': eer,
            'TAR@1e-3': tar_at_far_1e3,
            'fpr': fpr,
            'tpr': tpr
        }

    def evaluate_identification(self, identity_map, embeddings):
        """Calculate Rank-1 and Rank-5 Accuracy."""
        gallery_paths = []
        gallery_ids = []
        probe_paths = []
        probe_ids = []
        
        for identity, paths in identity_map.items():
            # Split: first image to gallery, rest to probe
            gallery_paths.append(paths[0])
            gallery_ids.append(identity)
            probe_paths.extend(paths[1:])
            probe_ids.extend([identity] * (len(paths) - 1))
            
        gallery_embs = np.array([embeddings[p] for p in gallery_paths])
        
        rank1_hits = 0
        rank5_hits = 0
        total_probes = len(probe_paths)
        
        for i, probe_path in enumerate(probe_paths):
            probe_emb = embeddings[probe_path]
            true_id = probe_ids[i]
            
            # Compute similarities with all gallery
            # Using dot product for speed as they are normed
            similarities = np.dot(gallery_embs, probe_emb)
            
            # Get indices of top 5
            top_indices = np.argsort(similarities)[-5:][::-1]
            top_ids = [gallery_ids[idx] for idx in top_indices]
            
            if true_id == top_ids[0]:
                rank1_hits += 1
            if true_id in top_ids:
                rank5_hits += 1
                
        return {
            'Rank-1': rank1_hits / total_probes if total_probes > 0 else 0,
            'Rank-5': rank5_hits / total_probes if total_probes > 0 else 0
        }

    def plot_roc(self, fpr, tpr, eer):
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (EER = {eer:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Mark EER point
        plt.plot(eer, 1-eer, 'ro', label=f'EER Point ({eer:.2f}, {1-eer:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('True Acceptance Rate (TAR)')
        plt.title(f'Phase 2: ROC Curve - {self.model_name} on VGGFace2')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig("Phase2_VGGFace2_ROC.png")
        print(f"[INFO] ROC curve saved to Phase2_VGGFace2_ROC.png")
        plt.show()

if __name__ == "__main__":
    # Check if we have a downloaded path from our scratch script
    default_dataset = "/Users/kerem/Documents/GitHub/thesis-project-image/yeni/datasets/vggface2_test"
    scratch_path_file = "/Users/kerem/Documents/GitHub/thesis-project-image/yeni/scratch/vggface2_path.txt"
    if os.path.exists(scratch_path_file):
        with open(scratch_path_file, "r") as f:
            downloaded_path = f.read().strip()
            if os.path.exists(downloaded_path):
                default_dataset = downloaded_path

    parser = argparse.ArgumentParser(description="Phase 2: InsightFace Recognition Benchmark on VGGFace2")
    parser.add_argument("--dataset", type=str, 
                        default=default_dataset,
                        help="Path to VGGFace2 dataset root (identity_id/image.jpg)")
    parser.add_argument("--models", nargs="+", default=["buffalo_s"],
                        help="List of InsightFace models to benchmark")
    parser.add_argument("--identities", type=int, default=100,
                        help="Number of random identities to select for the test")
    parser.add_argument("--max-images", type=int, default=20,
                        help="Maximum number of images to process per identity to save memory/time")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"\n[ERROR] Dataset path not found: {args.dataset}")
        print("Please provide the correct path using --dataset parameter.")
        print(f"Example: python benchmark_vggface2_recognition.py --dataset /path/to/vggface2")
        
        # If the user has a 'datasets' folder in the project root, mention it
        project_root = os.path.dirname(os.path.abspath(__file__))
        suggested_path = os.path.join(os.path.dirname(project_root), "datasets")
        if os.path.exists(suggested_path):
            print(f"\n[TIP] I found a 'datasets' folder at: {suggested_path}")
            print(f"Try: python benchmark_vggface2_recognition.py --dataset {suggested_path}")
    else:
        all_model_results = []
        
        for model_name in args.models:
            print(f"\n{'='*50}")
            print(f"RUNNING BENCHMARK FOR: {model_name}")
            print(f"{'='*50}")
            
            try:
                benchmark = VGGFace2Benchmark(
                    dataset_root=args.dataset,
                    model_name=model_name,
                    n_identities=args.identities,
                    max_images=args.max_images
                )
                benchmark.run_benchmark()
                
                # Load the result just saved to aggregate
                df_temp = pd.read_csv("Phase2_VGGFace2_Results.csv")
                all_model_results.append(df_temp)
                
                # Rename the plot to avoid overwriting if multiple models
                if len(args.models) > 1:
                    plot_name = f"Phase2_VGGFace2_ROC_{model_name}.png"
                    if os.path.exists("Phase2_VGGFace2_ROC.png"):
                        os.rename("Phase2_VGGFace2_ROC.png", plot_name)
                    
            except Exception as e:
                print(f"[CRITICAL ERROR] Failed to benchmark {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        if all_model_results:
            final_df = pd.concat(all_model_results, ignore_index=True)
            final_df.to_csv("Phase2_VGGFace2_Results.csv", index=False)
            print("\n[FINAL CONSOLIDATED RESULTS]")
            print(final_df.to_string())
