import metric

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test images
size = 32
torch.manual_seed(97823)
np.random.seed(97823)
img_a = np.random.uniform(-1, 1, (2, 3, size, size))
img_b = np.random.uniform(-1, 1, (3, 3, size, size))
label_a = torch.randint(0, 1000, img_a.shape[:1])
label_b = torch.randint(0, 1000, img_b.shape[:1])
dataset = np.random.uniform(-1, 1, (42, 3, size, size))
ind_a = torch.randint(0, 42, (2,))
ind_b = torch.randint(0, 42, (3,))

print('\n')
print('Test LPIPS')
lpips_metric = metric.LPIPSMetric(device)
print("Output shape (should be 2 x 3): ", lpips_metric.compute_similarity(img_a, img_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {lpips_metric.compute_similarity(img_a, img_a)}")
print(f"Same result? (i.e. test for deterministic): {metric.LPIPSMetric(device).compute_similarity(img_a, img_a)}")

print('\n')
print('Test DreamSim')
dreamsim_metric = metric.DreamSimMetric(device)
print("Output shape (should be 2 x 3): ", dreamsim_metric.compute_similarity(img_a, img_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {dreamsim_metric.compute_similarity(img_a, img_a)}")
print(f"Same result? (i.e. test for deterministic): {metric.DreamSimMetric(device).compute_similarity(img_a, img_a)}")
print("\nPrecompute:")
dreamsim_metric.precompute(dataset, batch_size=8)
print("Output shape (should be 2 x 3): ", dreamsim_metric.precomputed_similarity(ind_a, ind_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {dreamsim_metric.precomputed_similarity(ind_a, ind_a)}")

print('\n')
print('Test SSIM Metric')
ssim_metric = metric.SSIMMetric()
print("Output shape (should be 2 x 3): ", ssim_metric.compute_similarity(img_a, img_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {ssim_metric.compute_similarity(img_a, img_a)}")
print(f"Same result? (i.e. test for deterministic): {metric.SSIMMetric().compute_similarity(img_a, img_a)}")

print('\n')
print("Test MSE Metric")
print("Output shape (should be 2 x 3): ", metric.MSEMetric().compute_similarity(img_a, img_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {metric.MSEMetric().compute_similarity(img_a, img_a)}")

print('\n')
print("Test CosineMetric")
print("Output shape (should be 2 x 3): ", metric.CosineMetric().compute_similarity(img_a, img_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {metric.CosineMetric().compute_similarity(img_a, img_a)}")

print('\n')
print("Test LabelMetric")
print(label_a, label_b)
print("Output shape (should be 2 x 3): ", metric.LabelMetric().compute_similarity(label_a, label_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {metric.LabelMetric().compute_similarity(label_a, label_a)}")
