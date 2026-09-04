import metric

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test images
size = 32
np.random.seed(42)
dataset = np.random.uniform(-1, 1, (42, 3, size, size))
labels = np.random.randint(0, 1000, (dataset.shape[0],))
ind_a = np.random.randint(0, dataset.shape[0], (2,))
ind_b = np.random.randint(0, dataset.shape[0], (3,))
img_a = dataset[ind_a]
img_b = dataset[ind_b]
label_a = labels[ind_a]
label_b = labels[ind_b]

print('\n\nTest LPIPS')
lpips_metric = metric.LPIPSMetric(device)
print("Output shape (should be 2 x 3): ", lpips_metric.compute_similarity(img_a, img_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {lpips_metric.compute_similarity(img_a, img_a)}")
print(f"Same result? (i.e. test for deterministic): {metric.LPIPSMetric(device).compute_similarity(img_a, img_a)}")
print("\nPrecompute:")
lpips_metric.precompute(dataset, batch_size=8)
print("Same result after precompute? (i.e. test for deterministic):",
      lpips_metric.precomputed_similarity(ind_a, ind_a))
print("Same result after reload? (i.e. test for deterministic):",
      metric.PrecomputedMetric(lpips_metric.similarity_matrix).precomputed_similarity(ind_a, ind_a))

print('\n\nTest DreamSim')
dreamsim_metric = metric.DreamSimMetric(device)
print("Output shape (should be 2 x 3): ", dreamsim_metric.compute_similarity(img_a, img_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {dreamsim_metric.compute_similarity(img_a, img_a)}")
print(f"Same result? (i.e. test for deterministic): {metric.DreamSimMetric(device).compute_similarity(img_a, img_a)}")
print("\nPrecompute:")
dreamsim_metric.precompute(dataset, batch_size=8)
print("Output shape (should be 2 x 3): ", dreamsim_metric.precomputed_similarity(ind_a, ind_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {dreamsim_metric.precomputed_similarity(ind_a, ind_a)}")
print("Same result after reload? (i.e. test for deterministic):",
      metric.PrecomputedMetric(dreamsim_metric.similarity_matrix).precomputed_similarity(ind_a, ind_a))

print('\n\nTest SSIM Metric')
ssim_metric = metric.SSIMMetric()
print("Output shape (should be 2 x 3): ", ssim_metric.compute_similarity(img_a, img_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {ssim_metric.compute_similarity(img_a, img_a)}")
print(f"Same result? (i.e. test for deterministic): {metric.SSIMMetric().compute_similarity(img_a, img_a)}")
print("\nPrecompute:")
ssim_metric.precompute(dataset, batch_size=8)
print("Same result after precompute? (i.e. test for deterministic):",
      ssim_metric.precomputed_similarity(ind_a, ind_a))
print("Same result after reload? (i.e. test for deterministic):",
      metric.PrecomputedMetric(ssim_metric.similarity_matrix).precomputed_similarity(ind_a, ind_a))

print("\n\nTest MSE Metric")
mse_metric = metric.MSEMetric()
print("Output shape (should be 2 x 3): ", mse_metric.compute_similarity(img_a, img_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {mse_metric.compute_similarity(img_a, img_a)}")
print("\nPrecompute:")
mse_metric.precompute(dataset, batch_size=8)
print("Same result after precompute? (i.e. test for deterministic):",
      mse_metric.precomputed_similarity(ind_a, ind_a))
print("Same result after reload? (i.e. test for deterministic):",
      metric.PrecomputedMetric(mse_metric.similarity_matrix).precomputed_similarity(ind_a, ind_a))

print("\n\nTest CosineMetric")
cosine_metric = metric.CosineMetric()
print("Output shape (should be 2 x 3): ", cosine_metric.compute_similarity(img_a, img_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {cosine_metric.compute_similarity(img_a, img_a)}")
print("\nPrecompute:")
cosine_metric.precompute(dataset, batch_size=8)
print("Same result after precompute? (i.e. test for deterministic):",
      cosine_metric.precomputed_similarity(ind_a, ind_a))
print("Same result after reload? (i.e. test for deterministic):",
      metric.PrecomputedMetric(cosine_metric.similarity_matrix).precomputed_similarity(ind_a, ind_a))

print("\n\nTest LabelMetric")
label_metric = metric.LabelMetric()
print("Output shape (should be 2 x 3): ", label_metric.compute_similarity(label_a, label_b).shape)
print(f"Perceptual similarity (should be symmetric and 0 on diagonal): {label_metric.compute_similarity(label_a, label_a)}")
print("\nPrecompute:")
label_metric.precompute(labels, batch_size=8)
print("Same result after precompute? (i.e. test for deterministic):",
      label_metric.precomputed_similarity(ind_a, ind_a))
print("Same result after reload? (i.e. test for deterministic):",
      metric.PrecomputedMetric(label_metric.similarity_matrix).precomputed_similarity(ind_a, ind_a))