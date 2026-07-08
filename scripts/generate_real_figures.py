"""Generate real experiment figures from trained model checkpoint.
Run on the local machine where checkpoint and data exist."""
import sys
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Setup path
sys.path.insert(0, '/Users/gaoyucen/poi')

from poi_rec.data.dataset import POISequenceDataset, load_metadata, load_processed_arrays
from poi_rec.models.poi_model import POIRecommendationModel
from torch.utils.data import DataLoader

def get_model_and_data(config_path, checkpoint_path):
    """Load model, config, and data."""
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    
    processed_dir = '/Users/gaoyucen/poi/processed/NYC'
    metadata = load_metadata(processed_dir)
    arrays = load_processed_arrays(processed_dir)
    
    print("Creating model...")
    model = POIRecommendationModel(metadata, arrays, config)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()
    
    print("Loading test data...")
    test_dataset = POISequenceDataset(
        processed_dir, 'test', 
        max_seq_len=config.get('max_seq_len', 20),
        candidate_protocol=config.get('candidate_protocol', 'all_poi')
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    return config, model, test_loader

@torch.no_grad()
def collect_gate_values(model, test_loader, device):
    """Collect gate values by patching fusion module."""
    model = model.to(device)
    model.eval()
    
    gate_values = []
    
    def hook_fn(module, input, output):
        # gate module: nn.Sequential ending with Sigmoid
        # output = sigmoid(linear(...))
        gate_values.append(output.detach().cpu())
    
    # The gate is a nn.Sequential; we register hook on the full Sequential
    handle = model.fusion.gate.register_forward_hook(hook_fn)
    
    n_batches = 0
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(batch, include_priors=False, need_alignment_outputs=False)
        n_batches += 1
        if n_batches >= 50:
            break
    
    handle.remove()
    
    if gate_values:
        all_gates = torch.cat(gate_values, dim=0).numpy()
        print(f"Collected gate values shape: {all_gates.shape}")
        print(f"Mean gate: {all_gates.mean():.4f}, Std: {all_gates.std():.4f}")
        return all_gates
    return None

@torch.no_grad()
def collect_embeddings_for_tsne(model, test_loader, device, n_samples=400):
    """Collect aligned/unaligned topology and semantic embeddings for t-SNE."""
    model = model.to(device)
    model.eval()
    
    topo_before, sem_before = [], []
    topo_after, sem_after = [], []
    n_collected = 0
    
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        poi = batch['poi']
        topo_h = model.topology(poi)
        sem_h = model.semantic(poi)
        z_topo, z_sem = model.alignment(topo_h, sem_h)
        
        batch_size, seq_len = poi.shape
        for b in range(batch_size):
            for t in range(seq_len):
                if batch['attention_mask'][b, t] == 0:
                    continue
                topo_before.append(topo_h[b, t].cpu().numpy())
                sem_before.append(sem_h[b, t].cpu().numpy())
                topo_after.append(z_topo[b, t].cpu().numpy())
                sem_after.append(z_sem[b, t].cpu().numpy())
                n_collected += 1
                if n_collected >= n_samples:
                    break
            if n_collected >= n_samples:
                break
        if n_collected >= n_samples:
            break
    
    return (np.array(topo_before), np.array(sem_before),
            np.array(topo_after), np.array(sem_after))

def plot_gate_distribution(gate_values, save_path):
    """Plot gate distribution figure (Fig 2)."""
    g = gate_values.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].hist(g, bins=50, density=True, alpha=0.7, color='#2E86C1', edgecolor='white', linewidth=0.5)
    axes[0].axvline(g.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={g.mean():.3f}')
    axes[0].set_xlabel('Gate Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Gate Distribution (mean={g.mean():.3f}, std={g.std():.3f})')
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, 1)
    
    n_positions = min(10, gate_values.shape[1])
    pos_means = [gate_values[:, p, :].mean() for p in range(n_positions)]
    pos_stds = [gate_values[:, p, :].std() for p in range(n_positions)]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_positions))
    axes[1].bar(range(1, n_positions+1), pos_means, yerr=pos_stds, 
                capsize=5, color=colors, alpha=0.8, width=0.6)
    axes[1].axhline(g.mean(), color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean={g.mean():.3f}')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Mean Gate')
    axes[1].set_title('Gate by Position')
    axes[1].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gate distribution saved to {save_path}")

def plot_tsne(topo_before, sem_before, topo_after, sem_after, save_path):
    """Plot t-SNE figure (Fig 3)."""
    n_samples = min(200, len(topo_before))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(topo_before), n_samples, replace=False)
    
    combined_before = np.vstack([topo_before[idx], sem_before[idx]])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    embed_before = tsne.fit_transform(combined_before)
    
    combined_after = np.vstack([topo_after[idx], sem_after[idx]])
    embed_after = tsne.fit_transform(combined_after)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    axes[0].scatter(embed_before[:n_samples, 0], embed_before[:n_samples, 1],
                    c='#E74C3C', marker='o', s=20, alpha=0.6, label='Topology')
    axes[0].scatter(embed_before[n_samples:, 0], embed_before[n_samples:, 1],
                    c='#3498DB', marker='x', s=20, alpha=0.6, label='Semantic')
    axes[0].set_title('Before SemAlign', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    axes[1].scatter(embed_after[:n_samples, 0], embed_after[:n_samples, 1],
                    c='#E74C3C', marker='o', s=20, alpha=0.6, label='Topology')
    axes[1].scatter(embed_after[n_samples:, 0], embed_after[n_samples:, 1],
                    c='#3498DB', marker='x', s=20, alpha=0.6, label='Semantic')
    axes[1].set_title('After SemAlign', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE saved to {save_path}")

def main():
    config_path = '/Users/gaoyucen/poi/runs/nyc_mvp/config.json'
    checkpoint_path = '/Users/gaoyucen/poi/runs/nyc_mvp/best.pt'
    output_dir = '/Users/gaoyucen/poi/latex/figure/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    config, model, test_loader = get_model_and_data(config_path, checkpoint_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n=== Collecting gate values ===")
    gate_values = collect_gate_values(model, test_loader, device)
    if gate_values is not None:
        # Print detailed statistics
        g = gate_values.flatten()
        print(f"\n=== Detailed Gate Statistics ===")
        print(f"Overall mean: {g.mean():.4f}")
        print(f"Overall std: {g.std():.4f}")
        print(f"Overall var: {g.var():.4f}")
        n_positions = min(10, gate_values.shape[1])
        for p in range(n_positions):
            pm = gate_values[:, p, :].mean()
            ps = gate_values[:, p, :].std()
            print(f"  Position {p+1}: mean={pm:.4f}, std={ps:.4f}")
        # Position group statistics
        pos_groups = {'Front (1-3)': (0, 3), 'Middle (4-6)': (3, 6), 'Rear (7-10)': (6, 10)}
        for name, (start, end) in pos_groups.items():
            if start < n_positions:
                end_actual = min(end, n_positions)
                group_data = gate_values[:, start:end_actual, :].flatten()
                print(f"  {name}: mean={group_data.mean():.4f}, std={group_data.std():.4f}")
        
        plot_gate_distribution(gate_values, os.path.join(output_dir, 'gate_distribution.png'))
    
    print("\n=== Collecting embeddings for t-SNE ===")
    topo_before, sem_before, topo_after, sem_after = collect_embeddings_for_tsne(
        model, test_loader, device, n_samples=400
    )
    plot_tsne(topo_before, sem_before, topo_after, sem_after,
              os.path.join(output_dir, 'embeddings_tsne.png'))
    
    print("\nAll real experiment figures generated successfully!")

if __name__ == '__main__':
    main()