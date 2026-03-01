"""
Evaluate In-Context Learning performance with varying number of demonstrations.

This script measures:
- MSE loss improvement with more demos
- Per-track performance
- Trajectory quality metrics
"""

import torch
import numpy as np
import h5py
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_icl.models import ICLTransformerPolicy
from drone_icl.dataset_icl import ICLRacingDataset


def compute_metrics(pred_actions, true_actions, mask):
    """
    Compute evaluation metrics.

    Args:
        pred_actions: (T, 4) predicted actions
        true_actions: (T, 4) ground truth actions
        mask: (T,) boolean mask for valid timesteps

    Returns:
        Dictionary of metrics
    """
    # Apply mask
    pred = pred_actions[mask]
    true = true_actions[mask]

    # MSE Loss
    mse = torch.mean((pred - true) ** 2).item()

    # Per-action MSE
    mse_thrust = torch.mean((pred[:, 0] - true[:, 0]) ** 2).item()
    mse_roll = torch.mean((pred[:, 1] - true[:, 1]) ** 2).item()
    mse_pitch = torch.mean((pred[:, 2] - true[:, 2]) ** 2).item()
    mse_yaw = torch.mean((pred[:, 3] - true[:, 3]) ** 2).item()

    # Smoothness (variance of action derivatives)
    pred_diff = torch.diff(pred, dim=0)
    smoothness = torch.mean(torch.var(pred_diff, dim=0)).item()

    # Max error
    max_error = torch.max(torch.abs(pred - true)).item()

    return {
        'mse': mse,
        'mse_thrust': mse_thrust,
        'mse_roll': mse_roll,
        'mse_pitch': mse_pitch,
        'mse_yaw': mse_yaw,
        'smoothness': smoothness,
        'max_error': max_error,
        'num_steps': int(mask.sum())
    }


def evaluate_with_n_demos(model, dataset, episode_idx, num_demos, device='cpu'):
    """
    Evaluate single episode with N demonstrations.

    Args:
        model: ICL transformer model
        dataset: ICLRacingDataset
        episode_idx: Index of test episode
        num_demos: Number of demonstrations to use (0 for zero-shot)
        device: Device to run on

    Returns:
        Metrics dictionary
    """
    model.eval()

    # Get episode and track info
    target_ep_name = dataset.episodes[episode_idx]
    target_track = dataset.episode_to_track[target_ep_name]

    # Get demo episodes from same track
    same_track_episodes = [ep for ep in dataset.track_to_episodes[target_track]
                          if ep != target_ep_name]

    if num_demos == 0:
        # Zero-shot: use random demos or empty context
        demo_ep_names = []
    elif len(same_track_episodes) >= num_demos:
        demo_ep_names = np.random.choice(same_track_episodes, size=num_demos, replace=False).tolist()
    else:
        # Not enough demos, use all available
        demo_ep_names = same_track_episodes
        while len(demo_ep_names) < num_demos:
            demo_ep_names.append(np.random.choice(same_track_episodes))

    # Load target
    target_data = dataset._load_episode(target_ep_name)

    # Load demos
    if num_demos > 0:
        demo_data = [dataset._load_episode(name) for name in demo_ep_names]
        demo_states = torch.stack([d['states'] for d in demo_data]).unsqueeze(0)  # (1, N, T, D)
        demo_actions = torch.stack([d['actions'] for d in demo_data]).unsqueeze(0)
        demo_masks = torch.stack([d['mask'] for d in demo_data]).unsqueeze(0)
    else:
        # Zero-shot: empty context
        demo_states = torch.zeros(1, 1, dataset.max_seq_len, 19)
        demo_actions = torch.zeros(1, 1, dataset.max_seq_len, 4)
        demo_masks = torch.zeros(1, 1, dataset.max_seq_len, dtype=torch.bool)

    target_states = target_data['states'].unsqueeze(0)  # (1, T, D)
    target_actions = target_data['actions']
    target_mask = target_data['mask']

    # Move to device
    demo_states = demo_states.to(device)
    demo_actions = demo_actions.to(device)
    demo_masks = demo_masks.to(device)
    target_states = target_states.to(device)

    # Run model
    with torch.no_grad():
        pred_actions = model(
            demo_states=demo_states,
            demo_actions=demo_actions,
            demo_mask=demo_masks,
            current_states=target_states
        )  # (1, T, 4)

    pred_actions = pred_actions.squeeze(0).cpu()  # (T, 4)

    # Compute metrics
    metrics = compute_metrics(pred_actions, target_actions, target_mask)
    metrics['num_demos'] = num_demos
    metrics['track_type'] = target_track
    metrics['episode'] = target_ep_name

    return metrics, pred_actions


def evaluate_all(model, dataset, num_demos_list=[0, 1, 2, 3, 5], device='cpu'):
    """
    Evaluate on all validation episodes with varying number of demos.

    Args:
        model: ICL transformer model
        dataset: ICLRacingDataset
        num_demos_list: List of demo counts to test
        device: Device to run on

    Returns:
        Results dictionary
    """
    results = {
        'per_episode': [],
        'aggregated': {}
    }

    print(f"\nEvaluating on {len(dataset)} episodes...")

    for num_demos in num_demos_list:
        print(f"\n{'='*60}")
        print(f"Evaluating with {num_demos} demonstrations")
        print(f"{'='*60}")

        episode_metrics = []

        for ep_idx in tqdm(range(len(dataset)), desc=f'{num_demos} demos'):
            metrics, _ = evaluate_with_n_demos(
                model, dataset, ep_idx, num_demos, device
            )
            episode_metrics.append(metrics)
            results['per_episode'].append(metrics)

        # Aggregate metrics for this num_demos
        avg_metrics = {
            'num_demos': num_demos,
            'mse': np.mean([m['mse'] for m in episode_metrics]),
            'mse_std': np.std([m['mse'] for m in episode_metrics]),
            'mse_thrust': np.mean([m['mse_thrust'] for m in episode_metrics]),
            'mse_roll': np.mean([m['mse_roll'] for m in episode_metrics]),
            'mse_pitch': np.mean([m['mse_pitch'] for m in episode_metrics]),
            'mse_yaw': np.mean([m['mse_yaw'] for m in episode_metrics]),
            'smoothness': np.mean([m['smoothness'] for m in episode_metrics]),
            'max_error': np.mean([m['max_error'] for m in episode_metrics]),
        }

        # Per-track breakdown
        tracks = set(m['track_type'] for m in episode_metrics)
        per_track = {}
        for track in tracks:
            track_metrics = [m for m in episode_metrics if m['track_type'] == track]
            per_track[track] = {
                'mse': np.mean([m['mse'] for m in track_metrics]),
                'count': len(track_metrics)
            }

        avg_metrics['per_track'] = per_track
        results['aggregated'][num_demos] = avg_metrics

        print(f"\nResults for {num_demos} demos:")
        print(f"  Overall MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['mse_std']:.4f}")
        print(f"  Per-action MSE:")
        print(f"    Thrust: {avg_metrics['mse_thrust']:.4f}")
        print(f"    Roll:   {avg_metrics['mse_roll']:.4f}")
        print(f"    Pitch:  {avg_metrics['mse_pitch']:.4f}")
        print(f"    Yaw:    {avg_metrics['mse_yaw']:.4f}")
        print(f"  Smoothness: {avg_metrics['smoothness']:.4f}")
        print(f"\n  Per-track MSE:")
        for track, track_data in per_track.items():
            print(f"    {track}: {track_data['mse']:.4f} ({track_data['count']} episodes)")

    return results


def plot_results(results, output_dir):
    """Generate plots from evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    sns.set_style("whitegrid")

    # 1. Loss vs Number of Demos
    fig, ax = plt.subplots(figsize=(10, 6))

    num_demos = sorted(results['aggregated'].keys())
    mse_vals = [results['aggregated'][n]['mse'] for n in num_demos]
    mse_std = [results['aggregated'][n]['mse_std'] for n in num_demos]

    ax.errorbar(num_demos, mse_vals, yerr=mse_std, marker='o', linewidth=2,
                capsize=5, markersize=8, label='MSE Loss')
    ax.set_xlabel('Number of Demonstrations', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('ICL Performance vs Number of Demonstrations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_vs_demos.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loss_vs_demos.png'}")

    # 2. Per-action MSE breakdown
    fig, ax = plt.subplots(figsize=(10, 6))

    actions = ['thrust', 'roll', 'pitch', 'yaw']
    x = np.arange(len(num_demos))
    width = 0.2

    for i, action in enumerate(actions):
        vals = [results['aggregated'][n][f'mse_{action}'] for n in num_demos]
        ax.bar(x + i * width, vals, width, label=action.capitalize())

    ax.set_xlabel('Number of Demonstrations', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Per-Action MSE Loss', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(num_demos)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_action_mse.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'per_action_mse.png'}")

    # 3. Per-track performance (with 3 demos as reference)
    fig, ax = plt.subplots(figsize=(10, 6))

    ref_demos = 3 if 3 in num_demos else num_demos[-1]
    tracks = list(results['aggregated'][ref_demos]['per_track'].keys())
    track_mses = [results['aggregated'][ref_demos]['per_track'][t]['mse'] for t in tracks]

    bars = ax.bar(tracks, track_mses, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title(f'Performance per Track Type ({ref_demos} demos)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_track_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'per_track_performance.png'}")

    # 4. Heatmap of performance across demos and tracks
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create matrix: tracks x num_demos
    matrix = []
    for track in sorted(tracks):
        row = []
        for n in num_demos:
            if track in results['aggregated'][n]['per_track']:
                row.append(results['aggregated'][n]['per_track'][track]['mse'])
            else:
                row.append(np.nan)
        matrix.append(row)

    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=num_demos, yticklabels=sorted(tracks),
                cbar_kws={'label': 'MSE Loss'}, ax=ax)
    ax.set_xlabel('Number of Demonstrations', fontsize=12)
    ax.set_ylabel('Track Type', fontsize=12)
    ax.set_title('MSE Loss Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_heatmap.png'}")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='Evaluate ICL performance')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset HDF5')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--num-demos', type=int, nargs='+', default=[0, 1, 2, 3, 5],
                       help='List of demo counts to evaluate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    parser.add_argument('--max-seq-len', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--subsample-factor', type=int, default=10,
                       help='Subsample factor for dataset')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("ICL Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    print(f"Demo counts: {args.num_demos}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = ICLRacingDataset(
        h5_path=args.data,
        num_demos=max(args.num_demos),  # Load with max demos
        max_seq_len=args.max_seq_len,
        use_images=False,
        normalize=True,
        subsample_factor=args.subsample_factor
    )

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model = ICLTransformerPolicy(
        state_dim=19,
        action_dim=4,
        embed_dim=256,
        num_heads=8,
        num_decoder_layers=6,
        use_vision=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    # Run evaluation
    results = evaluate_all(
        model=model,
        dataset=dataset,
        num_demos_list=args.num_demos,
        device=args.device
    )

    # Save results
    results_path = output_dir / 'icl_evaluation.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(results, output_dir / 'plots')

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Demos':<8} {'MSE Loss':<12} {'Thrust':<10} {'Roll':<10} {'Pitch':<10} {'Yaw':<10}")
    print("-"*60)
    for n in sorted(results['aggregated'].keys()):
        r = results['aggregated'][n]
        print(f"{n:<8} {r['mse']:<12.4f} {r['mse_thrust']:<10.4f} {r['mse_roll']:<10.4f} {r['mse_pitch']:<10.4f} {r['mse_yaw']:<10.4f}")
    print("="*60)

    print(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
