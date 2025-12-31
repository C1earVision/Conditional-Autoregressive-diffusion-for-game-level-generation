import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Dict, Tuple

class PatchStitcher:
    def __init__(self,
                 patch_height: int = 14,
                 patch_width: int = 16,
                 stride: int = 4):

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride = stride

    def stitch_patches_to_level(self,
                            patches: np.ndarray,
                            target_width: int = None,
                            ) -> np.ndarray:
        num_patches = len(patches)
        first_patch = patches[0]
        ph, pw = first_patch.shape

        if ph != self.patch_height or pw != self.patch_width:
            self.patch_height = ph
            self.patch_width = pw

        target_width = (num_patches - 1) * self.stride + pw

        level = np.zeros((ph, target_width), dtype=np.int32)

        for i, patch in enumerate(patches):
            if patch.ndim != 2:
                raise ValueError(f"Each patch must be 2D. Got ndim={patch.ndim} at index {i}.")

            ph_i, pw_i = patch.shape
            if ph_i != ph:
                if ph_i > ph:
                    patch = patch[:ph, :]
                else:
                    patch = np.pad(patch, ((0, ph - ph_i), (0, 0)), mode='constant', constant_values=0)

            x_start = i * self.stride

            if i == num_patches - 1:
                x_end = x_start + pw_i
                patch_slice = patch
            else:
                x_end = x_start + self.stride
                patch_slice = patch[:, :self.stride]
            if x_end > target_width:
                overflow = x_end - target_width
                x_end = target_width
                patch_slice = patch_slice[:, :-overflow]
            level[:, x_start:x_end] = patch_slice

        return level

    def compare_cfg_difficulty(self,
                              sampler,
                              normalizer,
                              autoencoder,
                              parser,
                              difficulty_evaluator,
                              guidance_scales: List[float] = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0],
                              num_patches: int = 10,
                              target_difficulty: float = 0.8,
                              temperature: float = 0.5,
                              device: str = 'cuda',
                              save_path: Optional[str] = None) -> Dict:
        print(f"\n{'='*70}")
        print(f"CFG CONDITIONAL VS UNCONDITIONAL COMPARISON")
        print(f"{'='*70}")
        print(f"Target difficulty: {target_difficulty}")
        print(f"Guidance Scales: {guidance_scales}")
        print(f"Patches per condition: {num_patches}")
        print(f"{'='*70}\n")

        print("Step 1: Generating UNCONDITIONAL baseline (scale = 0.0)...")
        latents_uncond = sampler.sample_level(
            num_patches=num_patches,
            normalizer=normalizer,
            difficulty_target=target_difficulty,
            temperature=temperature,
            guidance_scale=0.0,
            show_progress=False
        )
        print(f"✓ Unconditional generation complete: {latents_uncond.shape}")

        latents_denorm = normalizer.denormalize(latents_uncond)
        with torch.no_grad():
            latents_denorm = latents_denorm.to(device)
            decoded_logits = autoencoder.decoder(latents_denorm)

            if decoded_logits.dim() == 4:
                patches = torch.argmax(decoded_logits, dim=1)
            elif decoded_logits.dim() == 3:
                patches = decoded_logits.long()
            elif decoded_logits.dim() == 5:
                decoded_logits = decoded_logits.squeeze(1)
                patches = torch.argmax(decoded_logits, dim=1)
            else:
                patches = torch.argmax(decoded_logits, dim=-1)

        patches_np_uncond = patches.cpu().numpy()
        print(f"✓ Decoded to patches: {patches_np_uncond.shape}")

        uncond_evals = difficulty_evaluator.evaluate_patches_batch(patches_np_uncond)
        uncond_difficulties = [e['scores']['difficulty_score'] for e in uncond_evals]
        uncond_enemies = [e['counts']['enemies'] for e in uncond_evals]
        uncond_cannons = [e['counts']['cannons'] for e in uncond_evals]

        uncond_mean_diff = np.mean(uncond_difficulties)
        uncond_std_diff = np.std(uncond_difficulties)
        print(f"✓ Unconditional difficulty: {uncond_mean_diff:.3f} ± {uncond_std_diff:.3f}")
        print(f"  Enemies: {np.mean(uncond_enemies):.2f}, Cannons: {np.mean(uncond_cannons):.2f}\n")

        print("Step 2: Generating CONDITIONAL with different guidance scales...\n")

        results = {
            'unconditional': {
                'mean_difficulty': uncond_mean_diff,
                'std_difficulty': uncond_std_diff,
                'mean_enemies': np.mean(uncond_enemies),
                'mean_cannons': np.mean(uncond_cannons),
                'difficulties': uncond_difficulties
            },
            'conditional_by_scale': {},
            'comparison_table': []
        }

        conditional_scales = [s for s in guidance_scales if s > 0.0]

        for scale in conditional_scales:
            print(f"  Testing scale = {scale}...")

            latents_cond = sampler.sample_level(
                num_patches=num_patches,
                normalizer=normalizer,
                difficulty_target=target_difficulty,
                temperature=temperature,
                guidance_scale=scale,
                show_progress=False
            )

            latents_denorm = normalizer.denormalize(latents_cond)
            with torch.no_grad():
                latents_denorm = latents_denorm.to(device)
                decoded_logits = autoencoder.decoder(latents_denorm)

                if decoded_logits.dim() == 4:
                    patches = torch.argmax(decoded_logits, dim=1)
                elif decoded_logits.dim() == 3:
                    patches = decoded_logits.long()
                elif decoded_logits.dim() == 5:
                    decoded_logits = decoded_logits.squeeze(1)
                    patches = torch.argmax(decoded_logits, dim=1)
                else:
                    patches = torch.argmax(decoded_logits, dim=-1)

            patches_np_cond = patches.cpu().numpy()

            cond_evals = difficulty_evaluator.evaluate_patches_batch(patches_np_cond)
            cond_difficulties = [e['scores']['difficulty_score'] for e in cond_evals]
            cond_enemies = [e['counts']['enemies'] for e in cond_evals]
            cond_cannons = [e['counts']['cannons'] for e in cond_evals]

            cond_mean_diff = np.mean(cond_difficulties)
            cond_std_diff = np.std(cond_difficulties)

            diff = cond_mean_diff - uncond_mean_diff

            results['conditional_by_scale'][scale] = {
                'mean_difficulty': cond_mean_diff,
                'std_difficulty': cond_std_diff,
                'mean_enemies': np.mean(cond_enemies),
                'mean_cannons': np.mean(cond_cannons),
                'difficulties': cond_difficulties
            }

            print(f"    ✓ Difficulty: {cond_mean_diff:.3f} ± {cond_std_diff:.3f}")
            print(f"    ✓ Δ from uncond: {diff:+.3f}\n")

        print("="*70)
        print("CONDITIONAL VS UNCONDITIONAL COMPARISON")
        print("="*70)
        print(f"{'Scale':<10} {'Uncond':<12} {'Cond':<12} {'Δ':<12} {'%Change':<12}")
        print("-"*70)

        for scale in conditional_scales:
            uncond = results['unconditional']['mean_difficulty']
            cond = results['conditional_by_scale'][scale]['mean_difficulty']
            diff = cond - uncond
            pct = (diff / uncond * 100) if uncond != 0 else 0

            print(f"{scale:<10.1f} {uncond:<12.3f} {cond:<12.3f} {diff:<+12.3f} {pct:<+12.1f}%")

            results['comparison_table'].append({
                'scale': scale,
                'uncond': uncond,
                'cond': cond,
                'diff': diff,
                'pct_change': pct
            })

        print("="*70)

        print("\nStep 3: Testing if different difficulty targets produce different outputs...")
        if len(conditional_scales) >= 2:
            scale_test = conditional_scales[0]
            diff_low_high = abs(
                np.mean(results['conditional_by_scale'][scale_test]['difficulties']) -
                results['unconditional']['mean_difficulty']
            )
            print(f"✓ Different targets produce different outputs: diff = {diff_low_high:.4f}")

        print("\n✓ All CFG comparison tests complete!")
        print("="*70 + "\n")

        if save_path:
            with open(save_path, 'w') as f:
                f.write("CONDITIONAL VS UNCONDITIONAL COMPARISON\n")
                f.write("="*70 + "\n")
                f.write(f"Target difficulty: {target_difficulty}\n")
                f.write(f"Patches per condition: {num_patches}\n\n")

                f.write(f"{'Scale':<10} {'Uncond':<12} {'Cond':<12} {'Δ':<12} {'%Change':<12}\n")
                f.write("-"*70 + "\n")

                for row in results['comparison_table']:
                    f.write(f"{row['scale']:<10.1f} {row['uncond']:<12.3f} {row['cond']:<12.3f} "
                          f"{row['diff']:<+12.3f} {row['pct_change']:<+12.1f}%\n")

            print(f"✓ Results saved to {save_path}")

        return results

    def evaluate_difficulty_comparison(self,
                                      sampler,
                                      normalizer,
                                      autoencoder,
                                      parser,
                                      difficulty_evaluator,
                                      target_difficulties: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
                                      num_samples_per_target: int = 5,
                                      guidance_scale: float = 3.0,
                                      temperature: float = 0.5,
                                      device: str = 'cuda',
                                      save_path: Optional[str] = None) -> Dict:

        print(f"\n{'='*70}")
        print(f"DIFFICULTY EVALUATION COMPARISON")
        print(f"{'='*70}")
        print(f"Target difficulties: {target_difficulties}")
        print(f"Samples per target: {num_samples_per_target}")
        print(f"Guidance Scale: {guidance_scale}")
        print(f"{'='*70}\n")

        results = {
            'target_difficulties': target_difficulties,
            'evaluations': [],
            'summary': {}
        }

        all_targets = []
        all_actual_scores = []

        for target_diff in target_difficulties:
            print(f"\nGenerating {num_samples_per_target} patches with target difficulty = {target_diff}...")

            target_scores = []
            actual_scores = []
            patches_list = []

            for sample_idx in range(num_samples_per_target):
                latent = sampler.sample_single_patch(
                    normalizer=normalizer,
                    previous_latent=None,
                    target_difficulty=target_diff,
                    previous_difficulties=None,
                    temperature=temperature,
                    guidance_scale=guidance_scale,
                    show_progress=False
                )

                latent_denorm = normalizer.denormalize(latent.unsqueeze(0))

                with torch.no_grad():
                    latent_denorm = latent_denorm.to(device)
                    decoded_logits = autoencoder.decoder(latent_denorm)

                    if decoded_logits.dim() == 4:
                        patch = torch.argmax(decoded_logits, dim=1)
                    elif decoded_logits.dim() == 3:
                        patch = decoded_logits.long()
                    elif decoded_logits.dim() == 5:
                        decoded_logits = decoded_logits.squeeze(1)
                        patch = torch.argmax(decoded_logits, dim=1)
                    else:
                        patch = torch.argmax(decoded_logits, dim=-1)

                    patch = patch.cpu().numpy()[0]

                eval_result = difficulty_evaluator.evaluate_patch(
                    patch,
                    metadata={'target_difficulty': target_diff, 'sample_idx': sample_idx}
                )

                difficulty_score = eval_result['scores']['difficulty_score']

                target_scores.append(target_diff)
                actual_scores.append(difficulty_score)
                patches_list.append(patch)

                all_targets.append(target_diff)
                all_actual_scores.append(difficulty_score)

                results['evaluations'].append({
                    'target_difficulty': target_diff,
                    'actual_difficulty': difficulty_score,
                    'sample_idx': sample_idx,
                    'patch': patch,
                    'full_evaluation': eval_result
                })

            mean_actual = np.mean(actual_scores)
            std_actual = np.std(actual_scores)

            results['summary'][target_diff] = {
                'mean_difficulty': mean_actual,
                'std_difficulty': std_actual,
                'target_difficulty': target_diff,
                'error': abs(mean_actual - target_diff),
                'samples': actual_scores
            }

            print(f"  Target: {target_diff:.2f} | "
                  f"Actual Difficulty: {mean_actual:.3f} ± {std_actual:.3f} | "
                  f"Error: {abs(mean_actual - target_diff):.3f}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        target_vals = list(results['summary'].keys())
        mean_vals = [results['summary'][t]['mean_difficulty'] for t in target_vals]
        std_vals = [results['summary'][t]['std_difficulty'] for t in target_vals]

        ax1.errorbar(target_vals, mean_vals, yerr=std_vals,
                    fmt='o', markersize=8, capsize=5, capthick=2,
                    label='Generated (Mean ± Std)', color='blue', alpha=0.7)
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Alignment', alpha=0.5)
        ax1.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Actual Difficulty Score', fontsize=12, fontweight='bold')
        ax1.set_title('Target vs Actual Difficulty (Aggregated)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)

        ax2 = axes[0, 1]
        ax2.scatter(all_targets, all_actual_scores, alpha=0.5, s=50, color='green')
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Alignment', alpha=0.5)
        ax2.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual Difficulty Score', fontsize=12, fontweight='bold')
        ax2.set_title('All Individual Samples', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)

        ax3 = axes[1, 0]
        errors = [results['summary'][t]['error'] for t in target_vals]
        ax3.bar(range(len(target_vals)), errors, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(target_vals)))
        ax3.set_xticklabels([f'{t:.1f}' for t in target_vals])
        ax3.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
        ax3.set_title('Prediction Error by Target', fontsize=14, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)

        ax4 = axes[1, 1]
        data_for_box = [results['summary'][t]['samples'] for t in target_vals]
        bp = ax4.boxplot(data_for_box, positions=range(len(target_vals)),
                        widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax4.plot(range(len(target_vals)), target_vals, 'go-',
                linewidth=2, markersize=8, label='Target', alpha=0.7)
        ax4.set_xticks(range(len(target_vals)))
        ax4.set_xticklabels([f'{t:.1f}' for t in target_vals])
        ax4.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Difficulty Score Distribution', fontsize=12, fontweight='bold')
        ax4.set_title('Distribution of Generated Difficulties', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Evaluation plot saved to {save_path}")

        plt.show()

        print(f"\n{'='*90}")
        print(f"DETAILED EVALUATION RESULTS")
        print(f"{'='*90}")
        print(f"{'Target':<10} {'Generated':<12} {'Std Dev':<12} {'Error':<12} {'% Error':<12} {'Range':<20}")
        print(f"{'-'*90}")

        for target in target_vals:
            summary = results['summary'][target]
            samples = summary['samples']
            range_str = f"[{min(samples):.3f}, {max(samples):.3f}]"
            pct_error = (summary['error'] / target * 100) if target != 0 else 0
            print(f"{target:<10.2f} {summary['mean_difficulty']:<12.3f} "
                  f"{summary['std_difficulty']:<12.3f} {summary['error']:<12.3f} "
                  f"{pct_error:<12.1f}% {range_str:<20}")

        overall_mae = np.mean([results['summary'][t]['error'] for t in target_vals])
        overall_correlation = np.corrcoef(all_targets, all_actual_scores)[0, 1]

        print(f"\n{'='*90}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*90}")
        print(f"Mean Absolute Error (MAE): {overall_mae:.4f}")
        print(f"Correlation Coefficient: {overall_correlation:.4f}")
        print(f"Total Samples Generated: {len(all_targets)}")
        print(f"{'='*90}\n")

        results['overall'] = {
            'mae': overall_mae,
            'correlation': overall_correlation,
            'total_samples': len(all_targets)
        }

        return results