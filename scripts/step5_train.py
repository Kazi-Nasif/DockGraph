#!/usr/bin/env python3 -u
"""
Production Training Script for Protein Docking
Trains on ALL targets from Benchmark 5.5
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from step2_data_loader import ProteinDockingLoader
from step3_feature_extraction import ProteinFeatureExtractor
from step4_model import ProteinDockingModel, prepare_graph_data


def log_print(message, log_file=None):
    """Print and log message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    sys.stdout.flush()
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')


def prepare_dataset(difficulties=["rigid_targets", "medium_targets", "difficult_targets"], 
                    max_total=None, log_file=None):
    """
    Prepare dataset from all difficulties
    
    Args:
        difficulties: List of difficulty levels
        max_total: Maximum total samples (None = all)
        log_file: Path to log file
    
    Returns:
        list of data dicts
    """
    log_print("="*80, log_file)
    log_print("PREPARING DATASET", log_file)
    log_print("="*80, log_file)
    
    loader = ProteinDockingLoader()
    extractor = ProteinFeatureExtractor()
    
    all_data = []
    
    for difficulty in difficulties:
        log_print(f"\nLoading {difficulty}...", log_file)
        
        benchmark_path = Path(loader.benchmark_path) / difficulty
        target_dirs = sorted([d for d in benchmark_path.iterdir() if d.is_dir()])
        
        log_print(f"  Found {len(target_dirs)} targets", log_file)
        
        for target_dir in tqdm(target_dirs, desc=f"  {difficulty}"):
            target_name = target_dir.name
            
            try:
                # Extract features
                features = extractor.extract_full_features(target_name, difficulty)
                
                # Prepare graph data (returns 3 items now!)
                receptor_data, ligand_data, bound_coords = prepare_graph_data(features)
                
                # Store data
                data_dict = {
                    'target_name': target_name,
                    'difficulty': difficulty,
                    'receptor_data': receptor_data,
                    'ligand_data': ligand_data,
                    'bound_coords': bound_coords  # Ground truth ligand position
                }
                
                all_data.append(data_dict)
                
            except Exception as e:
                log_print(f"    ✗ Error on {target_name}: {e}", log_file)
                continue
        
        log_print(f"  ✓ Loaded {len([d for d in all_data if d['difficulty'] == difficulty])} from {difficulty}", log_file)
        
        # Check if we've hit max
        if max_total and len(all_data) >= max_total:
            all_data = all_data[:max_total]
            break
    
    log_print(f"\n✓ Total dataset: {len(all_data)} samples", log_file)
    
    # Show breakdown
    log_print("\nDataset breakdown:", log_file)
    for diff in difficulties:
        count = len([d for d in all_data if d['difficulty'] == diff])
        log_print(f"  {diff}: {count}", log_file)
    
    return all_data


def train_model(model, train_data, val_data, device, num_epochs=50, 
                learning_rate=1e-3, save_dir=None, log_file=None):
    """
    Train the model
    
    Args:
        model: ProteinDockingModel
        train_data: Training dataset
        val_data: Validation dataset
        device: 'cuda' or 'cpu'
        num_epochs: Number of epochs
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
        log_file: Log file path
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'val_success_rates': [],
        'epoch_times': []
    }
    
    log_print("\n" + "="*80, log_file)
    log_print("TRAINING START", log_file)
    log_print("="*80, log_file)
    log_print(f"Train samples: {len(train_data)}", log_file)
    log_print(f"Val samples: {len(val_data)}", log_file)
    log_print(f"Epochs: {num_epochs}", log_file)
    log_print(f"Device: {device}", log_file)
    log_print(f"Learning rate: {learning_rate}", log_file)
    log_print("="*80, log_file)
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        log_print(f"\n{'='*80}", log_file)
        log_print(f"EPOCH {epoch}/{num_epochs}", log_file)
        log_print(f"{'='*80}", log_file)
        
        # TRAINING
        model.train()
        train_loss = 0.0
        train_count = 0
        
        log_print("Training...", log_file)
        
        for i, data_dict in enumerate(tqdm(train_data, desc="  Train")):
            try:
                # Move to device
                receptor_data = {
                    'node_features': data_dict['receptor_data']['node_features'].to(device),
                    'edge_index': data_dict['receptor_data']['edge_index'].to(device),
                    'coords': data_dict['receptor_data']['coords'].to(device)
                }
                
                ligand_data = {
                    'node_features': data_dict['ligand_data']['node_features'].to(device),
                    'edge_index': data_dict['ligand_data']['edge_index'].to(device),
                    'coords': data_dict['ligand_data']['coords'].to(device)
                }
                
                bound_coords = data_dict['bound_coords'].to(device)
                
                # Forward pass
                rotation, translation, confidence = model(receptor_data, ligand_data)
                pred_coords = model.apply_transformation(ligand_data['coords'], rotation, translation)
                
                # Calculate RMSD loss
                diff = pred_coords - bound_coords
                rmsd = torch.sqrt(torch.mean(torch.sum(diff**2, dim=1)))
                
                # Add confidence regularization
                if rmsd < 4.0:
                    conf_loss = -torch.log(confidence + 1e-6)  # Encourage high confidence
                else:
                    conf_loss = torch.log(confidence + 1e-6)  # Encourage low confidence
                
                loss = rmsd + 0.1 * conf_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += rmsd.item()
                train_count += 1
                
                # Log every 20 samples
                if (i + 1) % 20 == 0:
                    log_print(f"    [{i+1}/{len(train_data)}] RMSD: {rmsd.item():.4f} Å", log_file)
                
            except Exception as e:
                log_print(f"    ✗ Error on sample {i}: {e}", log_file)
                continue
        
        avg_train_loss = train_loss / max(train_count, 1)
        log_print(f"  Train RMSD: {avg_train_loss:.4f} Å", log_file)
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_count = 0
        val_success = 0  # RMSD < 4.0 Å
        
        log_print("Validating...", log_file)
        
        with torch.no_grad():
            for data_dict in tqdm(val_data, desc="  Val"):
                try:
                    receptor_data = {
                        'node_features': data_dict['receptor_data']['node_features'].to(device),
                        'edge_index': data_dict['receptor_data']['edge_index'].to(device),
                        'coords': data_dict['receptor_data']['coords'].to(device)
                    }
                    
                    ligand_data = {
                        'node_features': data_dict['ligand_data']['node_features'].to(device),
                        'edge_index': data_dict['ligand_data']['edge_index'].to(device),
                        'coords': data_dict['ligand_data']['coords'].to(device)
                    }
                    
                    bound_coords = data_dict['bound_coords'].to(device)
                    
                    rotation, translation, confidence = model(receptor_data, ligand_data)
                    pred_coords = model.apply_transformation(ligand_data['coords'], rotation, translation)
                    
                    diff = pred_coords - bound_coords
                    rmsd = torch.sqrt(torch.mean(torch.sum(diff**2, dim=1)))
                    
                    val_loss += rmsd.item()
                    val_count += 1
                    
                    if rmsd.item() < 4.0:
                        val_success += 1
                    
                except Exception as e:
                    continue
        
        avg_val_loss = val_loss / max(val_count, 1)
        val_success_rate = (val_success / max(val_count, 1)) * 100
        
        log_print(f"  Val RMSD: {avg_val_loss:.4f} Å", log_file)
        log_print(f"  Val Success (RMSD < 4 Å): {val_success_rate:.1f}%", log_file)
        
        # Update metrics
        epoch_time = time.time() - epoch_start
        metrics['train_losses'].append(float(avg_train_loss))
        metrics['val_losses'].append(float(avg_val_loss))
        metrics['val_success_rates'].append(float(val_success_rate))
        metrics['epoch_times'].append(float(epoch_time))
        
        # Save best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            
            if save_dir:
                best_path = save_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_success_rate': val_success_rate,
                    'metrics': metrics
                }, best_path)
                log_print(f"  ✓ Saved best model (val RMSD: {avg_val_loss:.4f})", log_file)
        
        # Save checkpoint every 10 epochs
        if save_dir and epoch % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
        
        # Save metrics
        if save_dir:
            metrics_path = save_dir / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        log_print(f"  Epoch time: {epoch_time:.1f}s", log_file)
        log_print(f"  Best: epoch {best_epoch}, val RMSD {best_val_loss:.4f} Å", log_file)
    
    log_print("\n" + "="*80, log_file)
    log_print("TRAINING COMPLETE", log_file)
    log_print("="*80, log_file)
    log_print(f"Best model: epoch {best_epoch}, val RMSD {best_val_loss:.4f} Å", log_file)
    
    return metrics


def main():
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    TRAIN_ALL = True  # Set to False for testing
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = PROJECT_ROOT / 'experiments' / f'{timestamp}_full_training'
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_file = exp_dir / 'training.log'
    
    log_print("="*80, log_file)
    log_print("PROTEIN DOCKING - PRODUCTION TRAINING", log_file)
    log_print("="*80, log_file)
    log_print(f"Experiment: {exp_dir.name}", log_file)
    log_print(f"Device: {DEVICE}", log_file)
    log_print(f"Epochs: {NUM_EPOCHS}", log_file)
    
    # Prepare dataset
    if TRAIN_ALL:
        dataset = prepare_dataset(
            difficulties=["rigid_targets", "medium_targets", "difficult_targets"],
            max_total=None,  # All targets
            log_file=log_file
        )
    else:
        # For testing: just rigid subset
        dataset = prepare_dataset(
            difficulties=["rigid_targets"],
            max_total=50,
            log_file=log_file
        )
    
    # Split train/val (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    split_idx = int(0.8 * len(dataset))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    
    log_print(f"\nSplit:", log_file)
    log_print(f"  Train: {len(train_data)} samples", log_file)
    log_print(f"  Val: {len(val_data)} samples", log_file)
    
    # Create model
    log_print("\nCreating model...", log_file)
    model = ProteinDockingModel(
        node_features=26,
        hidden_dim=128,
        num_layers=3
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    log_print(f"  Parameters: {num_params:,}", log_file)
    
    # Train
    metrics = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        save_dir=exp_dir,
        log_file=log_file
    )
    
    log_print("\n" + "="*80, log_file)
    log_print("DONE!", log_file)
    log_print(f"Results: {exp_dir}", log_file)
    log_print("="*80, log_file)


if __name__ == "__main__":
    main()
