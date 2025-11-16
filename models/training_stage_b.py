"""
Stage B: Generate synthetic SAR images on target domain
Input: RGB pre + building mask
Output: Synthetic SAR post-disaster images
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import argparse
import json

from enhanced_condsar import EnhancedDisasterControlNet
from training_utils import (
    setup_logger, setup_wandb, DisasterSARDataset,
    get_disaster_distribution, get_severity_distribution,
    log_to_wandb
)

logger = logging.getLogger(__name__)


class StageBGenerator:
    """Stage B: Generate synthetic SAR images"""

    def __init__(
        self,
        target_dataset_dir: str,
        controlnet_model_path: str,
        pretrained_model_name: str = "stabilityai/stable-diffusion-2-1-base",
        num_disaster_types: int = 5,
        batch_size: int = 4,
        device: str = "cuda:0",
        output_dir: str = "./synthetic_data",
        log_dir: str = "./logs",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: int = 42,
        num_workers: int = 4,
    ):
        """
        Initialize Stage B generator

        Args:
            target_dataset_dir: Path to target domain dataset
            controlnet_model_path: Path to trained ControlNet checkpoint
            pretrained_model_name: Base diffusion model name
            num_disaster_types: Number of disaster types
            batch_size: Generation batch size
            device: Device to generate on
            output_dir: Directory to save synthetic images
            log_dir: Directory for logs
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            controlnet_conditioning_scale: ControlNet conditioning scale
            seed: Random seed
            num_workers: Number of data loader workers
        """
        self.device = torch.device(device)
        self.num_disaster_types = num_disaster_types
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.num_workers = num_workers

        torch.manual_seed(seed)

        # Setup logging
        self.logger = setup_logger(
            "stage_b_generator",
            log_dir=log_dir,
            level=logging.INFO
        )

        # Setup W&B
        wandb_config = {
            "stage": "B",
            "batch_size": batch_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "seed": seed,
        }
        self.wandb_run = setup_wandb(
            project_name="condsar_stage_b",
            run_name="stage_b_generation",
            config=wandb_config
        )

        self.logger.info("=" * 80)
        self.logger.info("Stage B: Target Domain Synthetic SAR Generation")
        self.logger.info("=" * 80)

        # Load target dataset
        self.logger.info(f"Loading target dataset from {target_dataset_dir}")
        self.target_dataset = DisasterSARDataset(
            dataset_dir=target_dataset_dir,
            image_size=512,
            return_mask=True,
            return_metadata=False,  # Don't need GT SAR or metadata for generation
            logger=self.logger
        )

        self.target_loader = DataLoader(
            self.target_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.logger.info(f"Dataset loaded: {len(self.target_dataset)} samples")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)

        # Load trained ControlNet
        self.logger.info(f"Loading ControlNet from {controlnet_model_path}")
        checkpoint = torch.load(controlnet_model_path, map_location=self.device)

        # Create model and load weights
        from enhanced_condsar import create_enhanced_controlnet
        self.controlnet = create_enhanced_controlnet(
            pretrained_model_name_or_path=pretrained_model_name,
            num_disaster_types=num_disaster_types,
            torch_dtype=torch.float32
        )
        self.controlnet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.controlnet = self.controlnet.to(self.device)
        self.controlnet.eval()

        # Create diffusion pipeline
        self.logger.info("Creating diffusion pipeline")
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name,
            controlnet=self.controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.enable_attention_slicing()

        # Metadata to save
        self.metadata = {}

        self.logger.info("Stage B generator initialized successfully")

    @torch.no_grad()
    def generate_batch(
        self,
        batch: Dict,
        disaster_type: Optional[int] = None,
        disaster_severity: Optional[float] = None,
    ) -> Dict:
        """
        Generate synthetic SAR images for a batch

        Args:
            batch: Batch from target dataset
            disaster_type: Disaster type (if None, random)
            disaster_severity: Disaster severity (if None, random)

        Returns:
            Dictionary with generated images and metadata
        """
        batch_size = batch['rgb_image'].size(0)

        # Set disaster type and severity
        if disaster_type is None:
            disaster_types = get_disaster_distribution(batch_size, distribution="natural")
        else:
            disaster_types = [disaster_type] * batch_size

        if disaster_severity is None:
            severities = get_severity_distribution(batch_size, distribution="natural")
        else:
            severities = [disaster_severity] * batch_size

        # Convert to tensors
        disaster_type_tensor = torch.tensor(disaster_types, dtype=torch.long, device=self.device)
        severity_tensor = torch.tensor(severities, dtype=torch.float32, device=self.device)

        # Rescale RGB to [-1, 1] for diffusion model
        rgb_images = batch['rgb_image'].to(self.device) * 2 - 1

        # Generate SAR images
        with torch.no_grad():
            sar_images = self.pipeline(
                prompt=["satellite SAR image after disaster"] * batch_size,
                image=rgb_images,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                # Custom kwargs for our ControlNet
                disaster_type=disaster_type_tensor,
                disaster_severity=severity_tensor,
                building_mask=batch['building_mask'].to(self.device) if batch['building_mask'] is not None else None,
                output_type='np'
            ).images

        return {
            'image_ids': batch['image_id'],
            'sar_images': sar_images,
            'rgb_images': batch['rgb_image'],
            'disaster_types': disaster_types,
            'severities': severities,
        }

    def generate_all(
        self,
        num_variants_per_sample: int = 1,
        disaster_type_bias: Optional[int] = None,
        disaster_severity_bias: Optional[float] = None,
    ):
        """
        Generate synthetic SAR images for all target samples

        Args:
            num_variants_per_sample: Number of synthetic variants per target sample
            disaster_type_bias: If set, bias generation to this disaster type
            disaster_severity_bias: If set, bias generation to this severity
        """
        total_generated = 0

        for batch_idx, batch in enumerate(self.target_loader):
            self.logger.info(f"\nGenerating batch {batch_idx + 1}/{len(self.target_loader)}")

            for variant_idx in range(num_variants_per_sample):
                # Generate synthetic images
                result = self.generate_batch(
                    batch,
                    disaster_type=disaster_type_bias,
                    disaster_severity=disaster_severity_bias
                )

                # Save images and metadata
                for i, image_id in enumerate(result['image_ids']):
                    # Save synthetic SAR
                    sar_filename = self.output_dir / f"{image_id}_synthetic_sar_v{variant_idx}.png"
                    sar_image = result['sar_images'][i]

                    # Convert from [0, 1] to uint8
                    from PIL import Image
                    import numpy as np
                    sar_image_uint8 = (sar_image * 255).astype(np.uint8)
                    if len(sar_image_uint8.shape) == 3:
                        sar_image_uint8 = sar_image_uint8.mean(axis=2)
                    Image.fromarray(sar_image_uint8, mode='L').save(sar_filename)

                    # Save RGB (for reference)
                    rgb_filename = self.output_dir / f"{image_id}_rgb_input.png"
                    rgb_image_uint8 = (result['rgb_images'][i].numpy() * 255).astype(np.uint8)
                    rgb_image_uint8 = np.transpose(rgb_image_uint8, (1, 2, 0))
                    Image.fromarray(rgb_image_uint8).save(rgb_filename)

                    # Save metadata
                    meta_entry = {
                        'image_id': image_id,
                        'variant': variant_idx,
                        'disaster_type': int(result['disaster_types'][i]),
                        'disaster_severity': float(result['severities'][i]),
                        'sar_path': str(sar_filename),
                        'rgb_path': str(rgb_filename),
                    }
                    self.metadata[f"{image_id}_v{variant_idx}"] = meta_entry

                    total_generated += 1

                    if (total_generated % 10) == 0:
                        self.logger.info(f"Generated {total_generated} synthetic images")

        # Save metadata to JSON
        metadata_path = self.output_dir / "synthetic_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"Stage B Generation Completed!")
        self.logger.info(f"Generated {total_generated} synthetic images")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Metadata saved to: {metadata_path}")
        self.logger.info("=" * 80)

        # Log to W&B
        log_to_wandb({
            'total_generated': total_generated,
            'output_dir': str(self.output_dir)
        })

        if self.wandb_run is not None:
            self.wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description="Stage B: Generate Synthetic SAR Images")
    parser.add_argument("--target_dataset_dir", type=str, required=True, help="Path to target dataset")
    parser.add_argument("--controlnet_model_path", type=str, required=True, help="Path to trained ControlNet")
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./synthetic_data")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_variants_per_sample", type=int, default=1)
    parser.add_argument("--disaster_type_bias", type=int, default=None)
    parser.add_argument("--disaster_severity_bias", type=float, default=None)
    parser.add_argument("--wandb_offline", action="store_true")

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger("stage_b_main", log_dir=args.log_dir)
    logger.info(f"Arguments: {args}")

    # Create generator
    generator = StageBGenerator(
        target_dataset_dir=args.target_dataset_dir,
        controlnet_model_path=args.controlnet_model_path,
        pretrained_model_name=args.pretrained_model,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # Start generation
    generator.generate_all(
        num_variants_per_sample=args.num_variants_per_sample,
        disaster_type_bias=args.disaster_type_bias,
        disaster_severity_bias=args.disaster_severity_bias
    )


if __name__ == "__main__":
    main()

