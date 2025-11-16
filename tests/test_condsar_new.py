"""
Unit tests for CONDSAR models and training
"""
import unittest
import torch
import tempfile
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from enhanced_condsar import (
    DisasterTypeEmbedding,
    DisasterSeverityEmbedding,
    EnhancedDisasterControlNet,
)
from training_utils import (
    DisasterSARDataset,
    MetricsTracker,
    get_disaster_distribution,
    get_severity_distribution,
)


class TestDisasterEmbeddings(unittest.TestCase):
    """Test disaster type and severity embeddings"""

    def test_disaster_type_embedding(self):
        """Test DisasterTypeEmbedding"""
        embedding = DisasterTypeEmbedding(num_disaster_types=5, embedding_dim=128)

        # Test with batch
        batch_size = 4
        disaster_types = torch.randint(0, 5, (batch_size,))

        embeddings = embedding(disaster_types)

        self.assertEqual(embeddings.shape, (batch_size, 128))
        self.assertEqual(embeddings.dtype, torch.float32)

    def test_severity_embedding(self):
        """Test DisasterSeverityEmbedding"""
        embedding = DisasterSeverityEmbedding(num_severity_levels=4, embedding_dim=128)

        # Test with continuous values
        batch_size = 4
        severities = torch.rand(batch_size)

        embeddings = embedding(severities)

        self.assertEqual(embeddings.shape, (batch_size, 128))
        self.assertEqual(embeddings.dtype, torch.float32)

        # Test with one-hot encoding
        one_hot = torch.zeros(batch_size, 4)
        one_hot[torch.arange(batch_size), torch.randint(0, 4, (batch_size,))] = 1

        embeddings_onehot = embedding(one_hot)
        self.assertEqual(embeddings_onehot.shape, (batch_size, 128))


class TestEnhancedControlNet(unittest.TestCase):
    """Test EnhancedDisasterControlNet model"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.height = 64
        self.width = 64

    @unittest.skip("Requires downloading pretrained model - test locally with proper model")
    def test_controlnet_forward(self):
        """Test forward pass through ControlNet"""
        # This test requires pretrained model - skip in CI
        model = EnhancedDisasterControlNet(
            in_channels=4,
            num_disaster_types=5,
            disaster_embedding_dim=128,
            num_severity_levels=4,
            severity_embedding_dim=128,
            block_out_channels=(320, 640, 1280, 1280),
        ).to(self.device)

        # Create dummy inputs
        sample = torch.randn(self.batch_size, 4, self.height, self.width, device=self.device)
        timestep = torch.randint(0, 1000, (self.batch_size,), device=self.device)
        rgb_image = torch.randn(self.batch_size, 3, 512, 512, device=self.device)
        building_mask = torch.randn(self.batch_size, 1, 512, 512, device=self.device)
        disaster_type = torch.randint(0, 5, (self.batch_size,), device=self.device)
        disaster_severity = torch.rand(self.batch_size, device=self.device)

        # Forward pass
        with torch.no_grad():
            output = model(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=None,
                rgb_image=rgb_image,
                building_mask=building_mask,
                disaster_type=disaster_type,
                disaster_severity=disaster_severity,
            )

        self.assertIsNotNone(output)


class TestDataset(unittest.TestCase):
    """Test dataset loading and processing"""

    def setUp(self):
        """Create temporary dataset"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

        # Create subdirectories
        (self.dataset_dir / "pre").mkdir()
        (self.dataset_dir / "post").mkdir()
        (self.dataset_dir / "mask").mkdir()

        # Create dummy images
        import numpy as np
        from PIL import Image

        # Create dummy data
        for i in range(3):
            # RGB image
            rgb_data = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(rgb_data, 'RGB').save(
                self.dataset_dir / "pre" / f"image_{i}.jpg"
            )

            # SAR image
            sar_data = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            Image.fromarray(sar_data, 'L').save(
                self.dataset_dir / "post" / f"image_{i}.jpg"
            )

            # Mask
            mask_data = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
            Image.fromarray(mask_data, 'L').save(
                self.dataset_dir / "mask" / f"image_{i}.png"
            )

        # Create metadata
        metadata = {
            "image_0.jpg": {
                "disaster_type": 0,
                "severity": 0.3,
                "damage_level": 1
            },
            "image_1.jpg": {
                "disaster_type": 1,
                "severity": 0.6,
                "damage_level": 2
            },
            "image_2.jpg": {
                "disaster_type": 2,
                "severity": 0.9,
                "damage_level": 3
            }
        }

        with open(self.dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

    def tearDown(self):
        """Clean up"""
        self.temp_dir.cleanup()

    def test_dataset_loading(self):
        """Test dataset loading"""
        dataset = DisasterSARDataset(
            dataset_dir=str(self.dataset_dir),
            image_size=256,
            return_mask=True,
            return_metadata=True,
        )

        self.assertEqual(len(dataset), 3)

        # Test getting an item
        item = dataset[0]

        self.assertIn('rgb_image', item)
        self.assertIn('sar_image', item)
        self.assertIn('building_mask', item)
        self.assertIn('disaster_type', item)
        self.assertIn('disaster_severity', item)

        # Check shapes
        self.assertEqual(item['rgb_image'].shape, (3, 256, 256))
        self.assertEqual(item['sar_image'].shape, (1, 256, 256))
        self.assertEqual(item['building_mask'].shape, (1, 256, 256))

        # Check types
        self.assertEqual(item['disaster_type'].item(), 0)
        self.assertAlmostEqual(item['disaster_severity'].item(), 0.3, places=1)


class TestMetricsTracker(unittest.TestCase):
    """Test MetricsTracker utility"""

    def test_metrics_tracking(self):
        """Test metrics accumulation and computation"""
        tracker = MetricsTracker()

        # Add some metrics
        for i in range(10):
            tracker.update(loss=1.0 + i * 0.1, accuracy=0.5 + i * 0.05)

        # Check means
        mean_loss = tracker.get_mean('loss')
        mean_accuracy = tracker.get_mean('accuracy')

        self.assertAlmostEqual(mean_loss, 1.45, places=2)
        self.assertAlmostEqual(mean_accuracy, 0.725, places=2)

        # Check to_dict
        metrics_dict = tracker.to_dict()
        self.assertIn('loss', metrics_dict)
        self.assertIn('accuracy', metrics_dict)

        # Check reset
        tracker.reset()
        self.assertEqual(len(tracker.metrics), 0)


class TestDistributions(unittest.TestCase):
    """Test disaster distribution utilities"""

    def test_disaster_distribution_balanced(self):
        """Test balanced disaster distribution"""
        num_samples = 100
        disasters = get_disaster_distribution(num_samples, distribution="balanced")

        self.assertEqual(len(disasters), num_samples)
        self.assertTrue(all(0 <= d < 5 for d in disasters))

    def test_disaster_distribution_natural(self):
        """Test natural disaster distribution"""
        num_samples = 100
        disasters = get_disaster_distribution(num_samples, distribution="natural")

        self.assertEqual(len(disasters), num_samples)
        self.assertTrue(all(0 <= d < 5 for d in disasters))

    def test_severity_distribution_uniform(self):
        """Test uniform severity distribution"""
        num_samples = 100
        severities = get_severity_distribution(num_samples, distribution="uniform")

        self.assertEqual(len(severities), num_samples)
        self.assertTrue(all(0 <= s <= 1 for s in severities))

    def test_severity_distribution_natural(self):
        """Test natural severity distribution"""
        num_samples = 100
        severities = get_severity_distribution(num_samples, distribution="natural")

        self.assertEqual(len(severities), num_samples)
        self.assertTrue(all(0 <= s <= 1 for s in severities))


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration utilities"""

    def test_config_save_load(self):
        """Test saving and loading configuration"""
        from training_utils import setup_logger
        from train_pipeline import create_default_config, save_config, load_config

        config = create_default_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"

            # Save
            save_config(config, str(config_path))
            self.assertTrue(config_path.exists())

            # Load
            loaded_config = load_config(str(config_path))

            # Compare
            self.assertEqual(config['device'], loaded_config['device'])
            self.assertEqual(config['stage_a']['batch_size'], loaded_config['stage_a']['batch_size'])


class TestDataLoading(unittest.TestCase):
    """Test data loading and batching"""

    def setUp(self):
        """Create temporary dataset"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

        # Create subdirectories
        (self.dataset_dir / "pre").mkdir()
        (self.dataset_dir / "post").mkdir()

        # Create dummy images
        import numpy as np
        from PIL import Image

        for i in range(5):
            rgb_data = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(rgb_data, 'RGB').save(
                self.dataset_dir / "pre" / f"image_{i}.jpg"
            )

            sar_data = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            Image.fromarray(sar_data, 'L').save(
                self.dataset_dir / "post" / f"image_{i}.jpg"
            )

        # Minimal metadata
        metadata = {f"image_{i}.jpg": {"disaster_type": i % 5, "severity": 0.5} for i in range(5)}
        with open(self.dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

    def tearDown(self):
        """Clean up"""
        self.temp_dir.cleanup()

    def test_dataloader(self):
        """Test DataLoader with dataset"""
        from torch.utils.data import DataLoader

        dataset = DisasterSARDataset(
            dataset_dir=str(self.dataset_dir),
            image_size=256,
            return_mask=False,
            return_metadata=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )

        # Get one batch
        batch = next(iter(dataloader))

        self.assertEqual(batch['rgb_image'].shape[0], 2)
        self.assertEqual(batch['sar_image'].shape[0], 2)
        self.assertEqual(len(batch['image_id']), 2)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()

