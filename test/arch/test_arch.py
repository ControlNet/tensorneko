"""Smoke tests for tensorneko architecture classes, NekoModel, and NekoTrainer."""

import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, patch

import torch
import torch.nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from tensorneko.arch import AutoEncoder, GAN, WGAN, VQVAE, BinaryClassifier
from tensorneko.neko_model import NekoModel
from tensorneko.neko_trainer import NekoTrainer


# ---------------------------------------------------------------------------
# Concrete subclasses for ABC architectures
# ---------------------------------------------------------------------------


class _TinyAE(AutoEncoder):
    def build_encoder(self):
        return torch.nn.Linear(8, 4)

    def build_decoder(self):
        return torch.nn.Linear(4, 8)


class _TinyGAN(GAN):
    def build_generator(self):
        return torch.nn.Linear(4, 8)

    def build_discriminator(self):
        return torch.nn.Linear(8, 1)


class _TinyWGAN(WGAN):
    def build_generator(self):
        # Must output 4D (batch, C, H, W) for gradient_penalty_fn
        return torch.nn.Sequential(
            torch.nn.Linear(4, 1 * 2 * 2),
            torch.nn.Unflatten(1, (1, 2, 2)),
        )

    def build_discriminator(self):
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4, 1),
        )


class _TinyVQVAE(VQVAE):
    """Encoder outputs (B, C, H, W); decoder accepts (B, C, H, W) → (B, C, H, W)."""

    def build_encoder(self):
        # Input: (B, 1, 4, 4) → (B, 4, 2, 2)  latent_dim=4 is the channel dim
        return torch.nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)

    def build_decoder(self):
        # (B, 4, 2, 2) → (B, 1, 4, 4)
        return torch.nn.ConvTranspose2d(
            4, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )


class _SimpleModel(NekoModel):
    """Minimal concrete NekoModel for testing."""

    def __init__(self):
        super().__init__("test_model")
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch=None, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch=None, batch_idx=None, dataloader_idx=None):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        self.model = _TinyAE("ae", learning_rate=1e-3)

    def test_construction(self):
        self.assertIsInstance(self.model, AutoEncoder)
        self.assertIsInstance(self.model, NekoModel)

    def test_forward_shape(self):
        x = torch.rand(2, 8)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 8))

    def test_training_step(self):
        batch = (torch.rand(2, 8),)
        result = self.model.training_step(batch)
        self.assertIn("loss", result)
        self.assertEqual(result["loss"].dim(), 0)  # scalar

    def test_validation_step(self):
        batch = (torch.rand(2, 8),)
        result = self.model.validation_step(batch)
        self.assertIn("loss", result)
        self.assertEqual(result["loss"].dim(), 0)

    def test_configure_optimizers(self):
        opt_cfg = self.model.configure_optimizers()
        self.assertIn("optimizer", opt_cfg)
        self.assertIn("lr_scheduler", opt_cfg)


class TestGAN(unittest.TestCase):
    def setUp(self):
        self.model = _TinyGAN(latent_dim=4, g_learning_rate=1e-3, d_learning_rate=1e-3)

    def test_construction(self):
        self.assertIsInstance(self.model, GAN)
        self.assertIsInstance(self.model, NekoModel)
        self.assertFalse(self.model.automatic_optimization)

    def test_forward_shape(self):
        z = torch.rand(2, 4)
        out = self.model(z)
        self.assertEqual(out.shape, (2, 8))

    def test_d_step(self):
        x = torch.rand(2, 8)
        z = torch.rand(2, 4)
        result = self.model.d_step(x, z)
        self.assertIn("loss", result)
        self.assertIn("loss/d_loss", result)
        self.assertEqual(result["loss"].dim(), 0)

    def test_g_step(self):
        x = torch.rand(2, 8)
        z = torch.rand(2, 4)
        result = self.model.g_step(x, z)
        self.assertIn("loss", result)
        self.assertIn("loss/g_loss", result)
        self.assertEqual(result["loss"].dim(), 0)

    def test_validation_step(self):
        batch = (torch.rand(2, 8),)
        result = self.model.validation_step(batch)
        self.assertIn("loss", result)

    def test_configure_optimizers(self):
        opts = self.model.configure_optimizers()
        self.assertIsInstance(opts, list)
        self.assertEqual(len(opts), 2)


class TestWGAN(unittest.TestCase):
    def setUp(self):
        self.model = _TinyWGAN(
            latent_dim=4,
            g_learning_rate=1e-3,
            d_learning_rate=1e-3,
            gp_weight=10.0,
        )

    def test_construction(self):
        self.assertIsInstance(self.model, WGAN)
        self.assertIsInstance(self.model, GAN)

    def test_forward_shape(self):
        z = torch.rand(2, 4)
        out = self.model(z)
        self.assertEqual(out.shape, (2, 1, 2, 2))

    def test_d_step_with_gp(self):
        self.model.train()  # ensure training mode for gradient penalty
        x = torch.rand(2, 1, 2, 2)
        z = torch.rand(2, 4)
        result = self.model.d_step(x, z)
        self.assertIn("loss", result)
        self.assertIn("loss/d_loss", result)
        self.assertIn("loss/gp", result)
        # gp should be non-zero in training mode with gp_weight > 0
        self.assertGreater(result["loss/gp"].item(), 0.0)

    def test_d_step_eval_no_gp(self):
        self.model.eval()
        x = torch.rand(2, 1, 2, 2)
        z = torch.rand(2, 4)
        result = self.model.d_step(x, z)
        self.assertIn("loss/gp", result)
        self.assertEqual(result["loss/gp"].item(), 0.0)

    def test_configure_optimizers(self):
        opts = self.model.configure_optimizers()
        self.assertIsInstance(opts, list)
        self.assertEqual(len(opts), 2)


class TestVQVAE(unittest.TestCase):
    def setUp(self):
        self.model = _TinyVQVAE(latent_dim=4, n_embeddings=8)

    def test_construction(self):
        self.assertIsInstance(self.model, VQVAE)
        self.assertIsInstance(self.model, NekoModel)

    def test_forward_shape(self):
        x = torch.rand(2, 1, 4, 4)
        x_hat, embedding_loss = self.model(x)
        self.assertEqual(x_hat.shape, (2, 1, 4, 4))
        self.assertEqual(embedding_loss.dim(), 0)

    def test_training_step(self):
        batch = (torch.rand(2, 1, 4, 4),)
        result = self.model.training_step(batch)
        self.assertIn("loss", result)
        self.assertIn("loss/r_loss", result)
        self.assertIn("loss/e_loss", result)

    def test_validation_step(self):
        batch = (torch.rand(2, 1, 4, 4),)
        result = self.model.validation_step(batch)
        self.assertIn("loss", result)
        self.assertIn("loss/r_loss", result)
        self.assertIn("loss/e_loss", result)

    def test_configure_optimizers(self):
        opt_cfg = self.model.configure_optimizers()
        self.assertIn("optimizer", opt_cfg)
        self.assertIn("lr_scheduler", opt_cfg)


class TestBinaryClassifier(unittest.TestCase):
    def setUp(self):
        self.inner = torch.nn.Linear(8, 1)
        self.model = BinaryClassifier("bc", model=self.inner, learning_rate=1e-3)

    def test_construction(self):
        self.assertIsInstance(self.model, BinaryClassifier)
        self.assertIsInstance(self.model, NekoModel)

    def test_forward_shape(self):
        x = torch.rand(4, 8)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 1))

    def test_step(self):
        batch = (torch.rand(4, 8), torch.randint(0, 2, (4,)).float())
        result = self.model.step(batch)
        self.assertIn("loss", result)
        self.assertIn("metric/acc", result)
        self.assertIn("metric/f1", result)
        self.assertIn("metric/auc", result)

    def test_training_step(self):
        batch = (torch.rand(4, 8), torch.randint(0, 2, (4,)).float())
        result = self.model.training_step(batch)
        self.assertIn("loss", result)

    def test_validation_step(self):
        batch = (torch.rand(4, 8), torch.randint(0, 2, (4,)).float())
        result = self.model.validation_step(batch)
        self.assertIn("loss", result)

    def test_from_module(self):
        inner = torch.nn.Linear(8, 1)
        bc = BinaryClassifier.from_module(inner)
        self.assertIsInstance(bc, BinaryClassifier)
        x = torch.rand(2, 8)
        out = bc(x)
        self.assertEqual(out.shape, (2, 1))

    def test_configure_optimizers(self):
        opts = self.model.configure_optimizers()
        self.assertIsInstance(opts, list)
        self.assertEqual(len(opts), 1)


class TestNekoModel(unittest.TestCase):
    def test_init(self):
        model = _SimpleModel()
        self.assertEqual(model.name, "test_model")
        self.assertFalse(model.can_plot_graph)
        self.assertEqual(model.history, [])

    def test_forward(self):
        model = _SimpleModel()
        x = torch.rand(2, 4)
        y = model(x)
        self.assertEqual(y.shape, (2, 2))

    def test_predict_step(self):
        model = _SimpleModel()
        x = torch.rand(2, 4)
        y = model.predict_step(x, 0)
        self.assertEqual(y.shape, (2, 2))

    def test_test_step_delegates_to_validation(self):
        model = _SimpleModel()
        batch = (torch.rand(2, 4), torch.rand(2, 2))
        result = model.test_step(batch, 0)
        self.assertIn("loss", result)


class TestNekoModelLifecycle(unittest.TestCase):
    """Tests for NekoModel lifecycle hooks without running a real Lightning training loop."""

    def _make_model(self):
        """Create a _SimpleModel with mocked trainer/logger so lifecycle hooks work."""
        model = _SimpleModel()
        # Mock trainer attributes
        model.trainer = MagicMock()
        model.trainer.global_step = 10
        model.trainer.log_every_n_steps = 1
        model.trainer.log_on_step = True
        model.trainer.log_on_epoch = True
        model.trainer.training = True
        # Mock logger
        model.trainer.logger = MagicMock()
        # Patch log/log_dict to avoid Lightning internals
        model.log = MagicMock()
        model.log_dict = MagicMock()
        return model

    # -- on_train_batch_end ---------------------------------------------------

    @patch.object(NekoModel, "log_on_training_step_end")
    def test_on_train_batch_end_accumulates_outputs(self, mock_step_log):
        model = self._make_model()
        outputs = {"loss": torch.tensor(0.5), "acc": torch.tensor(0.9)}
        # Patch super() call to avoid Lightning internals
        with patch("lightning.pytorch.LightningModule.on_train_batch_end"):
            model.on_train_batch_end(outputs, batch=None, batch_idx=0)
        self.assertEqual(len(model.training_step_outputs), 1)
        # Values should be detached (they already are scalars, but the hook calls .detach())
        stored = model.training_step_outputs[0]
        self.assertIn("loss", stored)
        self.assertIn("acc", stored)

    @patch.object(NekoModel, "log_on_training_step_end")
    def test_on_train_batch_end_calls_step_log_when_log_on_step(self, mock_step_log):
        model = self._make_model()
        model.trainer.log_on_step = True
        model.trainer.global_step = 10
        model.trainer.log_every_n_steps = 5  # 10 % 5 == 0 → should log
        outputs = {"loss": torch.tensor(0.3)}
        with patch("lightning.pytorch.LightningModule.on_train_batch_end"):
            model.on_train_batch_end(outputs, batch=None, batch_idx=0)
        mock_step_log.assert_called_once()

    # -- on_validation_batch_end ----------------------------------------------

    def test_on_validation_batch_end_accumulates_outputs(self):
        model = self._make_model()
        outputs = {"loss": torch.tensor(0.4), "acc": torch.tensor(0.85)}
        with patch("lightning.pytorch.LightningModule.on_validation_batch_end"):
            model.on_validation_batch_end(
                outputs, batch=None, batch_idx=0, dataloader_idx=0
            )
        self.assertEqual(len(model.validation_step_outputs), 1)
        stored = model.validation_step_outputs[0]
        self.assertIn("loss", stored)
        self.assertAlmostEqual(stored["loss"].item(), 0.4, places=5)

    # -- on_train_epoch_end ---------------------------------------------------

    @patch.object(NekoModel, "log_on_training_epoch_end")
    def test_on_train_epoch_end_logs_when_epoch_mode(self, mock_epoch_log):
        model = self._make_model()
        model.trainer.log_on_epoch = True
        # Pre-populate outputs
        model.training_step_outputs.append({"loss": torch.tensor(0.5)})
        model.training_step_outputs.append({"loss": torch.tensor(0.3)})
        with patch("lightning.pytorch.LightningModule.on_train_epoch_end"):
            model.on_train_epoch_end()
        mock_epoch_log.assert_called_once()
        # Outputs should be cleared after epoch end
        self.assertEqual(len(model.training_step_outputs), 0)

    # -- on_validation_epoch_end ----------------------------------------------

    @patch.object(NekoModel, "log_on_validation_epoch_end")
    def test_on_validation_epoch_end_calls_log(self, mock_val_log):
        model = self._make_model()
        model.validation_step_outputs.append({"loss": torch.tensor(0.2)})
        with patch("lightning.pytorch.LightningModule.on_validation_epoch_end"):
            model.on_validation_epoch_end()
        mock_val_log.assert_called_once()
        self.assertEqual(len(model.validation_step_outputs), 0)

    # -- log_on_training_step_end ---------------------------------------------

    def test_log_on_training_step_end_logs_metrics(self):
        model = self._make_model()
        output = {"loss": torch.tensor(0.5), "acc": torch.tensor(0.9)}
        model.log_on_training_step_end(output)
        logger = model.trainer.logger
        # logger.log_metrics should have been called for each key
        self.assertEqual(logger.log_metrics.call_count, 2)
        # history should have one entry
        self.assertEqual(len(model.history), 1)
        self.assertIn("loss", model.history[0])
        self.assertIn("acc", model.history[0])

    # -- log_on_training_epoch_end --------------------------------------------

    def test_log_on_training_epoch_end_averages_and_logs(self):
        model = self._make_model()
        outputs = [
            {"loss": torch.tensor(0.6), "acc": torch.tensor(0.8)},
            {"loss": torch.tensor(0.4), "acc": torch.tensor(0.9)},
        ]
        model.log_on_training_epoch_end(outputs)
        logger = model.trainer.logger
        # Should log each key once
        self.assertEqual(logger.log_metrics.call_count, 2)
        # History should have averaged values
        self.assertEqual(len(model.history), 1)
        self.assertAlmostEqual(model.history[0]["loss"].item(), 0.5, places=5)
        self.assertAlmostEqual(model.history[0]["acc"].item(), 0.85, places=5)

    # -- log_on_validation_epoch_end ------------------------------------------

    def test_log_on_validation_epoch_end_adds_val_prefix(self):
        model = self._make_model()
        # Must populate history with "loss" key first
        model.history.append({"loss": torch.tensor(0.5)})
        val_outputs = [
            {"loss": torch.tensor(0.3), "acc": torch.tensor(0.95)},
            {"loss": torch.tensor(0.2), "acc": torch.tensor(0.97)},
        ]
        model.log_on_validation_epoch_end(val_outputs)
        # history[-1] should now have val_loss and val_acc
        self.assertIn("val_loss", model.history[-1])
        self.assertIn("val_acc", model.history[-1])
        self.assertAlmostEqual(model.history[-1]["val_loss"].item(), 0.25, places=5)

    def test_log_on_validation_epoch_end_skips_when_no_history(self):
        model = self._make_model()
        # history is empty → should return early
        val_outputs = [{"loss": torch.tensor(0.3)}]
        model.log_on_validation_epoch_end(val_outputs)
        # logger should NOT have been called
        model.trainer.logger.log_metrics.assert_not_called()

    # -- log_image ------------------------------------------------------------

    def test_log_image(self):
        model = self._make_model()
        image = torch.rand(3, 32, 32)
        model.log_image("test_img", image)
        add_image = model.trainer.logger.experiment.add_image
        add_image.assert_called_once()
        args = add_image.call_args[0]
        self.assertEqual(args[0], "test_img")
        self.assertTrue(torch.equal(args[1], torch.clip(image, 0, 1)))
        self.assertEqual(args[2], 10)

    # -- log_histogram --------------------------------------------------------

    def test_log_histogram(self):
        model = self._make_model()
        values = torch.randn(100)
        model.log_histogram("test_hist", values)
        model.trainer.logger.experiment.add_histogram.assert_called_once_with(
            "test_hist", values, 10
        )

    # -- on_train_start -------------------------------------------------------

    def test_on_train_start_logs_graph(self):
        model = self._make_model()
        model.can_plot_graph = True
        model.example_input_array = torch.rand(1, 4)
        with patch("lightning.pytorch.LightningModule.on_train_start"):
            model.on_train_start()
        model.trainer.logger.log_graph.assert_called_once_with(
            model, model.example_input_array
        )

    def test_on_train_start_skips_when_no_graph(self):
        model = self._make_model()
        model.can_plot_graph = False
        with patch("lightning.pytorch.LightningModule.on_train_start"):
            model.on_train_start()
        model.trainer.logger.log_graph.assert_not_called()

    # -- on_test_batch_end ----------------------------------------------------

    def test_on_test_batch_end_calls_log_dict(self):
        model = self._make_model()
        outputs = {"loss": torch.tensor(0.4), "acc": torch.tensor(0.88)}
        with patch("lightning.pytorch.LightningModule.on_test_batch_end"):
            model.on_test_batch_end(outputs, batch=None, batch_idx=0, dataloader_idx=0)
        model.log_dict.assert_called_once()


class TestNekoTrainer(unittest.TestCase):
    def test_init_with_logger(self):
        trainer = NekoTrainer(logger="test", max_epochs=1, enable_checkpointing=False)
        self.assertIsInstance(trainer, Trainer)
        self.assertFalse(trainer.has_no_logger)

    def test_init_without_logger(self):
        trainer = NekoTrainer(logger=None, max_epochs=1, enable_checkpointing=False)
        self.assertTrue(trainer.has_no_logger)

    def test_log_mode_epoch(self):
        trainer = NekoTrainer(
            logger="test",
            max_epochs=1,
            log_every_n_steps=0,
            enable_checkpointing=False,
        )
        self.assertTrue(trainer.log_on_epoch)
        self.assertFalse(trainer.log_on_step)

    def test_log_mode_step(self):
        trainer = NekoTrainer(
            logger="test",
            max_epochs=1,
            log_every_n_steps=50,
            enable_checkpointing=False,
        )
        self.assertFalse(trainer.log_on_epoch)
        self.assertTrue(trainer.log_on_step)


class TestNekoTrainerBranches(unittest.TestCase):
    """Tests for NekoTrainer constructor branches and logger property."""

    def test_default_checkpoint_creation(self):
        """checkpointing=True, no callback → warning + ModelCheckpoint created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                trainer = NekoTrainer(
                    logger="test",
                    max_epochs=1,
                    enable_checkpointing=True,
                    callbacks=None,
                    default_root_dir=tmpdir,
                )
            # Should have emitted a warning about default checkpoint
            warn_msgs = [str(x.message) for x in w]
            self.assertTrue(
                any("Checkpoint callback is not defined" in m for m in warn_msgs),
                f"Expected checkpoint warning, got: {warn_msgs}",
            )
            # A ModelCheckpoint should be among the callbacks
            cp_cbs = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
            self.assertGreaterEqual(len(cp_cbs), 1)

    def test_user_checkpoint_callback(self):
        """checkpointing=True + explicit ModelCheckpoint → NilCallback appended."""
        with tempfile.TemporaryDirectory() as tmpdir:
            user_ckpt = ModelCheckpoint(dirpath=tmpdir, monitor="val_loss")
            trainer = NekoTrainer(
                logger="test",
                max_epochs=1,
                enable_checkpointing=True,
                callbacks=[user_ckpt],
                default_root_dir=tmpdir,
            )
            # User checkpoint must be present
            self.assertIn(user_ckpt, trainer.callbacks)

    def test_error_checkpointing_false_with_checkpoint_callback(self):
        """checkpointing=False + ModelCheckpoint → ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            user_ckpt = ModelCheckpoint(dirpath=tmpdir, monitor="val_loss")
            with self.assertRaises(ValueError):
                NekoTrainer(
                    logger="test",
                    max_epochs=1,
                    enable_checkpointing=False,
                    callbacks=[user_ckpt],
                    default_root_dir=tmpdir,
                )

    def test_default_metric_callbacks_present(self):
        """LrLogger, EpochNumLogger, EpochTimeLogger always present."""
        from tensorneko.callback import LrLogger, EpochNumLogger, EpochTimeLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = NekoTrainer(
                logger="test",
                max_epochs=1,
                enable_checkpointing=False,
                default_root_dir=tmpdir,
            )
            cb_types = [type(c) for c in trainer.callbacks]
            self.assertIn(LrLogger, cb_types)
            self.assertIn(EpochNumLogger, cb_types)
            self.assertIn(EpochTimeLogger, cb_types)

    def test_logger_property_training_true(self):
        """training=True → logger returns logger_train."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = NekoTrainer(
                logger="test",
                max_epochs=1,
                enable_checkpointing=False,
                default_root_dir=tmpdir,
            )
            trainer.training = True
            self.assertIs(trainer.logger, trainer.logger_train)

    def test_logger_property_training_false(self):
        """training=False → logger returns logger_val."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = NekoTrainer(
                logger="test",
                max_epochs=1,
                enable_checkpointing=False,
                default_root_dir=tmpdir,
            )
            trainer.training = False
            self.assertIs(trainer.logger, trainer.logger_val)

    def test_logger_property_no_logger(self):
        """has_no_logger=True → logger returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = NekoTrainer(
                logger=None,
                max_epochs=1,
                enable_checkpointing=False,
                default_root_dir=tmpdir,
            )
            self.assertTrue(trainer.has_no_logger)
            self.assertIsNone(trainer.logger)


class TestArchBranches(unittest.TestCase):
    """Additional branch coverage for architecture classes."""

    # -- VQVAE predict_step ----------------------------------------------------

    def test_vqvae_predict_step(self):
        model = _TinyVQVAE(latent_dim=4, n_embeddings=8)
        x = torch.rand(2, 1, 4, 4)
        out = model.predict_step(x, batch_idx=0)
        # predict_step returns only x_hat (not the embedding_loss)
        self.assertEqual(out.shape, (2, 1, 4, 4))
        self.assertIsInstance(out, torch.Tensor)

    # -- BinaryClassifier predict_step -----------------------------------------

    def test_binary_classifier_predict_step(self):
        inner = torch.nn.Linear(8, 1)
        model = BinaryClassifier("bc", model=inner, learning_rate=1e-3)
        batch = (torch.rand(4, 8), torch.randint(0, 2, (4,)).float())
        out = model.predict_step(batch, batch_idx=0)
        self.assertEqual(out.shape, (4, 1))

    # -- GAN training_step with mocked optimizers ------------------------------

    def test_gan_training_step(self):
        model = _TinyGAN(latent_dim=4, g_learning_rate=1e-3, d_learning_rate=1e-3)
        # Mock optimizers: training_step calls self.optimizers() → (d_opt, g_opt)
        mock_d_opt = MagicMock()
        mock_g_opt = MagicMock()
        model.optimizers = MagicMock(return_value=(mock_d_opt, mock_g_opt))
        model.lr_schedulers = MagicMock(return_value=None)
        model.manual_backward = MagicMock()

        batch = (torch.rand(2, 8),)
        result = model.training_step(batch, batch_idx=0)

        self.assertIn("loss", result)
        self.assertIn("loss/d_loss", result)
        self.assertIn("loss/g_loss", result)
        # Verify optimizers were used
        mock_d_opt.zero_grad.assert_called()
        mock_d_opt.step.assert_called()
        mock_g_opt.zero_grad.assert_called()
        mock_g_opt.step.assert_called()
        # manual_backward should have been called (once for d, once for g)
        self.assertEqual(model.manual_backward.call_count, 2)

    # -- WGAN gradient_penalty_fn direct test ----------------------------------

    def test_wgan_gradient_penalty_fn(self):
        model = _TinyWGAN(
            latent_dim=4,
            g_learning_rate=1e-3,
            d_learning_rate=1e-3,
            gp_weight=10.0,
        )
        model.train()
        x_real = torch.rand(2, 1, 2, 2, requires_grad=False)
        x_fake = torch.rand(2, 1, 2, 2, requires_grad=False)
        gp = model.gradient_penalty_fn(2, x_real, x_fake)
        self.assertEqual(gp.dim(), 0)  # scalar
        self.assertGreaterEqual(gp.item(), 0.0)

    # -- AutoEncoder forward with tuple vs bare tensor -------------------------

    def test_autoencoder_training_step_bare_tensor(self):
        model = _TinyAE("ae", learning_rate=1e-3)
        x = torch.rand(2, 8)
        result = model.training_step(x)  # bare tensor, not wrapped in tuple
        self.assertIn("loss", result)
        self.assertEqual(result["loss"].dim(), 0)

    def test_autoencoder_training_step_tuple(self):
        model = _TinyAE("ae", learning_rate=1e-3)
        batch = (torch.rand(2, 8),)
        result = model.training_step(batch)  # tuple wrapping
        self.assertIn("loss", result)
        self.assertEqual(result["loss"].dim(), 0)


if __name__ == "__main__":
    unittest.main()
