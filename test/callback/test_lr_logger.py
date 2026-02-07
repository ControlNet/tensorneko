"""Tests for LrLogger callback."""

import unittest
from unittest.mock import MagicMock


class TestLrLogger(unittest.TestCase):
    def test_single_optimizer_single_param_group(self):
        """Test logging with single optimizer and single param group."""
        from tensorneko.callback.lr_logger import LrLogger

        cb = LrLogger()
        trainer = MagicMock()
        trainer.global_step = 10

        # Mock optimizer with single param group
        mock_opt = MagicMock()
        mock_opt.param_groups = [{"lr": 0.001}]
        trainer.optimizers = [mock_opt]

        # Mock pl_module
        pl_module = MagicMock()
        pl_module.distributed = False

        # Call the callback
        cb.on_train_epoch_start(trainer, pl_module)

        # Verify pl_module.log was called with correct key
        pl_module.log.assert_called_once_with(
            "learning_rate/opt0_lr0", 0.001, logger=False, sync_dist=False
        )

        # Verify pl_module.logger.log_metrics was called
        pl_module.logger.log_metrics.assert_called_once_with(
            {"learning_rate/opt0_lr0": 0.001}, step=10
        )

    def test_single_optimizer_multiple_param_groups(self):
        """Test logging with single optimizer and multiple param groups."""
        from tensorneko.callback.lr_logger import LrLogger

        cb = LrLogger()
        trainer = MagicMock()
        trainer.global_step = 20

        # Mock optimizer with multiple param groups
        mock_opt = MagicMock()
        mock_opt.param_groups = [{"lr": 0.001}, {"lr": 0.0001}]
        trainer.optimizers = [mock_opt]

        # Mock pl_module
        pl_module = MagicMock()
        pl_module.distributed = False

        # Call the callback
        cb.on_train_epoch_start(trainer, pl_module)

        # Verify both learning rates were logged
        self.assertEqual(pl_module.log.call_count, 2)
        self.assertEqual(pl_module.logger.log_metrics.call_count, 2)

        # Verify the calls
        calls = pl_module.log.call_args_list
        self.assertEqual(calls[0][0], ("learning_rate/opt0_lr0", 0.001))
        self.assertEqual(calls[1][0], ("learning_rate/opt0_lr1", 0.0001))

    def test_multiple_optimizers(self):
        """Test logging with multiple optimizers."""
        from tensorneko.callback.lr_logger import LrLogger

        cb = LrLogger()
        trainer = MagicMock()
        trainer.global_step = 30

        # Mock multiple optimizers
        mock_opt1 = MagicMock()
        mock_opt1.param_groups = [{"lr": 0.001}]

        mock_opt2 = MagicMock()
        mock_opt2.param_groups = [{"lr": 0.0005}]

        trainer.optimizers = [mock_opt1, mock_opt2]

        # Mock pl_module
        pl_module = MagicMock()
        pl_module.distributed = False

        # Call the callback
        cb.on_train_epoch_start(trainer, pl_module)

        # Verify both optimizers' learning rates were logged
        self.assertEqual(pl_module.log.call_count, 2)
        self.assertEqual(pl_module.logger.log_metrics.call_count, 2)

        # Verify the calls
        calls = pl_module.log.call_args_list
        self.assertEqual(calls[0][0], ("learning_rate/opt0_lr0", 0.001))
        self.assertEqual(calls[1][0], ("learning_rate/opt1_lr0", 0.0005))

    def test_multiple_optimizers_multiple_param_groups(self):
        """Test logging with multiple optimizers and multiple param groups."""
        from tensorneko.callback.lr_logger import LrLogger

        cb = LrLogger()
        trainer = MagicMock()
        trainer.global_step = 40

        # Mock multiple optimizers with multiple param groups
        mock_opt1 = MagicMock()
        mock_opt1.param_groups = [{"lr": 0.001}, {"lr": 0.0001}]

        mock_opt2 = MagicMock()
        mock_opt2.param_groups = [{"lr": 0.0005}, {"lr": 0.00005}]

        trainer.optimizers = [mock_opt1, mock_opt2]

        # Mock pl_module
        pl_module = MagicMock()
        pl_module.distributed = False

        # Call the callback
        cb.on_train_epoch_start(trainer, pl_module)

        # Verify all learning rates were logged
        self.assertEqual(pl_module.log.call_count, 4)
        self.assertEqual(pl_module.logger.log_metrics.call_count, 4)

        # Verify the keys
        calls = pl_module.log.call_args_list
        self.assertEqual(calls[0][0][0], "learning_rate/opt0_lr0")
        self.assertEqual(calls[1][0][0], "learning_rate/opt0_lr1")
        self.assertEqual(calls[2][0][0], "learning_rate/opt1_lr0")
        self.assertEqual(calls[3][0][0], "learning_rate/opt1_lr1")

        # Verify global_step was passed correctly
        log_metrics_calls = pl_module.logger.log_metrics.call_args_list
        for call in log_metrics_calls:
            self.assertEqual(call[1]["step"], 40)

    def test_distributed_flag(self):
        """Test that distributed flag is passed correctly."""
        from tensorneko.callback.lr_logger import LrLogger

        cb = LrLogger()
        trainer = MagicMock()
        trainer.global_step = 50

        mock_opt = MagicMock()
        mock_opt.param_groups = [{"lr": 0.001}]
        trainer.optimizers = [mock_opt]

        # Mock pl_module with distributed=True
        pl_module = MagicMock()
        pl_module.distributed = True

        # Call the callback
        cb.on_train_epoch_start(trainer, pl_module)

        # Verify sync_dist flag is True
        pl_module.log.assert_called_once_with(
            "learning_rate/opt0_lr0", 0.001, logger=False, sync_dist=True
        )
