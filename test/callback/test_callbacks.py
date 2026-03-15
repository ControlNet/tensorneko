"""Tests for tensorneko callbacks."""

import unittest
from unittest.mock import MagicMock, patch

from lightning.pytorch import Callback


class TestEarlyStoppingLR(unittest.TestCase):
    def test_init_valid_modes(self):
        from tensorneko.callback.earlystop_lr import EarlyStoppingLR

        cb = EarlyStoppingLR(lr_threshold=1e-6, mode="all")
        self.assertEqual(cb.lr_threshold, 1e-6)
        self.assertEqual(cb.mode, "all")

        cb2 = EarlyStoppingLR(lr_threshold=1e-5, mode="any")
        self.assertEqual(cb2.mode, "any")

    def test_init_invalid_mode_raises(self):
        from tensorneko.callback.earlystop_lr import EarlyStoppingLR

        with self.assertRaises(ValueError):
            EarlyStoppingLR(lr_threshold=1e-6, mode="invalid")

    def test_early_stop_all_below_threshold(self):
        from tensorneko.callback.earlystop_lr import EarlyStoppingLR

        cb = EarlyStoppingLR(lr_threshold=1e-6, mode="all")
        trainer = MagicMock()
        trainer.should_stop = False
        trainer.callback_metrics = {
            "learning_rate/opt0_lr0": 1e-7,
            "learning_rate/opt1_lr0": 1e-8,
        }
        cb.on_train_epoch_start(trainer, MagicMock())
        self.assertTrue(trainer.should_stop)

    def test_no_early_stop_above_threshold(self):
        from tensorneko.callback.earlystop_lr import EarlyStoppingLR

        cb = EarlyStoppingLR(lr_threshold=1e-6, mode="all")
        trainer = MagicMock()
        trainer.should_stop = False
        trainer.callback_metrics = {
            "learning_rate/opt0_lr0": 1e-3,
        }
        cb.on_train_epoch_start(trainer, MagicMock())
        self.assertFalse(trainer.should_stop)

    def test_early_stop_any_mode(self):
        from tensorneko.callback.earlystop_lr import EarlyStoppingLR

        cb = EarlyStoppingLR(lr_threshold=1e-6, mode="any")
        trainer = MagicMock()
        trainer.should_stop = False
        trainer.callback_metrics = {
            "learning_rate/opt0_lr0": 1e-3,
            "learning_rate/opt1_lr0": 1e-8,  # below threshold
        }
        cb.on_train_epoch_start(trainer, MagicMock())
        self.assertTrue(trainer.should_stop)

    def test_empty_metrics_no_crash(self):
        from tensorneko.callback.earlystop_lr import EarlyStoppingLR

        cb = EarlyStoppingLR(lr_threshold=1e-6, mode="all")
        trainer = MagicMock()
        trainer.should_stop = False
        trainer.callback_metrics = {}
        cb.on_train_epoch_start(trainer, MagicMock())
        self.assertFalse(trainer.should_stop)

    def test_no_lr_metrics_no_crash(self):
        from tensorneko.callback.earlystop_lr import EarlyStoppingLR

        cb = EarlyStoppingLR(lr_threshold=1e-6, mode="all")
        trainer = MagicMock()
        trainer.should_stop = False
        trainer.callback_metrics = {"loss": 0.5}
        cb.on_train_epoch_start(trainer, MagicMock())
        self.assertFalse(trainer.should_stop)


class TestSystemStatsLogger(unittest.TestCase):
    def setUp(self):
        try:
            import psutil
            from tensorneko.callback.system_stats_logger import SystemStatsLogger

            self.SystemStatsLogger = SystemStatsLogger
        except ImportError:
            self.skipTest("psutil not installed")

    def test_init(self):
        cb = self.SystemStatsLogger(on_epoch=True, on_step=False)
        self.assertTrue(cb.on_epoch)
        self.assertFalse(cb.on_step)
        self.assertIsNotNone(cb.psutil)

    def test_init_both_false_raises(self):
        with self.assertRaises(ValueError):
            self.SystemStatsLogger(on_epoch=False, on_step=False)

    def test_on_train_epoch_end(self):
        cb = self.SystemStatsLogger(on_epoch=True)
        trainer = MagicMock()
        trainer.global_step = 10
        pl_module = MagicMock()
        pl_module.distributed = False
        cb.on_train_epoch_end(trainer, pl_module)
        pl_module.logger.log_metrics.assert_called_once()
        logged = pl_module.logger.log_metrics.call_args[0][0]
        self.assertIn("system/cpu_usage_epoch", logged)
        self.assertIn("system/memory_usage_epoch", logged)

    def test_on_train_epoch_end_disabled(self):
        cb = self.SystemStatsLogger(on_epoch=False, on_step=True)
        trainer = MagicMock()
        pl_module = MagicMock()
        cb.on_train_epoch_end(trainer, pl_module)
        pl_module.logger.log_metrics.assert_not_called()

    def test_on_train_batch_end(self):
        cb = self.SystemStatsLogger(on_epoch=False, on_step=True)
        trainer = MagicMock()
        trainer.global_step = 5
        pl_module = MagicMock()
        pl_module.distributed = False
        cb.on_train_batch_end(trainer, pl_module, {}, None, 0)
        pl_module.logger.log_metrics.assert_called_once()

    def test_on_train_batch_end_disabled(self):
        cb = self.SystemStatsLogger(on_epoch=True, on_step=False)
        trainer = MagicMock()
        pl_module = MagicMock()
        cb.on_train_batch_end(trainer, pl_module, {}, None, 0)
        pl_module.logger.log_metrics.assert_not_called()


class TestGpuStatsLogger(unittest.TestCase):
    def test_import_error_without_gpumonitor(self):
        """GpuStatsLogger raises ImportError if gpumonitor not installed."""
        try:
            from tensorneko.callback.gpu_stats_logger import GpuStatsLogger

            # If it imports without error, gpumonitor is available
            # Just verify it's a Callback subclass
            self.assertTrue(issubclass(GpuStatsLogger, Callback))
        except ImportError:
            pass  # Expected if gpumonitor is not installed


class TestEpochLoggers(unittest.TestCase):
    def test_epoch_num_logger(self):
        from tensorneko.callback.epoch_num_logger import EpochNumLogger

        cb = EpochNumLogger()
        trainer = MagicMock()
        trainer.current_epoch = 5
        pl_module = MagicMock()
        pl_module.distributed = False
        cb.on_train_epoch_start(trainer, pl_module)
        pl_module.log.assert_called_once()
        call_args = pl_module.log.call_args
        self.assertEqual(call_args[0][0], "epoch/epoch")
        self.assertEqual(call_args[0][1], 5)

    def test_epoch_time_logger(self):
        import time
        from tensorneko.callback.epoch_time_logger import EpochTimeLogger

        cb = EpochTimeLogger()
        trainer = MagicMock()
        pl_module = MagicMock()
        pl_module.distributed = False
        cb.on_train_epoch_start(trainer, pl_module)
        time.sleep(0.01)  # Small delay
        cb.on_train_epoch_end(trainer, pl_module)
        pl_module.log.assert_called_once()
        call_args = pl_module.log.call_args
        self.assertEqual(call_args[0][0], "epoch/epoch_time")
        self.assertGreater(call_args[0][1], 0)

    def test_display_metrics_callback(self):
        from tensorneko.callback.display_metrics_callback import DisplayMetricsCallback

        self.assertTrue(issubclass(DisplayMetricsCallback, Callback))

    def test_lr_logger(self):
        from tensorneko.callback.lr_logger import LrLogger

        cb = LrLogger()
        self.assertIsInstance(cb, Callback)


class TestDisplayMetricsCallback(unittest.TestCase):
    def test_on_validation_epoch_end_prints_metrics(self):
        """Test that DisplayMetricsCallback prints trainer callback metrics."""
        from tensorneko.callback.display_metrics_callback import DisplayMetricsCallback

        cb = DisplayMetricsCallback()
        trainer = MagicMock()
        trainer.callback_metrics = {"loss": 0.5, "acc": 0.9}
        pl_module = MagicMock()

        with patch("builtins.print") as mock_print:
            cb.on_validation_epoch_end(trainer, pl_module)
            mock_print.assert_called_once_with({"loss": 0.5, "acc": 0.9})

    def test_on_validation_epoch_end_empty_metrics(self):
        """Test DisplayMetricsCallback with empty metrics."""
        from tensorneko.callback.display_metrics_callback import DisplayMetricsCallback

        cb = DisplayMetricsCallback()
        trainer = MagicMock()
        trainer.callback_metrics = {}
        pl_module = MagicMock()

        with patch("builtins.print") as mock_print:
            cb.on_validation_epoch_end(trainer, pl_module)
            mock_print.assert_called_once_with({})

    def test_display_metrics_is_callback(self):
        """Test that DisplayMetricsCallback is a proper Callback subclass."""
        from tensorneko.callback.display_metrics_callback import DisplayMetricsCallback

        cb = DisplayMetricsCallback()
        self.assertIsInstance(cb, Callback)


class TestSystemStatsLoggerMocked(unittest.TestCase):
    """Test SystemStatsLogger with full mocking - no optional deps required."""

    def test_init(self):
        """Test initialization with mocked psutil."""
        import sys
        import importlib
        from unittest.mock import MagicMock

        # Create mock psutil module
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 50.0
        mock_vm = MagicMock()
        mock_vm.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_vm

        # Mock psutil at module level
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            # Reload to pick up mock
            if "tensorneko.callback.system_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.system_stats_logger"]
            from tensorneko.callback.system_stats_logger import SystemStatsLogger

            importlib.reload(sys.modules["tensorneko.callback.system_stats_logger"])

            cb = SystemStatsLogger(on_epoch=True, on_step=False)
            self.assertTrue(cb.on_epoch)
            self.assertFalse(cb.on_step)
            self.assertIsNotNone(cb.psutil)

    def test_init_both_false_raises(self):
        """Test that both False raises ValueError."""
        import sys
        from unittest.mock import MagicMock

        mock_psutil = MagicMock()
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            if "tensorneko.callback.system_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.system_stats_logger"]
            from tensorneko.callback.system_stats_logger import SystemStatsLogger

            with self.assertRaises(ValueError):
                SystemStatsLogger(on_epoch=False, on_step=False)

    def test_on_train_epoch_end_logs_stats(self):
        """Test on_train_epoch_end logs CPU and memory stats."""
        import sys
        from unittest.mock import MagicMock

        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 50.0
        mock_vm = MagicMock()
        mock_vm.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_vm

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            if "tensorneko.callback.system_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.system_stats_logger"]
            from tensorneko.callback.system_stats_logger import SystemStatsLogger

            cb = SystemStatsLogger(on_epoch=True)
            trainer = MagicMock()
            trainer.global_step = 10
            pl_module = MagicMock()
            pl_module.distributed = False

            cb.on_train_epoch_end(trainer, pl_module)

            pl_module.logger.log_metrics.assert_called_once()
            logged = pl_module.logger.log_metrics.call_args[0][0]
            self.assertIn("system/cpu_usage_epoch", logged)
            self.assertIn("system/memory_usage_epoch", logged)
            self.assertEqual(logged["system/cpu_usage_epoch"], 50.0)
            self.assertEqual(logged["system/memory_usage_epoch"], 60.0)

    def test_on_train_epoch_end_disabled(self):
        """Test on_train_epoch_end does not log when on_epoch=False."""
        import sys
        from unittest.mock import MagicMock

        mock_psutil = MagicMock()
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            if "tensorneko.callback.system_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.system_stats_logger"]
            from tensorneko.callback.system_stats_logger import SystemStatsLogger

            cb = SystemStatsLogger(on_epoch=False, on_step=True)
            trainer = MagicMock()
            pl_module = MagicMock()

            cb.on_train_epoch_end(trainer, pl_module)
            pl_module.logger.log_metrics.assert_not_called()

    def test_on_train_batch_end_logs_stats(self):
        """Test on_train_batch_end logs CPU and memory stats."""
        import sys
        from unittest.mock import MagicMock

        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 50.0
        mock_vm = MagicMock()
        mock_vm.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_vm

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            if "tensorneko.callback.system_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.system_stats_logger"]
            from tensorneko.callback.system_stats_logger import SystemStatsLogger

            cb = SystemStatsLogger(on_epoch=False, on_step=True)
            trainer = MagicMock()
            trainer.global_step = 5
            pl_module = MagicMock()
            pl_module.distributed = False

            cb.on_train_batch_end(trainer, pl_module, {}, None, 0)

            pl_module.logger.log_metrics.assert_called_once()
            logged = pl_module.logger.log_metrics.call_args[0][0]
            self.assertIn("system/cpu_usage_step", logged)
            self.assertIn("system/memory_usage_step", logged)
            self.assertEqual(logged["system/cpu_usage_step"], 50.0)
            self.assertEqual(logged["system/memory_usage_step"], 60.0)

    def test_on_train_batch_end_disabled(self):
        """Test on_train_batch_end does not log when on_step=False."""
        import sys
        from unittest.mock import MagicMock

        mock_psutil = MagicMock()
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            if "tensorneko.callback.system_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.system_stats_logger"]
            from tensorneko.callback.system_stats_logger import SystemStatsLogger

            cb = SystemStatsLogger(on_epoch=True, on_step=False)
            trainer = MagicMock()
            pl_module = MagicMock()

            cb.on_train_batch_end(trainer, pl_module, {}, None, 0)
            pl_module.logger.log_metrics.assert_not_called()


class TestGpuStatsLoggerMocked(unittest.TestCase):
    """Test GpuStatsLogger with full mocking - no optional deps required."""

    def test_init(self):
        """Test initialization with mocked gpumonitor."""
        import sys
        from unittest.mock import MagicMock

        # Create mock gpumonitor module
        mock_gpumonitor = MagicMock()
        mock_monitor_class = MagicMock()
        mock_gpumonitor.monitor.GPUStatMonitor = mock_monitor_class

        with patch.dict(
            "sys.modules",
            {
                "gpumonitor": mock_gpumonitor,
                "gpumonitor.monitor": mock_gpumonitor.monitor,
            },
        ):
            if "tensorneko.callback.gpu_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.gpu_stats_logger"]
            from tensorneko.callback.gpu_stats_logger import GpuStatsLogger

            cb = GpuStatsLogger(delay=0.5, on_epoch=True, on_step=False)
            self.assertTrue(cb.on_epoch)
            self.assertFalse(cb.on_step)
            self.assertIsNotNone(cb.monitor_epoch)
            self.assertIsNone(cb.monitor_step)

    def test_init_both_false_raises(self):
        """Test that both False raises ValueError."""
        import sys
        from unittest.mock import MagicMock

        mock_gpumonitor = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "gpumonitor": mock_gpumonitor,
                "gpumonitor.monitor": mock_gpumonitor.monitor,
            },
        ):
            if "tensorneko.callback.gpu_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.gpu_stats_logger"]
            from tensorneko.callback.gpu_stats_logger import GpuStatsLogger

            with self.assertRaises(ValueError):
                GpuStatsLogger(on_epoch=False, on_step=False)

    def test_on_train_epoch_start_resets_monitor(self):
        """Test on_train_epoch_start calls monitor.reset()."""
        import sys
        from unittest.mock import MagicMock

        mock_gpumonitor = MagicMock()
        mock_monitor_instance = MagicMock()
        mock_gpumonitor.monitor.GPUStatMonitor.return_value = mock_monitor_instance

        with patch.dict(
            "sys.modules",
            {
                "gpumonitor": mock_gpumonitor,
                "gpumonitor.monitor": mock_gpumonitor.monitor,
            },
        ):
            if "tensorneko.callback.gpu_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.gpu_stats_logger"]
            from tensorneko.callback.gpu_stats_logger import GpuStatsLogger

            cb = GpuStatsLogger(on_epoch=True)
            trainer = MagicMock()
            pl_module = MagicMock()

            cb.on_train_epoch_start(trainer, pl_module)
            cb.monitor_epoch.reset.assert_called_once()

    def test_on_train_epoch_end_logs_gpu_stats(self):
        """Test on_train_epoch_end logs GPU stats when on_epoch=True."""
        import sys
        from unittest.mock import MagicMock

        # Create mock GPU object with required attributes
        mock_gpu = MagicMock()
        mock_gpu.index = 0
        mock_gpu.memory_used = 4000
        mock_gpu.memory_total = 8000
        mock_gpu.temperature = 65
        mock_gpu.utilization = 80
        mock_gpu.power_draw = 150
        mock_gpu.power_limit = 250
        mock_gpu.fan_speed = 50

        mock_gpumonitor = MagicMock()
        mock_monitor_instance = MagicMock()
        mock_avg_stats = MagicMock()
        mock_avg_stats.gpus = [mock_gpu]
        mock_monitor_instance.average_stats = mock_avg_stats
        mock_gpumonitor.monitor.GPUStatMonitor.return_value = mock_monitor_instance

        with patch.dict(
            "sys.modules",
            {
                "gpumonitor": mock_gpumonitor,
                "gpumonitor.monitor": mock_gpumonitor.monitor,
            },
        ):
            if "tensorneko.callback.gpu_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.gpu_stats_logger"]
            from tensorneko.callback.gpu_stats_logger import GpuStatsLogger

            cb = GpuStatsLogger(on_epoch=True, on_step=False)
            trainer = MagicMock()
            trainer.global_step = 10
            pl_module = MagicMock()
            pl_module.distributed = False

            cb.on_train_epoch_end(trainer, pl_module)

            pl_module.logger.log_metrics.assert_called_once()
            logged = pl_module.logger.log_metrics.call_args[0][0]
            self.assertIn("system/gpu0_memory_used_epoch", logged)
            self.assertIn("system/gpu0_temperature_epoch", logged)
            self.assertIn("system/gpu0_utilization_epoch", logged)
            self.assertEqual(logged["system/gpu0_memory_used_epoch"], 4000 / 1024)
            self.assertEqual(logged["system/gpu0_temperature_epoch"], 65)

    def test_on_train_epoch_end_disabled(self):
        """Test on_train_epoch_end does not log when on_epoch=False."""
        import sys
        from unittest.mock import MagicMock

        mock_gpumonitor = MagicMock()
        mock_gpumonitor.monitor.GPUStatMonitor.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "gpumonitor": mock_gpumonitor,
                "gpumonitor.monitor": mock_gpumonitor.monitor,
            },
        ):
            if "tensorneko.callback.gpu_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.gpu_stats_logger"]
            from tensorneko.callback.gpu_stats_logger import GpuStatsLogger

            cb = GpuStatsLogger(on_epoch=False, on_step=True)
            trainer = MagicMock()
            pl_module = MagicMock()

            cb.on_train_epoch_end(trainer, pl_module)
            pl_module.logger.log_metrics.assert_not_called()

    def test_on_train_batch_start_resets_monitor(self):
        """Test on_train_batch_start calls monitor.reset() when on_step=True."""
        import sys
        from unittest.mock import MagicMock

        mock_gpumonitor = MagicMock()
        mock_monitor_instance = MagicMock()
        mock_gpumonitor.monitor.GPUStatMonitor.return_value = mock_monitor_instance

        with patch.dict(
            "sys.modules",
            {
                "gpumonitor": mock_gpumonitor,
                "gpumonitor.monitor": mock_gpumonitor.monitor,
            },
        ):
            if "tensorneko.callback.gpu_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.gpu_stats_logger"]
            from tensorneko.callback.gpu_stats_logger import GpuStatsLogger

            cb = GpuStatsLogger(on_epoch=False, on_step=True)
            trainer = MagicMock()
            pl_module = MagicMock()

            cb.on_train_batch_start(trainer, pl_module, None, 0)
            cb.monitor_step.reset.assert_called_once()

    def test_on_train_batch_end_logs_gpu_stats(self):
        """Test on_train_batch_end logs GPU stats when on_step=True."""
        import sys
        from unittest.mock import MagicMock

        # Create mock GPU object
        mock_gpu = MagicMock()
        mock_gpu.index = 0
        mock_gpu.memory_used = 4000
        mock_gpu.memory_total = 8000
        mock_gpu.temperature = 65
        mock_gpu.utilization = 80
        mock_gpu.power_draw = 150
        mock_gpu.power_limit = 250
        mock_gpu.fan_speed = 50

        mock_gpumonitor = MagicMock()
        mock_monitor_instance = MagicMock()
        mock_avg_stats = MagicMock()
        mock_avg_stats.gpus = [mock_gpu]
        mock_monitor_instance.average_stats = mock_avg_stats
        mock_gpumonitor.monitor.GPUStatMonitor.return_value = mock_monitor_instance

        with patch.dict(
            "sys.modules",
            {
                "gpumonitor": mock_gpumonitor,
                "gpumonitor.monitor": mock_gpumonitor.monitor,
            },
        ):
            if "tensorneko.callback.gpu_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.gpu_stats_logger"]
            from tensorneko.callback.gpu_stats_logger import GpuStatsLogger

            cb = GpuStatsLogger(on_epoch=False, on_step=True)
            trainer = MagicMock()
            trainer.global_step = 5
            pl_module = MagicMock()
            pl_module.distributed = False

            cb.on_train_batch_end(trainer, pl_module, {}, None, 0)

            pl_module.logger.log_metrics.assert_called_once()
            logged = pl_module.logger.log_metrics.call_args[0][0]
            self.assertIn("system/gpu0_memory_used_step", logged)
            self.assertIn("system/gpu0_temperature_step", logged)
            self.assertEqual(logged["system/gpu0_memory_used_step"], 4000 / 1024)

    def test_on_train_batch_end_disabled(self):
        """Test on_train_batch_end does not log when on_step=False."""
        import sys
        from unittest.mock import MagicMock

        mock_gpumonitor = MagicMock()
        mock_gpumonitor.monitor.GPUStatMonitor.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "gpumonitor": mock_gpumonitor,
                "gpumonitor.monitor": mock_gpumonitor.monitor,
            },
        ):
            if "tensorneko.callback.gpu_stats_logger" in sys.modules:
                del sys.modules["tensorneko.callback.gpu_stats_logger"]
            from tensorneko.callback.gpu_stats_logger import GpuStatsLogger

            cb = GpuStatsLogger(on_epoch=True, on_step=False)
            trainer = MagicMock()
            pl_module = MagicMock()

            cb.on_train_batch_end(trainer, pl_module, {}, None, 0)
            pl_module.logger.log_metrics.assert_not_called()
