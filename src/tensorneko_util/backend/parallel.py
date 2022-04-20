from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor, Future
from enum import Enum
from os import cpu_count
from typing import Optional, Dict, Union, Tuple

_num_workers = cpu_count() // 2


def set_num_workers(num_workers):
    global _num_workers
    _num_workers = num_workers


class ParallelType(Enum):
    THREAD = "thread"
    PROCESS = "process"


class ExecutorPool:
    thread_pools: Dict[str, ThreadPoolExecutor] = {}
    process_pools: Dict[str, ProcessPoolExecutor] = {}

    @classmethod
    def get_thread_executor(cls, name: Optional[str] = None, num_workers: int = _num_workers) -> ThreadPoolExecutor:
        name = name or "default"
        if name not in cls.thread_pools:
            cls.thread_pools[name] = ThreadPoolExecutor(num_workers)
        return cls.thread_pools[name]

    @classmethod
    def get_process_executor(cls, name: Optional[str] = None, num_workers: int = _num_workers) -> ProcessPoolExecutor:
        name = name or "default"
        if name not in cls.process_pools:
            cls.process_pools[name] = ProcessPoolExecutor(num_workers)
        return cls.process_pools[name]

    @classmethod
    def get(cls, name: Optional[str] = None, num_workers=_num_workers,
        parallel_type: ParallelType = ParallelType.PROCESS
    ) -> Executor:
        if parallel_type == ParallelType.THREAD:
            return cls.get_thread_executor(name, num_workers)
        elif parallel_type == ParallelType.PROCESS:
            return cls.get_process_executor(name, num_workers)
        else:
            raise ValueError("Invalid parallel type: {}".format(parallel_type))

    @classmethod
    def __class_getitem__(cls, item: Union[str, Tuple[str, ParallelType]]) -> Executor:
        if isinstance(item, str):
            if item in cls.thread_pools and item in cls.process_pools:
                raise ValueError("Ambiguous indexing. Both thread and process pool {} found".format(item))
            elif item in cls.thread_pools:
                return cls.thread_pools[item]
            elif item in cls.process_pools:
                return cls.process_pools[item]
            else:
                raise ValueError("Executor {} not found".format(item))
        elif isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError("Invalid indexing. The format should be (name, parallel_type)")

            name, parallel_type = item
            if parallel_type == ParallelType.THREAD:
                return cls.get_thread_executor(name)
            elif parallel_type == ParallelType.PROCESS:
                return cls.get_process_executor(name)
            else:
                raise ValueError("Invalid parallel type: {}".format(parallel_type))
        else:
            raise ValueError("Invalid indexing. The format should be (name, parallel_type)")

    @classmethod
    def shutdown(cls, name: Optional[str] = None, parallel_type: Optional[ParallelType] = ParallelType.PROCESS) -> None:
        name = name or "default"
        if parallel_type == ParallelType.THREAD:
            if name in cls.thread_pools:
                cls.thread_pools[name].shutdown()
                del cls.thread_pools[name]
            else:
                raise ValueError("Thread pool {} not found".format(name))
        elif parallel_type == ParallelType.PROCESS:
            if name in cls.process_pools:
                cls.process_pools[name].shutdown()
                del cls.process_pools[name]
            else:
                raise ValueError("Process pool {} not found".format(name))
        else:
            raise ValueError("Invalid parallel type: {}".format(parallel_type))

    @classmethod
    def shutdown_all(cls):
        for name in cls.thread_pools:
            cls.thread_pools[name].shutdown()
        for name in cls.process_pools:
            cls.process_pools[name].shutdown()
        cls.thread_pools = {}
        cls.process_pools = {}

    @classmethod
    def submit(cls, func, *args, name: Optional[str] = None,
        parallel_type: Optional[ParallelType] = ParallelType.PROCESS, **kwargs
    ) -> Future:
        name = name or "default"

        return cls.get(name, parallel_type=parallel_type).submit(func, *args, **kwargs)
