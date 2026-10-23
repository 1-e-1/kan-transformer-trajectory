"""One benchmark GPU process at a time on Windows; not an algorithm module."""
from pathlib import Path
import msvcrt


class BenchmarkGPULock:
    def __init__(self,path):
        self.path=Path(path);self.handle=None

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.path.open('a+b')
        if self.path.stat().st_size==0:self.handle.write(b'0');self.handle.flush()
        self.handle.seek(0)
        try:msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as error:
            self.handle.close();self.handle=None
            raise RuntimeError('Another benchmark owns the GPU lock; run experiments serially') from error
        return self

    def __exit__(self,*args):
        if self.handle is not None:
            self.handle.seek(0);msvcrt.locking(self.handle.fileno(),msvcrt.LK_UNLCK,1)
            self.handle.close();self.handle=None
