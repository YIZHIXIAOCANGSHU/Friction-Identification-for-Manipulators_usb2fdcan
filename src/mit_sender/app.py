from __future__ import annotations

from PySide6.QtCore import QDir, QLockFile
from PySide6.QtWidgets import QMessageBox

from mit_sender.ui import MitSenderWindow, create_app


def _try_lock_once(lock: QLockFile) -> bool:
    if lock.tryLock(100):
        return True
    lock.removeStaleLockFile()
    return lock.tryLock(100)


def main() -> int:
    app = create_app()
    lock = QLockFile(QDir.temp().absoluteFilePath("damiao-mit-sender.lock"))
    lock.setStaleLockTime(30000)
    if not _try_lock_once(lock):
        QMessageBox.warning(None, "程序已在运行", "MIT 电机发送工具已经在运行，请先关闭旧窗口后再启动。")
        return 1
    window = MitSenderWindow()
    window.show()
    try:
        return app.exec()
    except KeyboardInterrupt:
        window.close()
        return 130
    finally:
        lock.unlock()


if __name__ == "__main__":
    raise SystemExit(main())
