# mem_logger.py
import os, time, threading, psutil

def start_mem_logger(interval_sec: int = 5):
    proc = psutil.Process(os.getpid())
    def _run():
        while True:
            rss = proc.memory_info().rss / 1e6
            # You can add more: vms/num_fds/threads if you like
            #print(f"[mem] rss={rss:.1f}MB")
            time.sleep(interval_sec)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
