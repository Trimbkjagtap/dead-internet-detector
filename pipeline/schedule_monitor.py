import os
import time

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from pipeline.monitor import run_monitor_cycle

load_dotenv()

INTERVAL_HOURS = int(os.getenv("MONITOR_INTERVAL_HOURS", "6"))


def _run_job() -> None:
    summary = run_monitor_cycle()
    print(f"Monitor cycle finished: queued={summary['queued_unique']} batches={summary['batches']}")


def run_scheduler() -> None:
    scheduler = BackgroundScheduler()
    scheduler.add_job(_run_job, "interval", hours=INTERVAL_HOURS, id="monitor_cycle", replace_existing=True)
    scheduler.start()
    print(f"Scheduler started. Monitor runs every {INTERVAL_HOURS} hour(s).")

    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        scheduler.shutdown()
        print("Scheduler stopped.")


if __name__ == "__main__":
    run_scheduler()
