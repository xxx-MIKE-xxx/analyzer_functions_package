# tasks.py
from celery import Celery
from analyzer_pkg.analysis.pipeline   import run_pipeline
from analyzer_pkg.analysis.reporting  import build_master_report

celery_app = Celery(__name__)
celery_app.config_from_object("your_settings_module")

@celery_app.task(bind=True, acks_late=True)
def process_video(self, json_pose_path: str):
    ctx = run_pipeline(json_pose_path)

    # assume each step wrote its CSV into ctx.output_dir
    report = build_master_report(ctx.output_dir)
    return report.to_dict(orient="records")

