import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

SETS = {"s_mem", "s_para", "h1", "h2", "s_neg", "dev_para"}
ARMS = {"A0", "A1", "A2", "A3"}
JUDGES = {"gpt", "claude"}


class JobRequest(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


def _choices(values, allowed: set, label: str) -> list[str]:
    values = list(values or [])
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"unknown {label}: {bad}")
    return values


def build_command(name: str, params: dict, python: str) -> list[str]:
    m = [python, "-m"]
    if name == "answers":
        cmd = m + ["privacy_bench.evaluate.answers"]
        sets = _choices(params.get("sets"), SETS, "sets")
        arms = _choices(params.get("arms"), ARMS, "arms")
        return cmd + (["--sets", *sets] if sets else []) + (["--arms", *arms] if arms else [])
    if name == "judge":
        judge = params.get("judge", "gpt")
        if judge not in JUDGES:
            raise ValueError(f"unknown judge: {judge}")
        cmd = m + ["privacy_bench.judge.run", "run", "--judge", judge]
        sets = _choices(params.get("sets"), SETS, "sets")
        arms = _choices(params.get("arms"), ARMS, "arms")
        return cmd + (["--sets", *sets] if sets else []) + (["--arms", *arms] if arms else [])
    if name == "analyze":
        return m + ["privacy_bench.analysis.run"]
    if name == "wandb_log":
        kind = params.get("kind", "evaluation")
        if kind not in ("evaluation", "gate"):
            raise ValueError(f"unknown kind: {kind}")
        return m + ["privacy_bench.evaluate.wandb_log", kind]
    if name == "registry_status":
        return m + ["privacy_bench.registry.registry", "status"]
    if name == "gate":
        return m + ["privacy_bench.registry.registry", "gate"] + (["--dry-run"] if params.get("dry_run", True) else [])
    if name == "promote":
        version = str(params.get("version", ""))
        if not version.isdigit():
            raise ValueError("version must be a registry version number")
        return m + ["privacy_bench.registry.registry", "promote", version]
    if name == "rollback":
        return m + ["privacy_bench.registry.registry", "rollback"]
    raise ValueError(f"unknown job: {name}")


def parse_output(text: str):
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    lines = []
    for line in text.splitlines():
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if len(lines) > 1:
        return lines
    start = text.rfind("\n{")
    if start >= 0:
        try:
            return json.loads(text[start + 1:])
        except json.JSONDecodeError:
            pass
    return lines[0] if lines else None


class Runner:
    def __init__(self, root: Path, log_dir: Path, python: str, env: dict):
        self.root = root
        self.log_dir = log_dir
        self.python = python
        self.env = env
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def busy(self) -> dict | None:
        return next((j for j in self.jobs.values() if j["status"] in ("queued", "running")), None)

    def start(self, name: str, params: dict) -> dict:
        command = build_command(name, params, self.python)
        with self.lock:
            running = self.busy()
            if running:
                raise RuntimeError(f"job {running['id']} ({running['name']}) is still {running['status']}")
            job_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
            job = {"id": job_id, "name": name, "params": params, "status": "queued", "returncode": None,
                   "created": time.time(), "started": None, "finished": None, "result": None}
            self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job, command), daemon=True).start()
        return job

    def _paths(self, job: dict) -> tuple[Path, Path]:
        return self.log_dir / f"{job['id']}.out", self.log_dir / f"{job['id']}.err"

    def _run(self, job: dict, command: list[str]) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        out_path, err_path = self._paths(job)
        job.update(status="running", started=time.time())
        with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
            returncode = subprocess.run(command, cwd=self.root, env=self.env, stdout=out, stderr=err).returncode
        job.update(
            status="succeeded" if returncode == 0 else "failed",
            returncode=returncode,
            finished=time.time(),
            result=parse_output(out_path.read_text(encoding="utf-8")),
        )

    def tail(self, job: dict, size: int = 1500) -> dict:
        out_path, err_path = self._paths(job)

        def read(path: Path) -> str:
            return path.read_text(encoding="utf-8", errors="replace")[-size:] if path.exists() else ""

        return {"stdout": read(out_path), "stderr": read(err_path)}

    def view(self, job: dict) -> dict:
        out = dict(job)
        out["tail"] = self.tail(job)
        end = job["finished"] or time.time()
        out["seconds"] = round(end - job["started"], 1) if job["started"] else None
        out["done"] = job["status"] in ("succeeded", "failed")
        return out


def create_app(runner: Runner, token: str) -> FastAPI:
    app = FastAPI(title="palr ops service")

    def auth(authorization: str = Header(default="")) -> None:
        if not secrets.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/health")
    def health() -> dict:
        running = runner.busy()
        return {"status": "ok", "running": running["id"] if running else None}

    @app.post("/jobs", status_code=202, dependencies=[Depends(auth)])
    def submit(req: JobRequest) -> dict:
        try:
            job = runner.start(req.name, req.params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return runner.view(job)

    @app.get("/jobs", dependencies=[Depends(auth)])
    def list_jobs() -> list[dict]:
        return [runner.view(j) for j in sorted(runner.jobs.values(), key=lambda j: j["created"], reverse=True)[:20]]

    @app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
    def get_job(job_id: str) -> dict:
        job = runner.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return runner.view(job)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP service that runs pipeline steps as background jobs for n8n")
    parser.add_argument("--config", default="configs/ops.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    from dotenv import load_dotenv

    load_dotenv()
    token = os.environ.get("OPS_TOKEN")
    if not token:
        raise SystemExit("set OPS_TOKEN in .env before starting the ops service")
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    runner = Runner(Path.cwd(), Path(cfg["log_dir"]), sys.executable, env)

    import uvicorn

    uvicorn.run(create_app(runner, token), host=cfg["host"], port=cfg["port"])


if __name__ == "__main__":
    main()
