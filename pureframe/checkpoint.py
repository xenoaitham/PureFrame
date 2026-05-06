import sqlite3
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

from pureframe.config import Config
from pureframe.pipeline.shots import ShotVerdict, Category, Action

class Job(BaseModel):
    id: int
    input_path: str
    output_path: str
    config_hash: str
    status: str
    total_shots: Optional[int] = None
    completed_shots: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    config_json: Optional[str] = None

class CheckpointStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        
    def _init_db(self):
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_shots INTEGER,
                    completed_shots INTEGER DEFAULT 0,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP,
                    error TEXT,
                    config_json TEXT
                );
                CREATE TABLE IF NOT EXISTS shot_verdicts (
                    job_id INTEGER REFERENCES jobs(id),
                    shot_index INTEGER,
                    verdict_json TEXT NOT NULL,
                    PRIMARY KEY (job_id, shot_index)
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_input ON jobs(input_path, config_hash);
            """)
            
    def find_or_create_job(self, input_path: Path, output_path: Path, config: Config) -> Job:
        inp_str = str(input_path.absolute())
        out_str = str(output_path.absolute())
        cfg_hash = config.config_hash
        
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM jobs 
                WHERE input_path = ? AND config_hash = ?
                ORDER BY id DESC LIMIT 1
            """, (inp_str, cfg_hash))
            row = cursor.fetchone()
            
            if row:
                return Job(**dict(row))
                
            cursor.execute("""
                INSERT INTO jobs (input_path, output_path, config_hash, status, config_json)
                VALUES (?, ?, ?, ?, ?)
            """, (inp_str, out_str, cfg_hash, "PENDING", config.model_dump_json()))
            job_id = cursor.lastrowid
            
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            return Job(**dict(cursor.fetchone()))
            
    def save_verdict(self, job_id: int, verdict: ShotVerdict) -> None:
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO shot_verdicts (job_id, shot_index, verdict_json)
                VALUES (?, ?, ?)
            """, (job_id, verdict.shot_index, verdict.model_dump_json()))
            
            self.conn.execute("""
                UPDATE jobs SET completed_shots = completed_shots + 1
                WHERE id = ?
            """, (job_id,))
            
    def load_verdicts(self, job_id: int) -> list[ShotVerdict]:
        cursor = self.conn.execute("""
            SELECT verdict_json FROM shot_verdicts
            WHERE job_id = ? ORDER BY shot_index ASC
        """, (job_id,))
        
        verdicts = []
        for row in cursor:
            verdicts.append(ShotVerdict.model_validate_json(row["verdict_json"]))
        return verdicts
        
    def update_status(self, job_id: int, status: str, **fields) -> None:
        updates = ["status = ?"]
        params = [status]
        for k, v in fields.items():
            updates.append(f"{k} = ?")
            params.append(v)
            
        if status in ("DONE", "FAILED"):
            updates.append("finished_at = CURRENT_TIMESTAMP")
            
        query = f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?"
        params.append(job_id)
        
        with self.conn:
            self.conn.execute(query, params)
            
    def list_unfinished(self) -> list[Job]:
        cursor = self.conn.execute("""
            SELECT * FROM jobs
            WHERE status NOT IN ('DONE')
            ORDER BY id DESC
        """)
        return [Job(**dict(r)) for r in cursor]
