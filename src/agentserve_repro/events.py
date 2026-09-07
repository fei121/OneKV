#!/usr/bin/env python3
"""Task 4 — unified event logger (plan's event model)."""
import json, time
from pathlib import Path
from threading import Lock

# Canonical event fields (order preserved)
EVENT_FIELDS = ["ts_ns", "session_id", "request_id", "event", "phase",
                "input_tokens", "output_tokens", "sm_share",
                "queue_decode", "queue_prefill"]

class EventLogger:
    """Appends one JSON object per line into a JSONL file. Thread-safe."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(path, "w")
        self.lock = Lock()
        self.t0 = time.time_ns()
        self._write_header()

    def _write_header(self):
        with self.lock:
            self.fp.write(json.dumps({"__meta__": "event_log_v1",
                                      "event_fields": EVENT_FIELDS,
                                      "t0_ns": self.t0}) + "\n")
            self.fp.flush()

    def log(self, event, session_id="", request_id="", phase="",
            input_tokens=0, output_tokens=0, sm_share=None,
            queue_decode=0, queue_prefill=0, ts_ns=None):
        rec = {
            "ts_ns": int(ts_ns if ts_ns is not None else time.time_ns() - self.t0),
            "session_id": session_id,
            "request_id": request_id,
            "event": event,
            "phase": phase,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "sm_share": sm_share,
            "queue_decode": int(queue_decode),
            "queue_prefill": int(queue_prefill),
        }
        with self.lock:
            self.fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.fp.flush()

    def close(self):
        with self.lock:
            self.fp.close()

class SessionClock:
    """Per-session stopwatch for request/sub-request timing, plus request-id counter."""

    def __init__(self, logger, session_id):
        self.logger = logger
        self.session_id = session_id
        self.req_seq = 0

    def next_request_id(self, phase):
        self.req_seq += 1
        return f"{self.session_id}:{self.req_seq}:{phase}"
