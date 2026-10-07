"""AwareLiquid 决策模型服务 — TypeSafe /v1/systemone（O1-Flash 单前向 readout）。

与 benchmarks/decision_train.py 共用 option_text/core_text/embed_options/
decision_scores，保证训练-推理口径一致。单前向 → 原生分布 → 毫秒级。

    /root/M2/.venv/bin/python jev_service_flash.py <ckpt> [port]
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/o1flash")
from mt_flash.config import FlashConfig  # noqa: E402
from mt_flash.model import O1Flash  # noqa: E402
from benchmarks.decision_train import (  # noqa: E402
    decision_scores, embed_options, option_text)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/root/decision/decision_v1.pt"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8199
MODEL_NAME = Path(CKPT).stem

print(f"loading {CKPT} ...", flush=True)
state = torch.load(CKPT, map_location="cpu", weights_only=False)
model = O1Flash(FlashConfig())
model.load_state_dict(state["model"])
model = model.cuda().eval()
print("ready.", flush=True)


def derive_labels(qtype: str, criteria):
    if qtype == "noul":
        return ["no", "yes"]
    if isinstance(criteria, dict):
        return [str(k) for k in criteria.keys()]
    if isinstance(criteria, list):
        return [str(i) for i in range(len(criteria))]
    return []


def row_from_request(payload: dict) -> dict:
    q = (payload.get("questions") or {}).get("decision") or {}
    criteria = q.get("criteria")
    labels = derive_labels(q.get("type"), criteria)
    state_ = payload.get("state", "")
    if not isinstance(state_, str):
        state_ = json.dumps(state_, ensure_ascii=False)
    return {
        "state": state_,
        "instructions": q.get("instructions", "") or "",
        "criteria": criteria if isinstance(criteria, dict) else None,
        "labels": labels,
    }


@torch.no_grad()
def decide(payload: dict) -> dict:
    q = (payload.get("questions") or {}).get("decision") or {}
    qtype = q.get("type")
    row = row_from_request(payload)
    labels = row["labels"]
    if not labels:
        raise ValueError("no labels derivable")
    ids = model._ids_for(
        (row["state"] + "\n" + row["instructions"]).strip())[None].cuda()
    mask = torch.ones_like(ids, dtype=torch.bool)
    y = model(ids)
    opts, omask = embed_options(model.heads, [row], y.device)
    scores = decision_scores(model.heads, y, opts, omask, mask)
    probs = F.softmax(scores, dim=-1)[0]
    prob_map = {lab: float(p) for lab, p in zip(labels, probs)}
    if qtype == "noul":
        p_yes = float(probs[labels.index("yes")])
        answer = {"type": "noul", "noul": p_yes}
    elif qtype == "choice":
        best = max(sorted(prob_map), key=lambda k: prob_map[k])
        answer = {"type": "choice", "choice": best, "probabilities": prob_map}
    else:
        answer = {"type": qtype, "probabilities": prob_map}
    return {"answers": {"decision": answer}, "model": MODEL_NAME,
            "usage": {"input_tokens": int(ids.shape[1]), "output_tokens": 0}}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(200, {"status": "ok", "model": MODEL_NAME})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/systemone":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n))
            t0 = time.time()
            resp = decide(payload)
            dt = (time.time() - t0) * 1000
            ans = resp["answers"]["decision"]
            print(f"[{ans.get('type')}] {dt:.1f}ms "
                  f"{ans.get('choice', ans.get('noul'))}", flush=True)
            self._send(200, resp)
        except Exception as ex:  # noqa: BLE001
            print("ERROR:", type(ex).__name__, str(ex)[:200], flush=True)
            self._send(500, {"error": f"{type(ex).__name__}: {str(ex)[:200]}"})

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    print(f"serving on 0.0.0.0:{PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
