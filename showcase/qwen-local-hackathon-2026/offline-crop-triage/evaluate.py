"""Compare two local Q4_K_M models against the same PlantVillageVQA rows under Linux/WSL."""

import argparse
import base64
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from mimetypes import guess_type
from pathlib import Path
import re
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, build_opener, ProxyHandler

from prepare import CROPS, condition

HTTP = build_opener(ProxyHandler({}))


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def score(row, answer):
    got, ref = tokens(answer), tokens(row["answer"])
    category = row["category"]
    if category == "Plant Species Identification":
        if row["answer"].lower() in CROPS:
            normalized = re.sub(r"\b(?:grapevine|vitis vinifera)\b", "grape", " ".join(got))
            crops = [crop for crop in CROPS if re.search(rf"\b{crop}\b", normalized)]
            return {"crop_accuracy": float(crops == [row["answer"].lower()])}
        return {"verification_accuracy": float(got[:1] == ref)}
    if category == "Specific Disease Identification":
        expected = condition(row["answer"])
        if expected is not None:
            return {"condition_accuracy": float(condition(answer) == expected)}
    if row["level"] == 1:
        return {}
    common = sum((Counter(got) & Counter(ref)).values())
    count = len(got) + len(ref)
    lcs = [0] * (len(ref) + 1)
    for word in got:
        prev = 0
        for i, expected in enumerate(ref, 1):
            saved = lcs[i]
            lcs[i] = prev + 1 if word == expected else max(lcs[i], lcs[i - 1])
            prev = saved
    return {"token_f1": 2 * common / count, "rouge_l_f1": 2 * lcs[-1] / count}


def request(port, endpoint, payload=None, timeout=180):
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(f"http://127.0.0.1:{port}/{endpoint}", body, {"Content-Type": "application/json"})
    # The evaluator only sends images to its own loopback server.
    with HTTP.open(req, timeout=timeout) as response:
        return json.load(response)


@contextmanager
def server(args, model, log):
    command = [
        str(args.server.resolve()), "-m", str(model), "--mmproj", str(args.projector),
        "--host", "127.0.0.1", "--port", str(args.port), "-ngl", "0",
        "--no-mmproj-offload", "-c", str(args.context), "-t", str(args.threads),
        "-np", "1", "--alias", "plant", "--jinja",
    ]
    with log.open("w") as stream, subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT) as proc:
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    if request(args.port, "health", timeout=2)["status"] == "ok":
                        break
                except (URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(f"Server failed to load {model}. See {log}")
            yield proc.pid
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def query(port, image, question, max_tokens):
    mime = guess_type(image)[0] or "image/jpeg"
    encoded = base64.b64encode(image.read_bytes()).decode()
    result = request(port, "v1/chat/completions", {
        "model": "plant",
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            {"type": "text", "text": question},
        ]}],
    })
    choice = result["choices"][0]
    return choice["message"]["content"] or "", choice["finish_reason"]


def generate(args, row):
    return query(args.port, args.test.parent / row["image"], row["question"], args.max_tokens)


def run(args, model, name, rows):
    metrics = defaultdict(list)
    elapsed = []
    output = args.out / f"{name}.jsonl"
    with server(args, model, args.out / f"{name}.log") as pid, output.open("w") as stream:
        generate(args, rows[0])  # Warmup is excluded from latency.
        for row in rows:
            start = time.monotonic()
            answer, finish = generate(args, row)
            seconds = time.monotonic() - start
            elapsed.append(seconds)
            scores = score(row, answer)
            for metric, value in scores.items():
                metrics[f"L{row['level']}/{row['category']}/{metric}"].append(value)
                if metric in {"token_f1", "rouge_l_f1"}:
                    metrics[f"L{row['level']}/all/{metric}"].append(value)
            print(json.dumps({**row, "output": answer, "finish_reason": finish,
                              "seconds": seconds, "scores": scores}), file=stream, flush=True)
        status = Path(f"/proc/{pid}/status").read_text()
        peak_kib = int(re.search(r"^VmHWM:\s+(\d+)", status, re.MULTILINE)[1])
    return {
        "metrics": {key: {"mean": sum(vals) / len(vals), "n": len(vals)}
                    for key, vals in sorted(metrics.items())},
        "seconds_per_qa": sum(elapsed) / len(elapsed),
        "peak_server_rss_mib": peak_kib / 1024,
        "artifact_bytes": model.stat().st_size + args.projector.stat().st_size,
    }


def write_report(out, summary):
    base, tuned = summary["base"], summary["tuned"]
    lines = [
        "# Held-out comparison",
        "",
        f"{summary['qa_pairs']} QA pairs from {summary['unique_images']} distinct image IDs.",
        "",
        "| Metric | Base | Fine-tuned |",
        "| --- | ---: | ---: |",
    ]
    for key, item in base["metrics"].items():
        other = tuned["metrics"][key]
        lines.append(f"| {key} (n={item['n']}) | {item['mean']:.3f} | {other['mean']:.3f} |")
    for label, key, scale in (
        ("Seconds per QA", "seconds_per_qa", 1),
        ("Peak server RSS (MiB)", "peak_server_rss_mib", 1),
        ("Model and projector (GiB)", "artifact_bytes", 1024 ** 3),
    ):
        lines.append(f"| {label} | {base[key] / scale:.3f} | {tuned[key] / scale:.3f} |")
    lines.extend(["", "Raw paired responses are in base.jsonl and tuned.jsonl."])
    if summary["limited_run"]:
        lines[2] += " This run used a row limit."
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "base", "tuned", "projector", "test", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--port", type=int, default=8172)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if not Path("/proc/self/status").exists():
        p.error("Run this evaluator under Linux or WSL2")
    if args.base.samefile(args.tuned):
        p.error("Provide different files for the base and tuned models")
    rows = [json.loads(line) for line in args.test.read_text().splitlines()][:args.limit]
    if not rows or (args.limit is not None and args.limit < 1):
        p.error("Expected at least one held-out row")
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    args.out.mkdir(parents=True)
    summary = {
        "qa_pairs": len(rows),
        "unique_images": len({row["image_id"] for row in rows}),
        "limited_run": args.limit is not None,
        "generation": {"threads": args.threads, "context": args.context, "max_tokens": args.max_tokens,
                       "temperature": 0, "seed": 42},
        "base": run(args, args.base, "base", rows),
        "tuned": run(args, args.tuned, "tuned", rows),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.out, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
