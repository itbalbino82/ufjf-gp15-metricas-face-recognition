#!/usr/bin/env python3
"""
GQM Metrics Runner for RealPython Face Recognition example.
Now also generates an executive summary in Markdown from REAL results,
INCLUDING breakdown por "condições" extraídas dos nomes de arquivo (ex.: _lowlight, _angle45, _blur).

Outputs:
- gqm_out/gqm_report.json
- gqm_out/predictions.csv
- gqm_out/gqm_summary.md
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def _lazy_imports():
    try:
        import face_recognition
    except Exception:
        face_recognition = None
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    except Exception:
        classification_report = confusion_matrix = accuracy_score = None
    try:
        import psutil
    except Exception:
        psutil = None
    return face_recognition, np, classification_report, confusion_matrix, accuracy_score, psutil


@dataclass
class PerfSample:
    timestamp: float
    cpu_percent: Optional[float] = None
    rss_mb: Optional[float] = None


@dataclass
class TimingStats:
    per_image_ms: List[float]
    total_ms: float
    def summary(self) -> Dict:
        if not self.per_image_ms:
            return {"count": 0}
        return {
            "count": len(self.per_image_ms),
            "mean_ms": statistics.mean(self.per_image_ms),
            "p50_ms": statistics.median(self.per_image_ms),
            "p95_ms": percentile(self.per_image_ms, 95),
            "max_ms": max(self.per_image_ms),
            "min_ms": min(self.per_image_ms),
            "total_ms": self.total_ms,
            "throughput_img_per_s": (len(self.per_image_ms) / (self.total_ms/1000.0)) if self.total_ms > 0 else None,
        }


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    k = (len(data)-1) * (p/100.0)
    f = int(k)
    c = min(f+1, len(data)-1)
    if f == c:
        return data[int(k)]
    d0 = data[f] * (c-k)
    d1 = data[c] * (k-f)
    return d0 + d1


def scan_labeled_images(root: Path) -> List[Tuple[str, Path]]:
    """Return list of (label, image_path). Label = subfolder name."""
    pairs = []
    if not root.exists():
        return pairs
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for img in sorted(label_dir.rglob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} and img.is_file():
                pairs.append((label, img))
    return pairs


def build_encodings(face_recognition, labeled_imgs: List[Tuple[str, Path]], model: str = "hog") -> Dict[str, List]:
    enc_by_label: Dict[str, List] = defaultdict(list)
    for label, img_path in labeled_imgs:
        image = face_recognition.load_image_file(str(img_path))
        boxes = face_recognition.face_locations(image, model=model)
        if not boxes:
            continue
        encs = face_recognition.face_encodings(image, known_face_locations=boxes)
        for e in encs:
            enc_by_label[label].append(e)
    return enc_by_label


def predict_image(face_recognition, np, enc_by_label: Dict[str, List], img_path: Path, model: str = "hog", tolerance: float = 0.6, default_unknown: bool = True) -> str:
    image = face_recognition.load_image_file(str(img_path))
    boxes = face_recognition.face_locations(image, model=model)
    if not boxes:
        return "unknown" if default_unknown else "no-face"
    encs = face_recognition.face_encodings(image, known_face_locations=boxes)
    if not encs:
        return "unknown" if default_unknown else "no-face"
    enc = encs[0]
    best_label = "unknown"
    best_dist = 1e9
    for label, known_encs in enc_by_label.items():
        if not known_encs:
            continue
        dists = face_recognition.face_distance(known_encs, enc)
        d = float(np.min(dists)) if len(dists) else 1e9
        if d < best_dist:
            best_dist = d
            best_label = label
    if best_dist <= tolerance:
        return best_label
    return "unknown"


def run_correctness(face_recognition, np, test_pairs: List[Tuple[str, Path]], enc_by_label: Dict[str, List], tolerance: float, model: str) -> Dict:
    labels = sorted(list(enc_by_label.keys()) + ["unknown"])
    y_true, y_pred, per_image_ms, files = [], [], [], []
    t0 = time.perf_counter()
    for true_label, img_path in test_pairs:
        im_start = time.perf_counter()
        pred = predict_image(face_recognition, np, enc_by_label, img_path, model=model, tolerance=tolerance)
        im_end = time.perf_counter()
        per_image_ms.append((im_end - im_start) * 1000.0)
        true = "unknown" if true_label.lower() == "unknown" else true_label
        y_true.append(true)
        y_pred.append(pred)
        files.append(str(img_path))
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0

    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        acc = accuracy_score(y_true, y_pred)
        bin_true = [0 if t == "unknown" else 1 for t in y_true]
        bin_pred = [0 if p == "unknown" else 1 for p in y_pred]
        tn = sum(1 for t, p in zip(bin_true, bin_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(bin_true, bin_pred) if t == 0 and p == 1)
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    except Exception as e:
        report, cm, acc, fpr = {"error": f"sklearn not available: {e}"}, [], None, None

    timing = TimingStats(per_image_ms=per_image_ms, total_ms=total_ms)

    return {
        "labels": labels,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "timing": timing.summary(),
        "per_image_ms": per_image_ms,
        "files": files,
        "fpr_overall_unknown_negative": fpr,
        "y_true": y_true,
        "y_pred": y_pred,
    }


CONDITION_RE = re.compile(r'_(lowlight|backlight|blur|occluded|noisy|angle\d+|lowres|highres)', re.IGNORECASE)

def extract_conditions(files: List[str]) -> Dict[str, List[int]]:
    """Return mapping condition -> list of indices where filename contains that condition token."""
    mapping: Dict[str, List[int]] = defaultdict(list)
    for i, fp in enumerate(files):
        name = Path(fp).stem
        for m in CONDITION_RE.finditer(name):
            token = m.group(1).lower()
            mapping[token].append(i)
    return mapping


def summarize_conditions(res: Dict) -> Dict[str, Dict]:
    """Compute per-condition accuracy and latency stats using indices mapping."""
    files = res.get("files", [])
    if not files:
        return {}
    cond_map = extract_conditions(files)
    y_true = res.get("y_true", [])
    y_pred = res.get("y_pred", [])
    lat = res.get("per_image_ms", [])
    out: Dict[str, Dict] = {}
    for cond, idxs in cond_map.items():
        if not idxs:
            continue
        total = len(idxs)
        correct = sum(1 for i in idxs if i < len(y_true) and i < len(y_pred) and y_true[i] == y_pred[i])
        acc = correct / total if total else 0.0
        lat_subset = [lat[i] for i in idxs if i < len(lat)]
        if lat_subset:
            mean_ms = statistics.mean(lat_subset)
            p95 = percentile(sorted(lat_subset), 95)
            p50 = statistics.median(lat_subset)
            mx = max(lat_subset); mn = min(lat_subset)
        else:
            mean_ms = p95 = p50 = mx = mn = None
        out[cond] = {
            "count": total,
            "accuracy": acc,
            "latency_mean_ms": mean_ms,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_min_ms": mn,
            "latency_max_ms": mx,
        }
    return out


def sample_resources(psutil, start_time: float, samples: List[PerfSample]):
    if psutil is None:
        return
    proc = psutil.Process(os.getpid())
    cpu = psutil.cpu_percent(interval=None)
    mem = proc.memory_info().rss / (1024 * 1024)
    samples.append(PerfSample(timestamp=time.perf_counter()-start_time, cpu_percent=cpu, rss_mb=mem))


def run_radon(repo_path: Path) -> Dict:
    try:
        out = subprocess.check_output(["radon", "cc", "-j", str(repo_path)], stderr=subprocess.STDOUT, text=True)
        data = json.loads(out)
        ranks = Counter()
        worst = []
        for file, entries in data.items():
            for e in entries:
                ranks[e.get("rank", "?")] += 1
                worst.append({"file": file, "name": e.get("name"),
                              "complexity": e.get("complexity"), "rank": e.get("rank")})
        worst_sorted = sorted(worst, key=lambda x: x["complexity"] or 0, reverse=True)[:20]
        return {"rank_distribution": dict(ranks), "top20_by_complexity": worst_sorted}
    except Exception as e:
        return {"error": f"radon failed or not installed: {e}"}


def run_pytest_cov(repo_path: Path) -> Dict:
    tests_dir = repo_path / "tests"
    if not tests_dir.exists():
        return {"note": "no tests/ directory; skipping coverage"}
    try:
        cmd = ["pytest", "-q", "--disable-warnings", "--maxfail=1", "--cov", str(repo_path), "--cov-report", "json"]
        subprocess.check_call(cmd, cwd=str(repo_path))
        cov_file = repo_path / "coverage.json"
        if cov_file.exists():
            with open(cov_file, "r", encoding="utf-8") as f:
                cov = json.load(f)
            totals = cov.get("totals", {})
            return {"coverage_totals": totals}
        return {"note": "coverage.json not found; coverage plugin may be missing"}
    except Exception as e:
        return {"error": f"pytest/coverage failed: {e}"}


def render_summary_md(report: Dict) -> str:
    ds = report.get("dataset", {})
    cp = report.get("correctness_and_performance", {})
    resrc = report.get("resources", {})
    codeq = report.get("code_quality", {})
    radon = codeq.get("radon", {}); cov = codeq.get("coverage", {})
    acc = cp.get("accuracy"); fpr = cp.get("fpr_overall_unknown_negative")
    clf = cp.get("classification_report", {})
    macro = clf.get("macro avg", {}) if isinstance(clf, dict) else {}
    precision = macro.get("precision"); recall = macro.get("recall"); f1 = macro.get("f1-score")
    timing = cp.get("timing", {})
    mean_ms = timing.get("mean_ms"); p50 = timing.get("p50_ms"); p95 = timing.get("p95_ms")
    tput = timing.get("throughput_img_per_s"); total_ms = timing.get("total_ms")
    cpu_avg = resrc.get("cpu_avg"); cpu_max = resrc.get("cpu_max")
    rss_avg = resrc.get("rss_mb_avg"); rss_max = resrc.get("rss_mb_max")
    ranks = radon.get("rank_distribution"); worst = radon.get("top20_by_complexity")
    coverage = cov.get("coverage_totals", {}).get("percent_covered")
    cond = report.get("conditions", {})

    def pct(x): return f"{x*100:.1f} %" if isinstance(x,(float,int)) else "n/d"
    def ms(x): return f"{x:.0f} ms" if isinstance(x,(float,int)) else "n/d"
    def imgs(x): return f"{x:.2f} img/s" if isinstance(x,(float,int)) else "n/d"

    lines = []
    lines.append("# Executive Summary — GQM Metrics for Face Recognition App\n")
    lines.append("Este documento resume as métricas **reais** extraídas pelo coletor GQM.\n")
    lines.append("---\n")
    lines.append("## 📌 G1 — Corretude")
    lines.append(f"- **Accuracy geral:** {pct(acc)}")
    lines.append(f"- **Precision média (macro):** {pct(precision)}")
    lines.append(f"- **Recall médio (macro):** {pct(recall)}")
    lines.append(f"- **F1-score médio (macro):** {pct(f1)}")
    lines.append(f"- **Taxa de falsos positivos (FPR, 'unknown' como negativo):** {pct(fpr)}\n")
    lines.append("👉 Interprete essas métricas considerando a composição do dataset (classes e proporções).")
    lines.append("\n---\n")
    lines.append("## ⚡ G2 — Desempenho")
    lines.append(f"- **Tempo médio por imagem:** {ms(mean_ms)}")
    lines.append(f"- **p50 latência:** {ms(p50)}")
    lines.append(f"- **p95 latência:** {ms(p95)}")
    lines.append(f"- **Throughput:** {imgs(tput)}")
    lines.append(f"- **Tempo total:** {ms(total_ms)}\n")
    lines.append("👉 Valores variam com hardware, backend HOG vs CNN e tamanho das imagens.")
    lines.append("\n---\n")
    lines.append("## 🖥️ G2 — Recursos")
    lines.append(f"- **CPU média:** {pct(cpu_avg/100.0) if isinstance(cpu_avg,(float,int)) else 'n/d'}")
    lines.append(f"- **CPU máxima:** {pct(cpu_max/100.0) if isinstance(cpu_max,(float,int)) else 'n/d'}")
    lines.append(f"- **Memória média (RSS):** {rss_avg:.0f} MB" if isinstance(rss_avg,(float,int)) else "- **Memória média (RSS):** n/d")
    lines.append(f"- **Memória máxima (RSS):** {rss_max:.0f} MB" if isinstance(rss_max,(float,int)) else "- **Memória máxima (RSS):** n/d")
    lines.append("\n👉 Amostragem leve via `psutil`.")
    lines.append("\n---\n")
    lines.append("## 🛠️ G3 — Qualidade de Código")
    if isinstance(ranks, dict) and ranks:
        rank_line = ", ".join([f"Rank {k}: {v}" for k,v in sorted(ranks.items())])
        lines.append(f"- **Complexidade ciclomática (Radon):** {rank_line}")
    else:
        lines.append("- **Complexidade ciclomática (Radon):** n/d")
    if isinstance(worst, list) and worst:
        topw = worst[0]
        lines.append(f"- **Pior caso:** `{topw.get('name','?')}` em `{topw.get('file','?')}` (complexidade {topw.get('complexity','?')}, Rank {topw.get('rank','?')})")
    else:
        lines.append("- **Pior caso:** n/d")
    lines.append(f"- **Cobertura de testes:** {coverage:.1f} %" if isinstance(coverage,(float,int)) else "- **Cobertura de testes:** n/d")
    lines.append("\n👉 Se não houver `tests/`, a cobertura será omitida.\n")
    lines.append("---\n")
    lines.append("## 🧪 G4 — Confiabilidade por Condição (automático)")
    if isinstance(cond, dict) and cond:
        lines.append("Resultados por sufixos encontrados nos nomes dos arquivos de teste:")
        for k, v in sorted(cond.items()):
            lines.append(f"- **{k}** — count: {v.get('count','n/d')}, accuracy: {pct(v.get('accuracy'))}, "
                         f"latência média: {ms(v.get('latency_mean_ms'))}, p95: {ms(v.get('latency_p95_ms'))}")
    else:
        lines.append("Nenhuma condição detectada. Adicione sufixos como `_lowlight`, `_angle45`, `_blur` nos nomes.")
    lines.append("\n---\n")
    lines.append("## 📊 Conclusão e Próximos Passos")
    lines.append("- Expandir dataset (baixa luz, ângulo, oclusão).")
    lines.append("- Adicionar testes e aumentar cobertura.")
    lines.append("- Avaliar `--model cnn`/GPU para tempo real.")
    lines.append("- Acompanhar métricas em execuções recorrentes.\n")
    return "\n".join(lines)


def main():
    face_recognition, np, *_ = _lazy_imports()
    parser = argparse.ArgumentParser(description="GQM Metrics Runner for face-recognition app (with condition breakdown)")
    parser.add_argument("--repo-path", type=str, required=True)
    parser.add_argument("--known-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./gqm_out")
    parser.add_argument("--model", type=str, choices=["hog", "cnn"], default="hog")
    parser.add_argument("--tolerance", type=float, default=0.6)
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    known_dir = Path(args.known_dir).resolve()
    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scan & encode
    known_pairs = scan_labeled_images(known_dir)
    test_pairs = scan_labeled_images(test_dir)
    enc_t0 = time.perf_counter()
    enc_by_label = build_encodings(face_recognition, known_pairs, model=args.model) if face_recognition else {}
    enc_t1 = time.perf_counter()

    # Correctness + performance
    res = run_correctness(face_recognition, np, test_pairs, enc_by_label, tolerance=args.tolerance, model=args.model) if face_recognition else {"error": "face_recognition not available"}

    # Add conditions summary
    conditions = summarize_conditions(res) if isinstance(res, dict) and "files" in res else {}

    # Radon + coverage
    radon_res = run_radon(repo_path)
    cov_res = run_pytest_cov(repo_path)

    # Shallow resources snapshot (to avoid long sampling)
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
        resources = {"samples": 1, "cpu_avg": cpu, "cpu_max": cpu, "rss_mb_avg": mem, "rss_mb_max": mem}
    except Exception:
        resources = {"note": "psutil not available or sampling failed"}

    report = {
        "gqm_version": "1.2",
        "dataset": {
            "known_count": len(known_pairs),
            "test_count": len(test_pairs),
            "labels_known": sorted(list(enc_by_label.keys())),
        },
        "encodings": {
            "total_time_ms": (enc_t1 - enc_t0) * 1000.0,
            "per_label_counts": {k: len(v) for k, v in enc_by_label.items()},
        },
        "correctness_and_performance": res,
        "conditions": conditions,
        "resources": resources,
        "code_quality": {"radon": radon_res, "coverage": cov_res},
        "args": vars(args),
        "host": {"python": sys.version, "platform": sys.platform},
    }

    # Write JSON
    out_json = out_dir / "gqm_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # CSV of predictions
    if isinstance(res, dict) and "y_true" in res and "y_pred" in res:
        csv_path = out_dir / "predictions.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("index,true,pred,file\n")
            for i, (t, p, fp) in enumerate(zip(res.get("y_true", []), res.get("y_pred", []), res.get("files", []))):
                f.write(f"{i},{t},{p},{fp}\n")

    # Markdown executive summary (REAL values + conditions)
    summary_md = render_summary_md(report)
    out_md = out_dir / "gqm_summary.md"
    out_md.write_text(summary_md, encoding="utf-8")

    print(f"[OK] Report JSON: {out_json}")
    print(f"[OK] Predictions CSV: {out_dir / 'predictions.csv'}")
    print(f"[OK] Executive Summary MD: {out_md}")


if __name__ == "__main__":
    main()
