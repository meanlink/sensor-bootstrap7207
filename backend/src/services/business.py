"""按温区分层的校准残差 bootstrap，固定随机种子可复现。

支持两种区间估计口径：

- reading（读数口径，旧算法）：每温区独立对读数有放回重采样，把每条读数当作独立样本。
- probe（探头口径，集群 bootstrap）：同一探头在各温区的全部读数作为一个整体一起抽取；
  重采样后仍按原温区样本占比合成总体平均残差。探头缺少某个温区时，该温区该探头不贡献读数。

旧数据没有探头编号时自动沿用读数口径；同批次若只有部分行带探头编号则拒绝分析。
"""
import math
import random
import numpy as np

SEED = 1007
READING_ESTIMATOR = "reading"
PROBE_ESTIMATOR = "probe"
ESTIMATOR_LABELS = {
    READING_ESTIMATOR: "读数分层 Bootstrap（按读数独立重采样）",
    PROBE_ESTIMATOR: "探头集群 Bootstrap（同探头跨温区整体重采样）",
}
PROBE_COLUMN = "probe"
MAX_RECORDS = 5000
_CHUNK = 200


def metadata():
    return {"name": "传感器校准偏差复核", "short": "sensor-bootstrap", "tagline": "保留温区构成，估计校准残差的可信范围。", "client": "望衡仪器 · 16 人校准实验室，复核工业温度探头的批次偏差", "workflow": "实验员汇总低、常、高温记录，只按整体平均误差判定是否重校。", "pains": ["不同温区的样本构成被打散", "偏差估计缺少区间", "同一探头跨温区重复测量，读数口径会低估相关偏差"], "upstream": "探头与标准源配对记录 WH_calibration_pairs.csv", "downstream": "计量工程师据残差分层区间安排复校",
            "entities": ["校准配对样本", "分层偏差复核", "计量复核简报"], "dataset_name": "T-24 探头批次 · 三温区 240 对读数",
            "columns": [{"key": "sample", "label": "读数编号", "type": "string"}, {"key": "probe", "label": "探头编号", "type": "string", "optional": True}, {"key": "band", "label": "温区", "type": "string"}, {"key": "reference", "label": "标准值 °C", "type": "number"}, {"key": "measured", "label": "探头值 °C", "type": "number"}],
            "parameters": [{"key": "offset", "label": "扣除补偿值 °C", "type": "number", "default": 0, "min": -2, "max": 2}, {"key": "confidence", "label": "置信度", "type": "select", "default": "0.95", "options": [{"value": str(x), "label": f"{x*100:g}%"} for x in [.9, .95, .99]]}, {"key": "resamples", "label": "重采样次数", "type": "number", "default": 2000, "min": 500, "max": 5000}, {"key": "estimator", "label": "区间估计方式", "type": "select", "default": READING_ESTIMATOR, "options": [{"value": READING_ESTIMATOR, "label": "读数口径（原算法，按读数独立重采样）"}, {"value": PROBE_ESTIMATOR, "label": "探头口径（同探头跨温区整体重采样）"}]}],
            "workbench_title": "分层校准区间估计", "action_label": "执行分层 Bootstrap", "unit": "°C", "accent": "oklch(50% 0.105 320)", "style": "metrology",
            "seed_story": ["08:00 实验员导入 WH_calibration_pairs.csv，240 对读数。", "08:15 核对低温、常温、高温各 80 对。", "08:30 设补偿值 0°C、95% 区间、2000 次采样。", "08:35 执行分层 bootstrap，查看总体及各温区区间。", "08:50 找出区间不跨 0 的温区。", "09:10 将补偿值设为 0.3°C，复核剩余偏差。", "09:40 导出计量简报，安排分温区补偿试验。"],
            "method": "残差=探头值−标准值−补偿值。读数口径：每温区独立对读数有放回抽取原样本量；探头口径：同一探头在各温区的读数作为一个整体有放回抽取，缺温区的探头在该温区不贡献读数。两种口径都按原温区样本占比合成总体平均残差，采用 percentile bootstrap 区间，NumPy RNG 固定 seed=1007。",
            "estimators": ESTIMATOR_LABELS,
            "limitations": ["区间依赖本批次及温区构成", "读数口径未建模同一探头重复测量的相关性，探头口径按探头集群重采样体现该相关性", "探头口径要求每温区至少 2 个探头；无探头编号的旧数据只能使用读数口径", "统计区间不是法定计量合格证据"]}


def seed_records():
    rng = random.Random(1007)
    rows = []
    # 20 支探头，每支在三个温区各重复测量 4 次，共 240 对读数。
    probes = [f"T24-P{p+1:02d}" for p in range(20)]
    for band, center, bias, sigma in [("低温", -10, -.08, .14), ("常温", 25, .22, .1), ("高温", 70, .66, .23)]:
        for i in range(80):
            probe = probes[i % len(probes)]
            ref = center + rng.uniform(-3, 3)
            rows.append({"sample": f"T24-{band}-{i+1:03}", PROBE_COLUMN: probe, "band": band, "reference": round(ref, 4), "measured": round(ref + bias + rng.gauss(0, sigma), 4)})
    return rows


def _parse_params(params):
    try:
        offset = float(params.get("offset", 0))
        confidence = float(params.get("confidence", .95))
        raw_b = float(params.get("resamples", 2000))
        resamples = int(raw_b)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("补偿值、置信度和重采样次数须为有效数值")
    if not math.isfinite(offset) or not -2 <= offset <= 2 or confidence not in (.9, .95, .99) or raw_b != resamples or not 500 <= resamples <= 5000:
        raise ValueError("补偿范围 −2–2°C，次数为 500–5000 整数，置信度为 0.9/0.95/0.99")
    # 历史任务没有 estimator 参数，默认读数口径以保持旧结论可复现。
    estimator = str(params.get("estimator", READING_ESTIMATOR))
    if estimator not in ESTIMATOR_LABELS:
        raise ValueError("区间估计方式须为读数口径 reading 或探头口径 probe")
    return offset, confidence, resamples, estimator


def _parse_records(records, offset):
    """返回按温区排序的 (温区, 残差列表, 探头列表（无编号时为空字符串）)。

    排序保证打乱读数顺序后重采样结论不变。
    """
    grouped = {}
    seen = set()
    has_probe_flags = set()
    for r in records:
        try:
            sample, band = str(r["sample"]), str(r["band"])
            ref, measured = float(r["reference"]), float(r["measured"])
            if sample in seen or not band.strip() or not all(math.isfinite(x) for x in (ref, measured)):
                raise ValueError()
            seen.add(sample)
            raw_probe = r.get(PROBE_COLUMN)
            if raw_probe is None or (isinstance(raw_probe, str) and not raw_probe.strip()):
                probe, has_probe = "", False
            else:
                probe, has_probe = str(raw_probe).strip(), True
            has_probe_flags.add(has_probe)
            grouped.setdefault(band, []).append((measured - ref - offset, probe))
        except (KeyError, TypeError, ValueError):
            raise ValueError("样本编号须唯一，温区及两个有限读数不能为空，探头编号需整列统一提供或留空")
    if len(grouped) < 2 or any(len(v) < 3 for v in grouped.values()):
        raise ValueError("至少两个温区，每温区至少 3 对读数")
    if len(records) > MAX_RECORDS:
        raise ValueError("单次最多 5000 对读数，请按批次拆分")
    if len(has_probe_flags) > 1:
        raise ValueError("探头编号存在空值：同一批次须全部带探头编号或全部不带，不允许混有空探头编号")
    parsed = []
    for band in sorted(grouped):
        entries = sorted(grouped[band], key=lambda item: (item[1], item[0]))
        values = [v for v, _ in entries]
        probes = [p for _, p in entries]
        parsed.append((band, values, probes))
    # 空探头与带编号探头同时出现时已在前面拒绝；这里只剩“全有”或“全无”。
    probe_available = True in has_probe_flags
    return parsed, probe_available


def _bootstrap_reading(rng, bands_values, resamples, q):
    """读数口径：每温区独立对读数有放回重采样原样本量。"""
    aggregate = np.zeros(resamples)
    total = sum(len(v) for v in bands_values)
    band_stats = []
    for values in bands_values:
        array = np.sort(np.asarray(values, dtype=float))
        means = rng.choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
        aggregate += means * array.size / total
        low, high = (float(x) for x in np.quantile(means, q))
        band_stats.append((array, float(array.mean()), low, high))
    low, high = (float(x) for x in np.quantile(aggregate, q))
    return band_stats, low, high


def _bootstrap_probe(rng, parsed, resamples, q):
    """探头口径：同一探头跨温区整体有放回重采样，仍按原温区占比合成。"""
    all_probes = sorted({p for _, _, probes in parsed for p in probes})
    if len(all_probes) < 2:
        raise ValueError("探头口径至少需要 2 支探头")
    probe_index = {p: i for i, p in enumerate(all_probes)}
    k = len(all_probes)
    total = sum(len(values) for _, values, _ in parsed)
    band_sums, band_counts, band_weights, band_means_raw = [], [], [], []
    for _, values, probes in parsed:
        if len(set(probes)) < 2:
            raise ValueError("探头口径要求每个温区至少出现 2 支不同探头")
        sums = np.zeros(k)
        counts = np.zeros(k, dtype=np.int64)
        for value, probe in zip(values, probes):
            idx = probe_index[probe]
            sums[idx] += value
            counts[idx] += 1
        band_sums.append(sums)
        band_counts.append(counts)
        band_weights.append(len(values) / total)
        band_means_raw.append(float(np.asarray(values, dtype=float).mean()))

    aggregate = np.empty(0)
    band_draw_means = [np.empty(0) for _ in parsed]
    remaining = resamples
    empty_chunks = 0
    while remaining > 0:
        size = min(_CHUNK, remaining)
        # 一次抽取决定整支探头在所有温区的去留，保证同探头读数一起重采样。
        picks = rng.integers(0, k, size=(size, k))
        valid = np.ones(size, dtype=bool)
        chunk_means = []
        for sums, counts in zip(band_sums, band_counts):
            drawn_sums = sums[picks].sum(axis=1)
            observed = counts[picks].sum(axis=1)
            valid &= observed > 0
            chunk_means.append(np.divide(drawn_sums, observed, where=observed > 0))
        if not bool(valid.all()):
            # 极少数重采样在某温区抽不到任何探头，重抽这些批次而非伪造估计。
            if not bool(valid.any()):
                empty_chunks += 1
                if empty_chunks > max(100, resamples // _CHUNK + 50):
                    raise ValueError("探头重采样在部分温区无法取得读数，请增加该温区探头数量")
                continue
            picks = picks[valid]
            chunk_means = [m[valid] for m in chunk_means]
        kept = picks.shape[0]
        chunk_aggregate = np.zeros(kept)
        for mean_band, weight in zip(chunk_means, band_weights):
            chunk_aggregate += mean_band * weight
        aggregate = np.concatenate([aggregate, chunk_aggregate])
        for i, mean_band in enumerate(chunk_means):
            band_draw_means[i] = np.concatenate([band_draw_means[i], mean_band])
        remaining -= kept

    band_stats = []
    for (band, values, _), raw_mean, draws in zip(parsed, band_means_raw, band_draw_means):
        low, high = (float(x) for x in np.quantile(draws, q))
        band_stats.append((np.sort(np.asarray(values, dtype=float)), raw_mean, low, high))
    low, high = (float(x) for x in np.quantile(aggregate, q))
    return band_stats, low, high, all_probes


def analyze(records, params):
    offset, confidence, resamples, requested = _parse_params(params)
    parsed, probe_available = _parse_records(records, offset)
    rng = np.random.default_rng(SEED)
    q = [(1 - confidence) / 2, (1 + confidence) / 2]

    warnings = []
    if requested == PROBE_ESTIMATOR and not probe_available:
        # 旧数据没有探头编号：无法集群重采样，明确回退到原算法并在结果中留痕。
        estimator = READING_ESTIMATOR
        warnings.append("该批次缺少探头编号，无法按探头集群重采样，已回退为读数口径；探头口径需导入含探头编号的数据。")
    else:
        estimator = requested
    if estimator == READING_ESTIMATOR and probe_available:
        warnings.append("读数口径把同一探头的重复测量当作独立样本，可能低估区间宽度；需要考虑探头内相关时请改用探头口径。")
    elif estimator == READING_ESTIMATOR and not probe_available:
        warnings.append("该批次缺少探头编号，仅支持读数口径；导入含探头编号的数据后可使用探头集群 Bootstrap。")

    if estimator == PROBE_ESTIMATOR:
        band_stats, low, high, all_probes = _bootstrap_probe(rng, parsed, resamples, q)
    else:
        band_stats, low, high = _bootstrap_reading(rng, [v for _, v, _ in parsed], resamples, q)
        all_probes = []

    total = sum(len(values) for _, values, _ in parsed)
    bias = sum(sum(values) for _, values, _ in parsed) / total
    rows = []
    for (band, values, probes), (array, raw_mean, b_low, b_high) in zip(parsed, band_stats):
        row = {"band": band, "n": len(array), "bias": round(raw_mean, 6), "low": round(b_low, 6), "high": round(b_high, 6), "review": "偏差区间未跨 0" if b_low > 0 or b_high < 0 else "区间跨 0"}
        if estimator == PROBE_ESTIMATOR:
            row["probes"] = len(set(probes))
        rows.append(row)

    columns = [{"key": "band", "label": "温区"}, {"key": "n", "label": "样本数"}]
    if estimator == PROBE_ESTIMATOR:
        columns.append({"key": "probes", "label": "探头数"})
    columns += [{"key": "bias", "label": "平均残差 °C"}, {"key": "low", "label": "区间下界 °C"}, {"key": "high", "label": "区间上界 °C"}, {"key": "review", "label": "复校线索"}]

    estimator_label = ESTIMATOR_LABELS[estimator]
    if estimator == PROBE_ESTIMATOR:
        summary = (f"探头集群重采样 {resamples} 次（{len(all_probes)} 支探头整体抽取），补偿后平均残差 {bias:.4f}°C，"
                   f"{confidence*100:g}% 区间 [{low:.4f}, {high:.4f}]°C。")
    else:
        summary = (f"读数分层重采样 {resamples} 次，补偿后平均残差 {bias:.4f}°C，"
                   f"{confidence*100:g}% 区间 [{low:.4f}, {high:.4f}]°C。")

    findings = [f"{r['band']} {r['n']} 对" + (f"、{r['probes']} 支探头" if estimator == PROBE_ESTIMATOR else "") + f"，平均残差 {r['bias']:.4f}°C；{r['review']}。" for r in rows]
    findings.append(f"原温区构成保持不变；本次扣除补偿值 {offset:g}°C，估计方式为{estimator_label}，RNG seed={SEED}。")
    if estimator == PROBE_ESTIMATOR:
        findings.append("同一探头在各温区的读数作为整体一起抽取；缺少某温区读数的探头在该温区不贡献，重采样后仍按原温区占比合成。")
    if not warnings:
        warnings.append("重复测量相关性未建模，正式计量判断仍需依照实验室规程。" if estimator == READING_ESTIMATOR
                        else "集群 bootstrap 依赖本批次探头数量；探头数偏少时间区间本身不稳定，正式计量判断仍需依照实验室规程。")

    return {"estimator": estimator, "estimator_label": estimator_label,
            "probe_available": probe_available,
            "summary": summary,
            "metrics": [{"label": "平均剩余偏差", "value": round(bias, 6), "unit": "°C"}, {"label": "区间下界", "value": round(low, 6), "unit": "°C"}, {"label": "区间上界", "value": round(high, 6), "unit": "°C"}, {"label": "重采样次数", "value": resamples}],
            "table": {"columns": columns, "rows": rows},
            "chart": {"type": "bar", "labels": [r["band"] for r in rows], "series": [{"name": "平均残差 °C", "values": [r["bias"] for r in rows]}, {"name": "区间下界", "values": [r["low"] for r in rows]}, {"name": "区间上界", "values": [r["high"] for r in rows]}]},
            "findings": findings, "method": metadata()["method"], "warnings": warnings}
