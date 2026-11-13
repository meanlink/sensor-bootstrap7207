"""按温区分层的校准残差 bootstrap，固定随机种子可复现。"""
import math
import random
import numpy as np


def metadata():
    return {"name": "传感器校准偏差复核", "short": "sensor-bootstrap", "tagline": "保留温区构成，估计校准残差的可信范围。", "client": "望衡仪器 · 16 人校准实验室，复核工业温度探头的批次偏差", "workflow": "实验员汇总低、常、高温记录，只按整体平均误差判定是否重校。", "pains": ["不同温区的样本构成被打散", "偏差估计缺少区间", "补偿值变化后要重复手算"], "upstream": "探头与标准源配对记录 WH_calibration_pairs.csv", "downstream": "计量工程师据残差分层区间安排复校",
            "entities": ["校准配对样本", "分层偏差复核", "计量复核简报"], "dataset_name": "T-24 探头批次 · 三温区 240 对读数",
            "columns": [{"key": "sample", "label": "读数编号", "type": "string"}, {"key": "band", "label": "温区", "type": "string"}, {"key": "reference", "label": "标准值 °C", "type": "number"}, {"key": "measured", "label": "探头值 °C", "type": "number"}],
            "parameters": [{"key": "offset", "label": "扣除补偿值 °C", "type": "number", "default": 0, "min": -2, "max": 2}, {"key": "confidence", "label": "置信度", "type": "select", "default": "0.95", "options": [{"value": str(x), "label": f"{x*100:g}%"} for x in [.9, .95, .99]]}, {"key": "resamples", "label": "重采样次数", "type": "number", "default": 2000, "min": 500, "max": 5000}],
            "workbench_title": "分层校准区间估计", "action_label": "执行分层 Bootstrap", "unit": "°C", "accent": "oklch(50% 0.105 320)", "style": "metrology",
            "seed_story": ["08:00 实验员导入 WH_calibration_pairs.csv，240 对读数。", "08:15 核对低温、常温、高温各 80 对。", "08:30 设补偿值 0°C、95% 区间、2000 次采样。", "08:35 执行分层 bootstrap，查看总体及各温区区间。", "08:50 找出区间不跨 0 的温区。", "09:10 将补偿值设为 0.3°C，复核剩余偏差。", "09:40 导出计量简报，安排分温区补偿试验。"],
            "method": "残差=探头值−标准值−补偿值；每温区独立有放回抽取原样本量，再按原温区样本占比合成总体平均残差；percentile bootstrap 区间；NumPy RNG 固定 seed=1007。", "limitations": ["区间依赖本批次及温区构成", "重复测量相关性尚未建模", "统计区间不是法定计量合格证据"]}


def seed_records():
    rng = random.Random(1007)
    rows = []
    for band, center, bias, sigma in [("低温", -10, -.08, .14), ("常温", 25, .22, .1), ("高温", 70, .66, .23)]:
        for i in range(80):
            ref = center + rng.uniform(-3, 3)
            rows.append({"sample": f"T24-{band}-{i+1:03}", "band": band, "reference": round(ref, 4), "measured": round(ref + bias + rng.gauss(0, sigma), 4)})
    return rows


def analyze(records, params):
    try:
        offset = float(params.get("offset", 0))
        confidence = float(params.get("confidence", .95))
        raw_b = float(params.get("resamples", 2000))
        resamples = int(raw_b)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("补偿值、置信度和重采样次数须为有效数值")
    if not math.isfinite(offset) or not -2 <= offset <= 2 or confidence not in (.9, .95, .99) or raw_b != resamples or not 500 <= resamples <= 5000:
        raise ValueError("补偿范围 −2–2°C，次数为 500–5000 整数，置信度为 0.9/0.95/0.99")
    grouped, seen = {}, set()
    for r in records:
        try:
            sample, band = str(r["sample"]), str(r["band"])
            ref, measured = float(r["reference"]), float(r["measured"])
            if sample in seen or not band.strip() or not all(math.isfinite(x) for x in (ref, measured)):
                raise ValueError()
            seen.add(sample)
            grouped.setdefault(band, []).append(measured - ref - offset)
        except (KeyError, TypeError, ValueError):
            raise ValueError("样本编号须唯一，温区及两个有限读数不能为空")
    if len(grouped) < 2 or any(len(v) < 3 for v in grouped.values()):
        raise ValueError("至少两个温区，每温区至少 3 对读数")
    if len(records) > 5000:
        raise ValueError("单次最多 5000 对读数，请按批次拆分")
    rng = np.random.default_rng(1007)
    q = [(1 - confidence) / 2, (1 + confidence) / 2]
    total = sum(map(len, grouped.values()))
    aggregate = np.zeros(resamples)
    rows = []
    for band, values in sorted(grouped.items()):
        array = np.array(values)
        # 保留每层样本量，避免整体重抽样改变温区结构。
        means = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
        aggregate += means * len(array) / total
        low, high = [float(x) for x in np.quantile(means, q)]
        rows.append({"band": band, "n": len(array), "bias": round(float(array.mean()), 6), "low": round(low, 6), "high": round(high, 6), "review": "偏差区间未跨 0" if low > 0 or high < 0 else "区间跨 0"})
    low, high = [float(x) for x in np.quantile(aggregate, q)]
    bias = sum(sum(v) for v in grouped.values()) / total
    return {"summary": f"分层重采样 {resamples} 次，补偿后平均残差 {bias:.4f}°C，{confidence*100:g}% 区间 [{low:.4f}, {high:.4f}]°C。",
            "metrics": [{"label": "平均剩余偏差", "value": round(bias, 6), "unit": "°C"}, {"label": "区间下界", "value": round(low, 6), "unit": "°C"}, {"label": "区间上界", "value": round(high, 6), "unit": "°C"}, {"label": "重采样次数", "value": resamples}],
            "table": {"columns": [{"key": "band", "label": "温区"}, {"key": "n", "label": "样本数"}, {"key": "bias", "label": "平均残差 °C"}, {"key": "low", "label": "区间下界 °C"}, {"key": "high", "label": "区间上界 °C"}, {"key": "review", "label": "复校线索"}], "rows": rows},
            "chart": {"type": "bar", "labels": [r["band"] for r in rows], "series": [{"name": "平均残差 °C", "values": [r["bias"] for r in rows]}, {"name": "区间下界", "values": [r["low"] for r in rows]}, {"name": "区间上界", "values": [r["high"] for r in rows]}]},
            "findings": [f"{r['band']} {r['n']} 对，平均残差 {r['bias']:.4f}°C；{r['review']}。" for r in rows] + [f"原温区构成保持不变；本次扣除补偿值 {offset:g}°C，RNG seed=1007。"], "method": metadata()["method"], "warnings": ["重复测量相关性未建模，正式计量判断仍需依照实验室规程。"]}
