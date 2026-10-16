#!/usr/bin/env python3
"""JSON driven CLI for running Alpha Machine simulations."""

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

EXIT_INPUT = 2
EXIT_RUNTIME = 3
REGIONS = {"USA", "GLB", "EUR", "ASI", "CHN", "HKG", "JPN", "KOR", "TWN"}
NEUTRALIZATIONS = {"NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "COUNTRY", "SLOW", "FAST"}


def fail(message, code=EXIT_INPUT):
    print(json.dumps({"ok": False, "error": {"message": message}}, ensure_ascii=False), file=sys.stderr)
    return code


def load_research(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取有效 JSON 文件: {}".format(exc)) from exc
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象")
    alphas = data.get("alphas")
    if not isinstance(alphas, list) or not alphas:
        raise ValueError("alphas 必须是非空数组")
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("settings 必须是对象")
    region = settings.get("region", "USA")
    universe = settings.get("universe", "TOP3000")
    neutralization = settings.get("neutralization", "SUBINDUSTRY")
    start = settings.get("start", 0)
    if region not in REGIONS:
        raise ValueError("不支持的 region: {}".format(region))
    if not isinstance(universe, str) or not universe.strip():
        raise ValueError("universe 必须是非空字符串")
    if neutralization not in NEUTRALIZATIONS:
        raise ValueError("不支持的 neutralization: {}".format(neutralization))
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("settings.start 必须是非负整数")
    parsed = []
    for index, item in enumerate(alphas):
        if not isinstance(item, dict):
            raise ValueError("alphas[{}] 必须是对象".format(index))
        expression = item.get("expression")
        decay = item.get("decay", 6)
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("alphas[{}].expression 必须是非空字符串".format(index))
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or decay < 0:
            raise ValueError("alphas[{}].decay 必须是非负有限数值".format(index))
        parsed.append({"expression": expression.strip(), "decay": decay})
    return {"settings": {"region": region, "universe": universe.strip(),
                         "neutralization": neutralization, "start": start}, "alphas": parsed}


def run_simulation(args):
    try:
        request = load_research(args.file)
    except ValueError as exc:
        return fail(str(exc))
    try:
        # Import only after validating input; machine_lib itself defines network clients.
        import machine_lib
        settings = request["settings"]
        with contextlib.redirect_stdout(sys.stderr):
            batches = machine_lib.load_task_pool_single(
                [(a["expression"], a["decay"]) for a in request["alphas"]], args.batch_size
            )
            results = machine_lib.single_simulate(
                batches, settings["neutralization"], settings["region"],
                settings["universe"], settings["start"]
            ) or []
    except Exception as exc:
        return fail("模拟调用失败: {}".format(exc), EXIT_RUNTIME)
    results = [dict(r) for r in results]
    summary = {"COMPLETE": 0, "WARNING": 0, "FAILED": 0}
    for record in results:
        status = record.get("status")
        if status == "COMPLETE":
            summary["COMPLETE"] += 1
        elif status == "WARNING":
            summary["WARNING"] += 1
        else:
            summary["FAILED"] += 1
    if not results:
        simulation_status = "unknown"
    elif summary["FAILED"] == 0:
        simulation_status = "complete"
    elif summary["COMPLETE"] or summary["WARNING"]:
        simulation_status = "partial"
    else:
        simulation_status = "failed"
    print(json.dumps({
        "ok": True,
        "operation": "simulate",
        "expression_count": len(request["alphas"]),
        "simulation_status": simulation_status,
        "settings": settings,
        "result_summary": summary,
        "result": results,
    }, ensure_ascii=False))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="以 JSON 批量运行 Alpha Machine 模拟")
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate", help="验证 JSON 并调用现有模拟函数")
    simulate.add_argument("file", help="research JSON 文件")
    simulate.add_argument("--batch-size", type=int, default=3, help="每批表达式数量（默认 3）")
    simulate.set_defaults(handler=run_simulation)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "simulate" and args.batch_size <= 0:
        return fail("--batch-size 必须是正整数")
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
