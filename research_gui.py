#!/usr/bin/env python3
"""Tkinter desktop interface for Alpha Machine research simulations."""

import json
import math
import queue
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from research_cli import NEUTRALIZATIONS, REGIONS, load_research


class QueueWriter:
    def __init__(self, events):
        self.events = events

    def write(self, text):
        if text:
            safe_text = re.sub(
                r"(?i)(password|username|authorization|cookie|token)(\\s*[:=]\\s*)[^\\s,;]+",
                r"\\1\\2[已隐藏]", text.rstrip())
            self.events.put(("log", safe_text))

    def flush(self):
        pass


class ResearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Alpha Machine 研究模拟")
        self.root.geometry("900x720")
        self.events = queue.Queue()
        self.running = False
        self.path = None
        self.expression_text = tk.Text(root, height=14, wrap="none")
        self.region = tk.StringVar(value="USA")
        self.universe = tk.StringVar(value="TOP3000")
        self.neutralization = tk.StringVar(value="SUBINDUSTRY")
        self.batch_size = tk.StringVar(value="3")
        self.start = tk.StringVar(value="0")
        self._build()
        self.root.after(100, self._drain_events)

    def _build(self):
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        buttons = ttk.Frame(root, padding=8)
        buttons.grid(row=0, column=0, sticky="ew")
        ttk.Button(buttons, text="加载 JSON", command=self.load).pack(side="left")
        ttk.Button(buttons, text="保存 JSON", command=self.save).pack(side="left", padx=6)

        editor = ttk.LabelFrame(root, text="Alpha 表达式（每行一条；可写 expression | decay）", padding=8)
        editor.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(0, weight=1)
        self.expression_text.grid(in_=editor, row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(editor, orient="vertical", command=self.expression_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.expression_text.configure(yscrollcommand=scroll.set)

        settings = ttk.LabelFrame(root, text="模拟设置", padding=8)
        settings.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        fields = [("Region", self.region, sorted(REGIONS)),
                  ("Universe", self.universe, None),
                  ("Neutralization", self.neutralization, sorted(NEUTRALIZATIONS)),
                  ("批量大小", self.batch_size, None), ("起始任务", self.start, None)]
        for col, (label, var, values) in enumerate(fields):
            ttk.Label(settings, text=label).grid(row=0, column=col, padx=4, sticky="w")
            widget = ttk.Combobox(settings, textvariable=var, values=values, state="readonly" if values else "normal", width=17)
            widget.grid(row=1, column=col, padx=4, sticky="ew")
            settings.columnconfigure(col, weight=1)
        action = ttk.Frame(root, padding=8)
        action.grid(row=3, column=0, sticky="ew")
        self.start_button = ttk.Button(action, text="开始模拟", command=self.start_simulation)
        self.start_button.pack(side="left")
        self.status = ttk.Label(action, text="就绪")
        self.status.pack(side="left", padx=12)
        logs = ttk.LabelFrame(root, text="日志", padding=8)
        logs.grid(row=4, column=0, sticky="nsew", padx=8, pady=4)
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)
        self.log_text = tk.Text(logs, height=10, state="disabled", wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(logs, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def _parsed_alphas(self):
        lines = self.expression_text.get("1.0", "end").splitlines()
        alphas = []
        for number, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            expression, sep, decay_text = line.rpartition("|")
            if not sep:
                expression, decay = line, 6
            else:
                expression = expression.strip()
                try:
                    decay = float(decay_text.strip())
                    if decay.is_integer():
                        decay = int(decay)
                except ValueError as exc:
                    raise ValueError(f"第 {number} 行 decay 必须是非负有限数值") from exc
            if not expression:
                raise ValueError(f"第 {number} 行表达式不能为空")
            if isinstance(decay, bool) or not math.isfinite(decay) or decay < 0:
                raise ValueError(f"第 {number} 行 decay 必须是非负有限数值")
            alphas.append({"expression": expression, "decay": decay})
        if not alphas:
            raise ValueError("请至少输入一条 Alpha 表达式")
        return alphas

    def _config(self):
        alphas = self._parsed_alphas()
        try:
            batch = int(self.batch_size.get())
            start = int(self.start.get())
        except ValueError as exc:
            raise ValueError("批量大小和起始任务必须是整数") from exc
        if batch <= 0 or start < 0:
            raise ValueError("批量大小必须为正整数，起始任务必须为非负整数")
        config = {"settings": {"region": self.region.get(), "universe": self.universe.get().strip(),
                               "neutralization": self.neutralization.get(), "start": start}, "alphas": alphas}
        # Keep GUI and CLI validation/defaults in sync.
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(config, handle)
            validation_path = handle.name
        try:
            normalized = load_research(validation_path)
        finally:
            Path(validation_path).unlink(missing_ok=True)
        return normalized, batch

    def load(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not filename:
            return
        try:
            config = load_research(filename)
        except ValueError as exc:
            messagebox.showerror("无法加载配置", str(exc))
            return
        s = config["settings"]
        self.region.set(s["region"])
        self.universe.set(s["universe"])
        self.neutralization.set(s["neutralization"])
        self.start.set(str(s["start"]))
        self.expression_text.delete("1.0", "end")
        self.expression_text.insert("1.0", "\n".join(f'{a["expression"]} | {a["decay"]}' for a in config["alphas"]))
        self.path = filename
        self._log(f"已加载配置：{filename}")

    def save(self):
        try:
            config, _ = self._config()
        except (ValueError, OSError) as exc:
            messagebox.showerror("输入无效", str(exc))
            return
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="research.json")
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as handle:
                json.dump(config, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.path = filename
        self._log(f"配置已保存：{filename}")

    def start_simulation(self):
        if self.running:
            return
        try:
            config, batch_size = self._config()
        except (ValueError, OSError) as exc:
            messagebox.showerror("输入无效", str(exc))
            return
        confirmed = messagebox.askyesno(
            "确认真实模拟",
            "继续后将调用 machine_lib 登录 WorldQuant BRAIN，并向服务提交真实模拟任务。\n"
            "这会产生真实服务端操作。是否继续？",
            icon="warning",
        )
        if not confirmed:
            return
        self.running = True
        self.start_button.configure(state="disabled")
        self.status.configure(text="运行中")
        self._log(f"开始处理 {len(config['alphas'])} 条表达式；模拟状态：未知")
        thread = threading.Thread(target=self._worker, args=(config, batch_size), daemon=True)
        thread.start()

    def _worker(self, config, batch_size):
        try:
            import contextlib
            import io
            import machine_lib
            settings = config["settings"]
            batches = machine_lib.load_task_pool_single(
                [(a["expression"], a["decay"]) for a in config["alphas"]], batch_size)
            with contextlib.redirect_stdout(QueueWriter(self.events)), contextlib.redirect_stderr(QueueWriter(self.events)):
                result = machine_lib.single_simulate(batches, settings["neutralization"],
                                                     settings["region"], settings["universe"], settings["start"])
            state = "未知（函数返回 None，无法确认任务完成状态）" if result is None else "未知（函数未提供可确认的状态）"
            self.events.put(("done", f"调用结束；模拟状态：{state}"))
        except Exception as exc:
            # Avoid rendering likely credential-like values from third-party exception text.
            detail = re.sub(r"(?i)(password|username|authorization|cookie)(\s*[:=]\s*)\S+", r"\1\2[已隐藏]", str(exc))
            self.events.put(("error", f"模拟调用出错：{detail}"))

    def _log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _drain_events(self):
        try:
            while True:
                kind, text = self.events.get_nowait()
                self._log(text)
                if kind in ("done", "error"):
                    self.running = False
                    self.start_button.configure(state="normal")
                    self.status.configure(text="结束；状态未知" if kind == "done" else "调用出错")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)


def main():
    root = tk.Tk()
    ResearchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
