"""
parquet_to_csv.py
批量将 Parquet 文件转换为 CSV 文件，带基本鲁棒性
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path


def convert_parquet_to_csv(
    parquet_path: Path, csv_path: Path, overwrite: bool = False, encoding: str = "utf-8"
):
    """
    将单个 Parquet 文件转换为 CSV 文件
    """
    if not parquet_path.exists():
        print(f"[ERROR] Parquet 文件不存在: {parquet_path}")
        return False

    if csv_path.exists() and not overwrite:
        print(f"[SKIP] CSV 文件已存在且未设置覆盖: {csv_path}")
        return True

    try:
        df = pd.read_parquet(parquet_path)
        # 自动创建父目录
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding=encoding)
        print(f"[OK] {parquet_path} → {csv_path}")
        return True
    except Exception as e:
        print(f"[ERROR] 转换失败 {parquet_path} → {csv_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Parquet 转 CSV 工具")
    parser.add_argument("input", type=str, help="输入 Parquet 文件或目录")
    parser.add_argument("output", type=str, help="输出 CSV 文件或目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 CSV")
    parser.add_argument("--encoding", type=str, default="utf-8", help="CSV 文件编码")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # 单文件模式
        if output_path.is_dir():
            csv_file = output_path / (input_path.stem + ".csv")
        else:
            csv_file = output_path
        convert_parquet_to_csv(
            input_path, csv_file, overwrite=args.overwrite, encoding=args.encoding
        )
    elif input_path.is_dir():
        # 目录模式
        parquet_files = list(input_path.rglob("*.parquet"))
        if not parquet_files:
            print(f"[WARN] 未找到任何 Parquet 文件: {input_path}")
            sys.exit(0)

        for pq_file in parquet_files:
            # 输出路径保持原目录结构
            relative_path = pq_file.relative_to(input_path).with_suffix(".csv")
            csv_file = output_path / relative_path
            convert_parquet_to_csv(
                pq_file, csv_file, overwrite=args.overwrite, encoding=args.encoding
            )
    else:
        print(f"[ERROR] 输入路径不存在: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
