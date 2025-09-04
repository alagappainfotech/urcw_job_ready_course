"""
Module 4: Python Programs & Filesystem Interaction - Exercises and Labs
"""

from pathlib import Path
import argparse
import sys
import shutil

# Quick Checks

def qc_sysargv_demo():
    print("Script name:", sys.argv[0])
    print("Args:", sys.argv[1:])


def qc_pathlib_ops():
    p = Path.cwd()
    print("CWD:", p)
    print("Exists:", p.exists())
    print("Entries (first 5):", [x.name for x in list(p.iterdir())[:5]])


# Try This

def try_argparse_demo():
    parser = argparse.ArgumentParser(description="Demo CLI")
    parser.add_argument("path", help="Directory to list")
    parser.add_argument("-a", "--all", action="store_true", help="Show all entries")
    # For notebook/demo use: avoid parsing real sys.argv
    args = parser.parse_args([])
    d = Path(args.path) if args.path else Path.cwd()
    items = list(d.iterdir())
    if not args.all:
        items = [i for i in items if not i.name.startswith('.')]
    print("Items:", [i.name for i in items[:10]])


def try_file_copy_move(tmp_dir: str = "./m4_tmp"):
    d = Path(tmp_dir)
    d.mkdir(exist_ok=True)
    src = d / "sample.txt"
    src.write_text("hello")
    dst = d / "backup.txt"
    shutil.copy(src, dst)
    print("Copied:", src, "->", dst)


# Labs

def lab_wc_like_stats(path: str = "./"):
    """Compute lines, words, chars for all .py files recursively."""
    base = Path(path)
    files = list(base.rglob("*.py"))
    total = {"lines": 0, "words": 0, "chars": 0}
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        words = text.split()
        stats = {"lines": len(lines), "words": len(words), "chars": len(text)}
        total = {k: total[k] + stats[k] for k in total}
    print("Total across .py files:", total)


def lab_backup_dot_test(src: str = "./", dest: str = "./backup_test"):
    """Move all *.test files (non-symlinks) into backup directory."""
    s = Path(src)
    t = Path(dest)
    t.mkdir(exist_ok=True)
    moved = 0
    for p in s.rglob("*.test"):
        if not p.is_symlink():
            target = t / p.name
            p.replace(target)
            moved += 1
    print(f"Moved {moved} .test files to {t}")


def try_fileinput_stream_demo():
    """Stream lines from stdin or files using fileinput (demo pattern)."""
    import fileinput
    count = 0
    for line in fileinput.input(files=(), mode='r'):
        count += 1
    print("Streamed lines:", count)


if __name__ == "__main__":
    qc_sysargv_demo()
    qc_pathlib_ops()
    try_argparse_demo()
    try_file_copy_move()
    try_fileinput_stream_demo()
    lab_wc_like_stats()
    lab_backup_dot_test()


