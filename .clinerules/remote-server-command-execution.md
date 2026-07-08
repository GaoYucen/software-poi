# Remote Server Command Execution Rules

This project runs on a remote Linux server through VS Code Remote SSH.

1. Use the Python interpreter from the py11 Conda environment directly.
   - py11 environment path: `/opt/conda/envs/py11`
   - py11 Python executable: `/opt/conda/envs/py11/bin/python`
2. Prefer the absolute Python executable path (`/opt/conda/envs/py11/bin/python`) instead of `conda activate`.
3. Do not use shell heredocs such as `python - <<'PY'`.
4. Do not combine multiple diagnostic commands with `&&`.
5. Run diagnostic commands one at a time.
6. If `conda run` is necessary, always use `--no-capture-output`.
7. Prefer short, non-interactive commands that terminate explicitly.
8. Proxy environment variables may be present by default, for example:
   - `HTTP_PROXY=http://127.0.0.1:7897`
   - `HTTPS_PROXY=http://127.0.0.1:7897`
   - `ALL_PROXY=socks5h://127.0.0.1:7897`
   But `127.0.0.1` is resolved on the remote server itself, not on the local workstation.

## Long-running training commands

Never run long-running ML training, pretraining, fine-tuning, or evaluation
commands in the foreground.

For long-running experiments:

1. Run the process detached with `nohup` or `tmux`.
2. Redirect both stdout and stderr to a log file.
3. Do not stream or repeatedly read tqdm output.
4. When checking status, read at most the last 30-50 lines.
5. Prefer filtered summaries such as:
   - epoch
   - train/validation loss
   - evaluation metrics
   - best checkpoint
   - warning
   - error
   - traceback
6. Never read a complete large training log unless explicitly requested.
