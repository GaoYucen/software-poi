# Long-Running Training Jobs

For long-running training and experiment commands:

1. Never poll a training log file at fixed intervals.
2. Never repeatedly run `tail`, `cat`, `head`, or similar commands while a job is running.
3. Launch long-running training in a detached process or tmux session.
4. During execution, check only a compact process status or one-line status file.
5. Do not read the training log while the job status is RUNNING.
6. If waiting for completion is necessary, use one silent blocking wait command that produces no periodic output.
7. After the process exits:
   - On success, read only the final metrics or summary file.
   - On failure, read only the last 50–80 relevant log lines.
8. Never use `tail -f` inside the Cline agent loop.
9. Full logs are for human inspection, not for repeated LLM context ingestion.
10. Progress output from tqdm must not be repeatedly sent to the model.