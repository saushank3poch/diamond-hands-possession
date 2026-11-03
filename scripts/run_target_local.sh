#!/bin/sh
# Keep the caller's output descriptor only in this shell. Capture even Python
# startup errors; do not print paths, arguments, timing, or exception text.
exec 3>&1
exec >/dev/null 2>&1
umask 077
code=70
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || {
    printf 'TARGET_RUN FAIL exit_code=70\n' >&3
    exit 70
}

if [ "$#" -ge 3 ] && [ "$2" = "--" ] &&
   [ ! -L "$root/runs" ] && [ ! -L "$root/runs/target" ] &&
   mkdir -p "$root/runs/target"; then
    run_dir=$(mktemp -d "$root/runs/target/run-XXXXXXXX")
    if [ -n "$run_dir" ]; then
        config=$1
        shift 2
        (
            exec 3>&-
            cd "$root" || exit 70
            export POSSESSION_TARGET_LOG_DIR="$run_dir"
            "${PYTHON:-python3}" "$root/scripts/run_experiment.py" \
                --mode target --run-dir "$run_dir" --config "$config" -- "$@"
        ) >"$run_dir/bootstrap.log" 2>&1
        code=$?
    fi
else
    code=64
fi

if [ "$code" -eq 0 ]; then
    printf 'TARGET_RUN PASS exit_code=0\n' >&3
else
    printf 'TARGET_RUN FAIL exit_code=%s\n' "$code" >&3
fi
exit "$code"
