from pathlib import Path

_TMP_FILE_DIR = Path("/Volumes/data/haga/data/inter_files")

ICA_EPOCHS = _TMP_FILE_DIR / "ica_epochs"
AVG_ERPS = _TMP_FILE_DIR / "avg_erps"


_inter_dirs = [ICA_EPOCHS, AVG_ERPS]
for dir in _inter_dirs:
    if not dir.exists():
        dir.mkdir(parents=True)
