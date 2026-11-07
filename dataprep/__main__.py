"""`uv run python -m dataprep` : load -> clean -> split -> manifest -> SFT jsonl -> token stats."""

from dataprep.formatting import write_jsonl
from dataprep.loader import load_clean
from dataprep.manifest import write_manifest
from dataprep.split import assign_splits
from dataprep.tokenstats import write as write_tokenstats
from settings.config import get_settings


def main() -> None:
    s = get_settings()
    df = assign_splits(load_clean(), s.seed)
    path = write_manifest(df, s.data_dir / "splits")
    write_jsonl(df, s.data_dir / "processed")
    stats = write_tokenstats(df, s.target_model_id, s.results_dir / "data")
    print(f"manifest -> {path}")
    print(df.groupby("split").agg(patients=("patient_id", "nunique"), annotations=("annotation_id", "size")))
    print(df.pivot_table(index="expert_eligibility", columns="split", values="annotation_id",
                         aggfunc="count", fill_value=0))
    print(f"token stats: max total={stats['overall']['total_tokens']['max']} "
          f"truncated={stats['n_truncated']} evidence_lost={stats['n_with_evidence_lost']}")


if __name__ == "__main__":
    main()
