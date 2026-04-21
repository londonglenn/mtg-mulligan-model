from __future__ import annotations

import traceback

import ingest
import preprocess
import train
import evaluate


def main(
    do_ingest: bool = True,
    update_legacy_global: bool = True,
    set_latest: bool = True,
) -> dict:
    """
    Full end-to-end pipeline:

        ingest -> preprocess -> train -> evaluate

    Returns a summary dictionary with:
        - dataset_id
        - run_id
    """
    try:
        if do_ingest:
            print("\n=== INGEST ===")
            ingest.main()

        print("\n=== PREPROCESS ===")
        dataset_id = preprocess.main(update_legacy_global=update_legacy_global)

        print("\n=== TRAIN ===")
        run_id = train.main(dataset_id=dataset_id, set_latest=set_latest)

        print("\n=== EVALUATE ===")
        evaluate.main(run_id=run_id)

        print("\n=== PIPELINE COMPLETE ===")
        print(f"Dataset ID: {dataset_id}")
        print(f"Run ID: {run_id}")

        return {
            "dataset_id": dataset_id,
            "run_id": run_id,
        }

    except Exception as e:
        print(f"\nPipeline failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()