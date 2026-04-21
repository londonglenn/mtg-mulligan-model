from __future__ import annotations

import traceback

import ingest
import preprocess
import train
import evaluate
import publish_model


def main(
    do_ingest: bool = True,
    update_legacy_global: bool = True,
    set_latest: bool = True,
    do_publish: bool = False,
    set_published_latest: bool = True,
) -> dict:
    """
    Full end-to-end pipeline:

        ingest -> preprocess -> train -> evaluate -> [optional publish]

    Returns a summary dictionary with:
        - dataset_id
        - run_id
        - optional publish_result
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

        result = {
            "dataset_id": dataset_id,
            "run_id": run_id,
        }

        if do_publish:
            print("\n=== PUBLISH ===")
            publish_result = publish_model.main(
                run_id=run_id,
                set_latest=set_published_latest,
            )
            result["publish_result"] = publish_result

        print("\n=== PIPELINE COMPLETE ===")
        print(f"Dataset ID: {dataset_id}")
        print(f"Run ID: {run_id}")
        if do_publish:
            print("Publish step completed.")

        return result

    except Exception as e:
        print(f"\nPipeline failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()