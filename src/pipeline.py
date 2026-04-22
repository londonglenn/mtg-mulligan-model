from __future__ import annotations

import traceback

import ingest
import preprocess
import train
import evaluate
import publish_model


FULL_FEATURE_EXPERIMENT_ID = "full"

DEFAULT_MODEL_TYPES = [
    "logreg_l1",
    "xgboost",
    "lightgbm",
]

DEFAULT_ABLATION_EXPERIMENT_IDS = [
    "full",
    "no_mana_buckets",
    "land_only",
    "cards_only",
]


def prompt_pipeline_mode() -> str:
    print("\nSelect pipeline mode:")
    print("  1) Single full-feature study (all model types)")
    print("  2) Full ablation study (all model types × feature ablations)")

    while True:
        choice = input("\nEnter 1 or 2: ").strip()

        if choice == "1":
            return "single"
        if choice == "2":
            return "ablation"

        print("Invalid selection. Please enter 1 or 2.")


def run_single_experiment(
    dataset_id: str,
    experiment_id: str,
    model_type: str,
    set_latest: bool = True,
    do_publish: bool = False,
    set_published_latest: bool = True,
) -> dict:
    print(f"\n=== TRAIN: model_type={model_type}, experiment_id={experiment_id} ===")
    run_id = train.main(
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        model_type=model_type,
        set_latest=set_latest,
    )

    print(f"\n=== EVALUATE: model_type={model_type}, experiment_id={experiment_id} ===")
    evaluate.main(run_id=run_id)

    result = {
        "model_type": model_type,
        "experiment_id": experiment_id,
        "run_id": run_id,
    }

    if do_publish:
        print(f"\n=== PUBLISH: model_type={model_type}, experiment_id={experiment_id} ===")
        publish_result = publish_model.main(
            run_id=run_id,
            set_latest=set_published_latest,
        )
        result["publish_result"] = publish_result

    return result


def run_model_comparison(
    dataset_id: str,
    model_types: list[str] | None = None,
    experiment_id: str = FULL_FEATURE_EXPERIMENT_ID,
    set_latest: bool = False,
    do_publish: bool = False,
    set_published_latest: bool = False,
) -> list[dict]:
    if model_types is None:
        model_types = DEFAULT_MODEL_TYPES

    results = []

    for model_type in model_types:
        result = run_single_experiment(
            dataset_id=dataset_id,
            experiment_id=experiment_id,
            model_type=model_type,
            set_latest=set_latest,
            do_publish=do_publish,
            set_published_latest=set_published_latest,
        )
        results.append(result)

    return results


def run_ablation_grid(
    dataset_id: str,
    model_types: list[str] | None = None,
    experiment_ids: list[str] | None = None,
    set_latest: bool = False,
    do_publish: bool = False,
    set_published_latest: bool = False,
) -> list[dict]:
    if model_types is None:
        model_types = DEFAULT_MODEL_TYPES

    if experiment_ids is None:
        experiment_ids = DEFAULT_ABLATION_EXPERIMENT_IDS

    results = []

    for model_type in model_types:
        for experiment_id in experiment_ids:
            result = run_single_experiment(
                dataset_id=dataset_id,
                experiment_id=experiment_id,
                model_type=model_type,
                set_latest=set_latest,
                do_publish=do_publish,
                set_published_latest=set_published_latest,
            )
            results.append(result)

    return results


def main(
    do_ingest: bool = True,
    update_legacy_global: bool = True,
    set_latest: bool = True,
    do_publish: bool = False,
    set_published_latest: bool = True,
) -> dict:
    try:
        mode = prompt_pipeline_mode()

        if do_ingest:
            print("\n=== INGEST ===")
            ingest.main()

        print("\n=== PREPROCESS ===")
        dataset_id = preprocess.main(update_legacy_global=update_legacy_global)

        if mode == "ablation":
            print("\n=== FULL ABLATION GRID ===")
            ablation_results = run_ablation_grid(
                dataset_id=dataset_id,
                model_types=DEFAULT_MODEL_TYPES,
                experiment_ids=DEFAULT_ABLATION_EXPERIMENT_IDS,
                set_latest=False,
                do_publish=False,
                set_published_latest=False,
            )

            result = {
                "mode": "ablation",
                "dataset_id": dataset_id,
                "ablation_results": ablation_results,
            }

            print("\n=== PIPELINE COMPLETE ===")
            print(f"Mode: {result['mode']}")
            print(f"Dataset ID: {dataset_id}")
            print("Ablation runs:")
            for item in ablation_results:
                print(
                    f"  - model_type={item['model_type']}, "
                    f"experiment_id={item['experiment_id']}: "
                    f"{item['run_id']}"
                )

            return result

        # Single full-feature model comparison
        comparison_results = run_model_comparison(
            dataset_id=dataset_id,
            model_types=DEFAULT_MODEL_TYPES,
            experiment_id=FULL_FEATURE_EXPERIMENT_ID,
            set_latest=False,
            do_publish=False,
            set_published_latest=False,
        )

        final_result = {
            "mode": "single",
            "dataset_id": dataset_id,
            "comparison_results": comparison_results,
        }

        print("\n=== PIPELINE COMPLETE ===")
        print(f"Mode: {final_result['mode']}")
        print(f"Dataset ID: {dataset_id}")
        print("Model comparison runs:")
        for item in comparison_results:
            print(
                f"  - model_type={item['model_type']}, "
                f"experiment_id={item['experiment_id']}: "
                f"{item['run_id']}"
            )

        return final_result

    except Exception as e:
        print(f"\nPipeline failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()