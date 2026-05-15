from __future__ import annotations

from salary_pipeline import train_and_save


def main() -> None:
    model_card = train_and_save()
    print("Best model:", model_card["best_model"])
    for result in model_card["results"]:
        print(
            f"{result['model_name']}: "
            f"train={result['train_accuracy']:.4f} "
            f"test={result['test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
