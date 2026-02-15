import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class OntologyAlignmentEvaluator:
    def __init__(self, gt_path: str, logmap_threshold: float = 0.5):
        self.gt_pairs = self._load_gt(gt_path)
        self.logmap_threshold = logmap_threshold

    def _load_gt(self, path: str) -> set:
        df = pd.read_csv(path, sep="\t", header=None)
        return {frozenset((row[0], row[1])) for _, row in df.iterrows()}

    def attach_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Label"] = df.apply(
            lambda r: frozenset((r["Source"], r["Target"])) in self.gt_pairs,
            axis=1,
        )
        df["LogMapPred"] = df["LogMapScore"] > self.logmap_threshold
        return df

    def _metrics(self, y_true, y_pred) -> dict:
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape != (2, 2):
            cm = [[0, 0], [0, 0]]

        TP = cm[1][1]
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        youdens_index = sensitivity + specificity - 1

        return {
            "Precision": precision,
            "Recall": recall,
            "FN": FN,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "YoudenIndex": youdens_index,
            "ConfusionMatrix": cm,
            "F1": f1,
            "TP": TP,
            "TN": TN,
            "FP": FP,
        }

    def analyze_llm_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ChangedByLLM"] = df["LogMapPred"] != df["LLMDecision"]

        df["LLMHelped"] = (
            df["ChangedByLLM"] &
            (df["LLMDecision"] == df["Label"])
        )

        df["LLMHurt"] = (
            df["ChangedByLLM"] &
            (df["LLMDecision"] != df["Label"])
        )

        return df

    def _save_confusion_matrix(self, cm, save_path: Path, title: str) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def evaluate(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        experiment_type: str,
        prompts_used: str,
        results_dir: str = "results",
    ) -> dict:

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        results_path = (
            Path(results_dir)
            / timestamp
            / dataset_name
            / experiment_type
        )
        results_path.mkdir(parents=True, exist_ok=True)

        df = self.attach_labels(df)
        df = self.analyze_llm_impact(df)

        logmap_metrics = self._metrics(df["Label"], df["LogMapPred"])
        llm_metrics = self._metrics(df["Label"], df["LLMDecision"])

        impact = {
            "LLM_corrected_pairs": int(df["LLMHelped"].sum()),
            "LLM_introduced_errors": int(df["LLMHurt"].sum()),
            "Total_changed_by_LLM": int(df["ChangedByLLM"].sum()),
        }

        # SAVE METRICS CSV
        metrics_df = pd.DataFrame([
            {"System": "LogMap", **{k: v for k, v in logmap_metrics.items() if k != "ConfusionMatrix"}},
            {"System": "LogMap+LLM", **{k: v for k, v in llm_metrics.items() if k != "ConfusionMatrix"}},
        ])
        metrics_df.to_csv(results_path / "metrics.csv", index=False)

        # SAVE LLM IMPACT
        pd.DataFrame(
            list(impact.items()),
            columns=["Metric", "Value"]
        ).to_csv(results_path / "llm_impact.csv", index=False)

        # SAVE CONFUSION MATRICES
        self._save_confusion_matrix(
            logmap_metrics["ConfusionMatrix"],
            results_path / "confusion_matrix_logmap.png",
            "LogMap Confusion Matrix",
        )

        self._save_confusion_matrix(
            llm_metrics["ConfusionMatrix"],
            results_path / "confusion_matrix_logmap_llm.png",
            "LogMap + LLM Confusion Matrix",
        )

        # SAVE DETAILED RESULTS
        df.to_csv(results_path / "detailed_results.csv", index=False)

        with open(results_path / "prompts_used.txt", "w") as f:
            if isinstance(prompts_used, list):
                for prompt in prompts_used:
                    f.write(prompt.strip() + "\n\n")
            else:
                f.write(str(prompts_used).strip())

        return {
            "LogMap": logmap_metrics,
            "LogMap+LLM": llm_metrics,
            "LLM Impact": impact,
        }