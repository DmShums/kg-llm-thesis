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
        # Label based on GT
        df["Label"] = df.apply(
            lambda r: frozenset((r.get("Source"), r.get("Target"))) in self.gt_pairs, axis=1
        )

        # LogMap prediction, safe fallback
        df["LogMapPred"] = df.get("LogMapScore", pd.Series([0.0]*len(df))) > self.logmap_threshold
        df["LogMapPred"] = df["LogMapPred"].fillna(False)
        return df

    def _metrics(self, y_true, y_pred) -> dict:
        if y_true is None or y_pred is None or len(y_true) != len(y_pred):
            return {k: pd.NA for k in ["Precision","Recall","F1","TP","TN","FP","FN","Sensitivity","Specificity","YoudenIndex","ConfusionMatrix"]}

        cm = confusion_matrix(y_true, y_pred, labels=[False, True])
        if cm.shape != (2, 2):
            cm = [[0, 0], [0, 0]]

        TN, FP = cm[0]
        FN, TP = cm[1]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        youdens_index = sensitivity + specificity - 1

        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "YoudenIndex": youdens_index,
            "ConfusionMatrix": cm,
        }

    def _save_confusion_matrix(self, cm, save_path: Path, title: str) -> None:
        if cm is None or len(cm) == 0:
            print(f"[WARN] Cannot save confusion matrix for {title}, skipping.")
            return
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
        prompts_used,
        second_system_name: str = "LLM",
        second_system_pred_col: str = "LLMDecision",
        results_dir: str = "results",
        display_logmap_decision: bool = False
    ) -> dict:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        results_path = Path(results_dir) / timestamp / dataset_name / experiment_type
        results_path.mkdir(parents=True, exist_ok=True)

        df = self.attach_labels(df)

        logmap_metrics = self._metrics(df["Label"], df["LogMapPred"])

        if second_system_pred_col in df.columns:
            second_metrics = self._metrics(df["Label"], df[second_system_pred_col])
        else:
            print(f"[WARN] Column '{second_system_pred_col}' not found. Filling NA metrics.")
            second_metrics = {k: pd.NA for k in logmap_metrics.keys()}

        # Save metrics CSV
        metrics_df = pd.DataFrame([
            {"System": "LogMap", **{k: v for k, v in logmap_metrics.items() if k != "ConfusionMatrix"}},
            {"System": second_system_name, **{k: v for k, v in second_metrics.items() if k != "ConfusionMatrix"}},
        ])
        metrics_df.drop_duplicates(subset="System", inplace=True)
        metrics_df.to_csv(results_path / "metrics.csv", index=False)

        # Save confusion matrices safely
        self._save_confusion_matrix(logmap_metrics.get("ConfusionMatrix"), results_path / "confusion_matrix_logmap.png", "LogMap Confusion Matrix")
        self._save_confusion_matrix(second_metrics.get("ConfusionMatrix"), results_path / f"confusion_matrix_{second_system_name}.png", f"{second_system_name} Confusion Matrix")

        # Save detailed results
        df.to_csv(results_path / "detailed_results.csv", index=False)

        # Save prompts used
        with open(results_path / "prompts_used.txt", "w") as f:
            if isinstance(prompts_used, list):
                for prompt in prompts_used:
                    f.write(prompt.strip() + "\n\n")
            else:
                f.write(str(prompts_used).strip())
        
        if "LLMTotalTokens" in df.columns:
            token_stats = {
                "total_tokens": df["LLMTotalTokens"].sum(skipna=True),
                "avg_tokens_per_call": df["LLMTotalTokens"].mean(skipna=True),
                "max_tokens": df["LLMTotalTokens"].max(skipna=True),
                "min_tokens": df["LLMTotalTokens"].min(skipna=True),
            }

            pd.DataFrame([token_stats]).to_csv(results_path / "token_usage_summary.csv", index=False)
        
        if display_logmap_decision:
            return {"LogMap": logmap_metrics, second_system_name: second_metrics}
        else:
            return {second_system_name: second_metrics}

    # ============================================================
    # New method: evaluate mediator systems
    # ============================================================
    def evaluate_labeled_mappings(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        system_name: str,
        results_dir: str = "results"
    ) -> dict:
        """
        Evaluate a set of mappings that already contain a Prediction column (True/False).
        Computes metrics against the ground truth stored in self.gt_pairs.
        """
        # -------------------------
        # Prepare output folder
        # -------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        results_path = Path(results_dir) / timestamp / dataset_name / system_name
        results_path.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # Compute Label column
        # -------------------------
        df["Label"] = df.apply(lambda r: frozenset((r["Source"], r["Target"])) in self.gt_pairs, axis=1)

        # -------------------------
        # Use Prediction column directly
        # -------------------------
        y_true = df["Label"].tolist()
        y_pred = df["Prediction"].tolist()

        # -------------------------
        # Compute metrics
        # -------------------------
        cm = confusion_matrix(y_true, y_pred, labels=[False, True])
        if cm.shape != (2, 2):
            cm = [[0, 0], [0, 0]]

        TN, FP = cm[0]
        FN, TP = cm[1]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        youden_index = sensitivity + specificity - 1

        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "YoudenIndex": youden_index,
            "ConfusionMatrix": cm
        }

        # -------------------------
        # Save metrics
        # -------------------------
        metrics_df = pd.DataFrame([{"System": system_name, **{k: v for k, v in metrics.items() if k != "ConfusionMatrix"}}])
        metrics_df.to_csv(results_path / "metrics.csv", index=False)

        # Save confusion matrix
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{system_name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(results_path / f"confusion_matrix_{system_name}.png")
        plt.close()

        # Save detailed results
        df.to_csv(results_path / "detailed_results.csv", index=False)

        return {system_name: metrics}