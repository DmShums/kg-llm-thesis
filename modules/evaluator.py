import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class OntologyAlignmentEvaluator:
    def __init__(self, gt_path: str, logmap_threshold: float = 0.5):
        self.gt_pairs = self._load_gt(gt_path)
        self.logmap_threshold = logmap_threshold

    def _load_gt(self, path: str) -> set:
        df = pd.read_csv(path, sep="|", header=None)
        return {frozenset((r[0], r[1])) for _, r in df.iterrows()}

    def attach_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Label"] = df.apply(
            lambda r: frozenset((r["Source"], r["Target"])) in self.gt_pairs,
            axis=1,
        )
        df["LogMapPred"] = df["LogMapScore"] > self.logmap_threshold
        return df

    def _metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return {
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "ConfusionMatrix": cm,
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

    def token_statistics(self, df: pd.DataFrame) -> dict:
        tokens = df[["TokensIn", "TokensOut"]].to_numpy()
        return {
            "MeanInputTokens": tokens[:, 0].mean(),
            "MeanOutputTokens": tokens[:, 1].mean(),
            "TotalInputTokens": tokens[:, 0].sum(),
            "TotalOutputTokens": tokens[:, 1].sum(),
        }

    def plot_confidence(self, df: pd.DataFrame):
        correct = df[df["LLMDecision"] == df["Label"]]
        wrong = df[df["LLMDecision"] != df["Label"]]

        plt.hist(correct["LLMConfidence"], bins=30, alpha=0.6, label="Correct")
        plt.hist(wrong["LLMConfidence"], bins=30, alpha=0.6, label="Wrong")
        plt.legend()
        plt.title("LLM Confidence vs Correctness")
        plt.show()

    def evaluate(self, df: pd.DataFrame, plot: bool = True) -> dict:
        df = self.attach_labels(df)
        df = self.analyze_llm_impact(df)

        logmap_metrics = self._metrics(df["Label"], df["LogMapPred"])
        llm_metrics = self._metrics(df["Label"], df["LLMDecision"])

        impact = {
            "LLM_corrected_pairs": int(df["LLMHelped"].sum()),
            "LLM_introduced_errors": int(df["LLMHurt"].sum()),
            "Total_changed_by_LLM": int(df["ChangedByLLM"].sum()),
        }

        token_stats = self.token_statistics(df)

        if plot:
            self.plot_confidence(df)

        report = {
            "LogMap": logmap_metrics,
            "LogMap+LLM": llm_metrics,
            "LLM Impact": impact,
            "Token Usage": token_stats,
        }

        return report, df
