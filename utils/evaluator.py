import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.explainability import (
    generate_gradcam,
    generate_gradcam_plus_plus
)

from utils.integrated_gradients import (
    generate_integrated_gradients
)

from utils.evaluation import (
    normalize_map,
    threshold_map,
    resize_mask,
    compute_iou,
    pointing_game,
    heatmap_coverage
)


class ExplainabilityEvaluator:

    def __init__(self, model, test_loader, device):

        self.model = model
        self.test_loader = test_loader
        self.device = device

        self.target_layers = [model.layer4[-1]]

        # Summary results
        self.results = []

        # Per-image results
        self.image_results = []

    def _evaluate_method(self, method_name):

        ious = []
        pointing_scores = []
        coverages = []

        self.model.eval()

        image_index = 0

        for images, masks, labels in tqdm(
            self.test_loader,
            desc=f"Evaluating {method_name}"
        ):

            images = images.to(self.device)
            labels = labels.to(self.device)

            # ---------------- Explanation ----------------

            if method_name == "Grad-CAM":

                explanation = generate_gradcam(
                    self.model,
                    images,
                    self.target_layers,
                    target_class=labels.item()
                )

            elif method_name == "Grad-CAM++":

                explanation = generate_gradcam_plus_plus(
                    self.model,
                    images,
                    self.target_layers,
                    target_class=labels.item()
                )

            elif method_name == "Integrated Gradients":

                explanation = generate_integrated_gradients(
                    model=self.model,
                    input_tensor=images,
                    target_class=labels.item()
                )

            else:
                raise ValueError("Unknown method")

            # ---------------- Normalize ----------------

            explanation = normalize_map(explanation)

            binary = threshold_map(
                explanation,
                threshold=0.5
            )

            gt_mask = masks.squeeze().cpu().numpy()

            gt_mask = resize_mask(
                gt_mask,
                explanation.shape
            )

            gt_mask = gt_mask > 0

            # ---------------- Metrics ----------------

            iou = compute_iou(
                binary,
                gt_mask
            )

            point = pointing_game(
                explanation,
                gt_mask
            )

            coverage = heatmap_coverage(
                explanation,
                gt_mask
            )

            ious.append(iou)
            pointing_scores.append(point)
            coverages.append(coverage)

            # Save every image result

            self.image_results.append({

                "Image Index": image_index,

                "Class": labels.item(),

                "Method": method_name,

                "IoU": iou,

                "Pointing Game": point,

                "Heatmap Coverage": coverage

            })

            image_index += 1

        result = {

            "Method": method_name,

            "Mean IoU": np.mean(ious),

            "Pointing Game": np.mean(pointing_scores),

            "Heatmap Coverage": np.mean(coverages)

        }

        self.results.append(result)

        return result

    def evaluate_gradcam(self):

        return self._evaluate_method("Grad-CAM")

    def evaluate_gradcampp(self):

        return self._evaluate_method("Grad-CAM++")

    def evaluate_integrated_gradients(self):

        return self._evaluate_method("Integrated Gradients")

    def save_results(

        self,

        summary_file="../results/explainability_summary.csv",

        detailed_file="../results/explainability_image_results.csv"

    ):

        summary_df = pd.DataFrame(self.results)

        detailed_df = pd.DataFrame(self.image_results)

        summary_df.to_csv(
            summary_file,
            index=False
        )

        detailed_df.to_csv(
            detailed_file,
            index=False
        )

        print("\nSummary Results")

        print(summary_df)

        print("\nSaved:", summary_file)

        print("Saved:", detailed_file)