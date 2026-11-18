import os
import numpy as np
from scipy.stats import ks_2samp


def check_data_drift():

    print("Running Data Drift Check (KS-Test)...")

    np.random.seed(42)

    # Reference data
    reference_data = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=(100, 4)
    )

    # Current production data
    current_data = np.random.normal(
        loc=0.4,
        scale=1.1,
        size=(100, 4)
    )

    drift_found = False
    threshold = 0.05

    for i in range(reference_data.shape[1]):

        stat, p_value = ks_2samp(
            reference_data[:, i],
            current_data[:, i]
        )

        if p_value < threshold:

            print(
                f"Drift detected in feature {i} "
                f"(p-value: {p_value:.4f})"
            )

            drift_found = True

        else:
            print(
                f"Feature {i} stable "
                f"(p-value: {p_value:.4f})"
            )

    # Send result to GitHub Actions
    github_output = os.getenv("GITHUB_OUTPUT")

    if github_output:

        with open(github_output, "a") as f:

            f.write(
                f"drift_detected="
                f"{'true' if drift_found else 'false'}\n"
            )

    return drift_found


if __name__ == "__main__":
    check_data_drift()