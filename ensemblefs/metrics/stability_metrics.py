from ensemblefs.core.novovicova import StabilityNovovicova


def compute_stability_metrics(features_list):
    """
    Computes the stability metrics using the StabilityNovovicova class.

    Args:
        features_list (list of lists): Each sublist represents features selected in one dataset.

    Returns:
        float: The computed stability measure SH(S).
    """
    # Initialize the StabilityNovovicova class
    stability_calculator = StabilityNovovicova(features_list)

    # Compute the stability
    result = stability_calculator.compute_stability()

    return result
