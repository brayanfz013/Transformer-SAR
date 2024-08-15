def calculate_best_model(data):
	"""
	Analyzes a JSON structure containing model performance data and identifies
	the model with the best overall performance based on:
	    - Lowest PSNR (Peak Signal-to-Noise Ratio)
	    - Highest SSIM (Structural Similarity Index Measure)
	    - Lowest MSE (Mean Squared Error)

	Args:
	    data (dict): A dictionary representing the JSON data containing
	                 model performance information.

	Returns:
	    str: The name of the model with the best overall performance.
	"""

	best_model = None
	best_score = float("inf")  # Initialize with positive infinity

	for model_name, model_data in data.items():
		for entry in model_data:
			score = entry["psnr"] + (1 - entry["ssim"]) + entry["mse"]
			# Combine PSNR, SSIM (inverted), and MSE into a single score

			if score < best_score:
				best_model = model_name
				best_score = score

	return best_model
