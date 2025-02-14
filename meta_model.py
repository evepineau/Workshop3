{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPeczt0qM00+ompj6mvu4QS",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/evepineau/Workshop3/blob/main/meta_model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hq_L6lR5tNM1"
      },
      "outputs": [],
      "source": [
        "import pickle\n",
        "import numpy as np\n",
        "from collections import Counter\n",
        "\n",
        "def load_model(model_path):\n",
        "    with open(model_path, \"rb\") as f:\n",
        "        return pickle.load(f)\n",
        "\n",
        "def get_prediction(features):\n",
        "    models = {\n",
        "        \"Logistic Regression\": load_model(\"logistic_model.pkl\"),\n",
        "        \"Random Forest\": load_model(\"random_forest_model.pkl\"),\n",
        "        \"SVM\": load_model(\"svm_model.pkl\"),\n",
        "    }\n",
        "\n",
        "    predictions = {}\n",
        "    for name, model in models.items():\n",
        "        pred = model.predict([features])[0]\n",
        "        predictions[name] = [\"Setosa\", \"Versicolor\", \"Virginica\"][pred]\n",
        "\n",
        "    return predictions\n",
        "\n",
        "def meta_model_prediction(features):\n",
        "    predictions = get_prediction(features)\n",
        "    prediction_values = list(predictions.values())\n",
        "\n",
        "    final_prediction = Counter(prediction_values).most_common(1)[0][0]\n",
        "\n",
        "    return {\n",
        "        \"individual_predictions\": predictions,\n",
        "        \"final_prediction\": final_prediction\n",
        "    }\n",
        "\n",
        "test_features = [5.1, 3.5, 1.4, 0.2]\n",
        "print(meta_model_prediction(test_features))\n"
      ]
    }
  ]
}