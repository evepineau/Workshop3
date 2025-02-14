{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPvBUsRIijz55Hjy0meOmAh",
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
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hq_L6lR5tNM1",
        "outputId": "18864ebb-bd3a-427b-baaf-203825449099"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'individual_predictions': {'Logistic Regression': 'Setosa', 'Random Forest': 'Setosa', 'SVM': 'Setosa'}, 'final_prediction': 'Setosa'}\n"
          ]
        }
      ],
      "source": [
        "import requests\n",
        "from collections import Counter\n",
        "\n",
        "# API endpoints\n",
        "MODEL_URLS = {\n",
        "    \"logistic\": \"http://127.0.0.1:5001/predict\",\n",
        "    \"random_forest\": \"http://127.0.0.1:5002/predict\",\n",
        "    \"svm\": \"http://127.0.0.1:5003/predict\"\n",
        "}\n",
        "\n",
        "# Model weights (Proof-of-Stake simulation)\n",
        "MODEL_WEIGHTS = {\n",
        "    \"logistic\": 1.0,\n",
        "    \"random_forest\": 1.2,\n",
        "    \"svm\": 0.8\n",
        "}\n",
        "\n",
        "# Sample features\n",
        "params = {\n",
        "    \"sepal_length\": 5.1,\n",
        "    \"sepal_width\": 3.5,\n",
        "    \"petal_length\": 1.4,\n",
        "    \"petal_width\": 0.2\n",
        "}\n",
        "\n",
        "def get_prediction():\n",
        "    weighted_predictions = []\n",
        "\n",
        "    for model_name, url in MODEL_URLS.items():\n",
        "        response = requests.get(url, params=params)\n",
        "        if response.status_code == 200:\n",
        "            pred = response.json()[\"prediction\"]\n",
        "            weighted_predictions.extend([pred] * int(MODEL_WEIGHTS[model_name] * 10))\n",
        "\n",
        "    return weighted_predictions\n",
        "\n",
        "def meta_model_prediction():\n",
        "    predictions = get_prediction()\n",
        "    final_prediction = Counter(predictions).most_common(1)[0][0]\n",
        "\n",
        "    return {\n",
        "        \"individual_predictions\": predictions,\n",
        "        \"final_prediction\": final_prediction\n",
        "    }\n",
        "\n",
        "print(meta_model_prediction())"
      ]
    }
  ]
}