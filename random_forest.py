{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOOsJiHOIUbpUsCMm7a1zEQ",
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
        "<a href=\"https://colab.research.google.com/github/evepineau/Workshop3/blob/main/random_forest.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "--TrIBlKsm9_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0c1255f0-4f7b-4602-905f-d37a149c78e4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Random Forest Accuracy: 1.00\n",
            "Random Forest model saved to C:\\Users\\evepi\\OneDrive - De Vinci\\ESILV\\A4\\S8\\Decentralization Technologies\\Workshop3_EvePINEAU_CDOF6\\Workshop3/random_forest_model.pkl\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "import pickle\n",
        "import numpy as np\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "# Define the save directory\n",
        "save_directory = r\"C:\\Users\\evepi\\OneDrive - De Vinci\\ESILV\\A4\\S8\\Decentralization Technologies\\Workshop3_EvePINEAU_CDOF6\\Workshop3\"\n",
        "os.makedirs(save_directory, exist_ok=True)\n",
        "\n",
        "# Load the Iris dataset\n",
        "iris = load_iris()\n",
        "X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)\n",
        "\n",
        "# Train Random Forest Model\n",
        "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "rf_model.fit(X_train, y_train)\n",
        "\n",
        "# Evaluate accuracy\n",
        "y_pred = rf_model.predict(X_test)\n",
        "print(f\"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}\")\n",
        "\n",
        "# Save the model\n",
        "model_path = os.path.join(save_directory, \"random_forest_model.pkl\")\n",
        "with open(model_path, \"wb\") as f:\n",
        "    pickle.dump(rf_model, f)\n",
        "print(f\"Random Forest model saved to {model_path}\")\n"
      ]
    }
  ]
}