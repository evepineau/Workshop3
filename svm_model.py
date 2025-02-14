{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMm7O64U0JXhti67f7vdbGI",
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
        "<a href=\"https://colab.research.google.com/github/evepineau/Workshop3/blob/main/svm_model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "Xg3dkmEAs3Yl",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7a0d3e0c-cc75-4269-b552-d945a7895345"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "SVM Accuracy: 1.00\n",
            "SVM model saved to C:\\Users\\evepi\\OneDrive - De Vinci\\ESILV\\A4\\S8\\Decentralization Technologies\\Workshop3_EvePINEAU_CDOF6\\Workshop3/svm_model.pkl\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "import pickle\n",
        "import numpy as np\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.svm import SVC\n",
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
        "# Train Support Vector Machine (SVM) Model\n",
        "svm_model = SVC(probability=True)\n",
        "svm_model.fit(X_train, y_train)\n",
        "\n",
        "# Evaluate accuracy\n",
        "y_pred = svm_model.predict(X_test)\n",
        "print(f\"SVM Accuracy: {accuracy_score(y_test, y_pred):.2f}\")\n",
        "\n",
        "# Save the model\n",
        "model_path = os.path.join(save_directory, \"svm_model.pkl\")\n",
        "with open(model_path, \"wb\") as f:\n",
        "    pickle.dump(svm_model, f)\n",
        "print(f\"SVM model saved to {model_path}\")\n"
      ]
    }
  ]
}