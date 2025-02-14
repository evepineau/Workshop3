{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMl46e2+2mg3erNvQwaWdr8",
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
      }
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "avPuHbY-sAQf",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d5eafd26-61fd-487a-cf27-75399a02a0fe"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Logistic Regression Accuracy: 1.00\n",
            "Logistic Regression model saved to C:\\Users\\evepi\\OneDrive - De Vinci\\ESILV\\A4\\S8\\Decentralization Technologies\\Workshop3_EvePINEAU_CDOF6\\Workshop3/logistic_model.pkl\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "import pickle\n",
        "import numpy as np\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LogisticRegression\n",
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
        "# Train Logistic Regression Model\n",
        "log_model = LogisticRegression(max_iter=200)\n",
        "log_model.fit(X_train, y_train)\n",
        "\n",
        "# Evaluate accuracy\n",
        "y_pred = log_model.predict(X_test)\n",
        "print(f\"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.2f}\")\n",
        "\n",
        "# Save the model\n",
        "model_path = os.path.join(save_directory, \"logistic_model.pkl\")\n",
        "with open(model_path, \"wb\") as f:\n",
        "    pickle.dump(log_model, f)\n",
        "print(f\"Logistic Regression model saved to {model_path}\")"
      ]
    }
  ]
}
