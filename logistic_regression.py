{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMmIjKeE3vSriOayUbg8Ke6",
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
        "<a href=\"https://colab.research.google.com/github/evepineau/Workshop3/blob/main/logistic_regression.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "avPuHbY-sAQf"
      },
      "outputs": [],
      "source": [
        "import pickle\n",
        "import numpy as np\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "iris = load_iris()\n",
        "X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)\n",
        "\n",
        "log_model = LogisticRegression(max_iter=200)\n",
        "log_model.fit(X_train, y_train)\n",
        "\n",
        "y_pred = log_model.predict(X_test)\n",
        "print(f\"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.2f}\")\n",
        "\n",
        "with open(\"logistic_model.pkl\", \"wb\") as f:\n",
        "    pickle.dump(log_model, f)\n",
        "print(\"Logistic Regression model saved.\")"
      ]
    }
  ]
}