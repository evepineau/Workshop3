{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNrNcd6N3JHxnckbcZ8+ffW",
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
      "execution_count": null,
      "metadata": {
        "id": "--TrIBlKsm9_"
      },
      "outputs": [],
      "source": [
        "import pickle\n",
        "import numpy as np\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "iris = load_iris()\n",
        "X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)\n",
        "\n",
        "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "rf_model.fit(X_train, y_train)\n",
        "\n",
        "y_pred = rf_model.predict(X_test)\n",
        "print(f\"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}\")\n",
        "\n",
        "with open(\"random_forest_model.pkl\", \"wb\") as f:\n",
        "    pickle.dump(rf_model, f)\n",
        "print(\"Random Forest model saved.\")"
      ]
    }
  ]
}