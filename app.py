{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMMhCn8Q/Qmb/2nbTsD/bUV"
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
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "SalsbEma05Vc"
      },
      "outputs": [],
      "source": [
        "from flask import Flask, request, jsonify\n",
        "import joblib\n",
        "import numpy as np\n",
        "from sklearn.datasets import load_iris\n",
        "\n",
        "app = Flask(__name__)\n",
        "\n",
        "# Load models\n",
        "models = {\n",
        "    \"logistic\": joblib.load(\"logistic_model.pkl\"),\n",
        "    \"random_forest\": joblib.load(\"random_forest_model.pkl\"),\n",
        "    \"svm\": joblib.load(\"svm_model.pkl\"),\n",
        "}\n",
        "\n",
        "# Load dataset for class labels\n",
        "iris = load_iris()\n",
        "target_names = iris.target_names\n",
        "\n",
        "@app.route('/predict', methods=['GET'])\n",
        "def predict():\n",
        "    model_type = request.args.get(\"model\")\n",
        "    if model_type not in models:\n",
        "        return jsonify({\"error\": \"Model type not found. Choose logistic, random_forest, or svm.\"})\n",
        "\n",
        "    try:\n",
        "        features = [float(request.args.get(param)) for param in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]\n",
        "        features = np.array([features])\n",
        "        model = models[model_type]\n",
        "\n",
        "        prediction = model.predict_proba(features)[0]\n",
        "        response = {\"probabilities\": prediction.tolist(), \"class_labels\": target_names.tolist()}\n",
        "\n",
        "        return jsonify(response)\n",
        "    except Exception as e:\n",
        "        return jsonify({\"error\": str(e)})\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    app.run(host='0.0.0.0', port=5000, debug=True)"
      ]
    }
  ]
}