{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMVwmC2MxU9uyD/Gav3Twu/"
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
        "id": "Wzsh4yo058JD"
      },
      "outputs": [],
      "source": [
        "from flask import Flask, request, jsonify\n",
        "import joblib\n",
        "import numpy as np\n",
        "\n",
        "app = Flask(__name__)\n",
        "\n",
        "# Load model\n",
        "model = joblib.load(\"logistic_model.pkl\")\n",
        "\n",
        "@app.route('/predict', methods=['GET'])\n",
        "def predict():\n",
        "    try:\n",
        "        features = [float(request.args.get(param)) for param in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]\n",
        "        features = np.array([features])\n",
        "        prediction = model.predict(features)[0]\n",
        "        return jsonify({\"prediction\": int(prediction)})\n",
        "    except Exception as e:\n",
        "        return jsonify({\"error\": str(e)})\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    app.run(port=5001)\n"
      ]
    }
  ]
}