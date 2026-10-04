import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Get values from the form
    category = int(request.form['category'])
    sellable_online = int(request.form['sellable_online'])
    other_colors = int(request.form['other_colors'])

    # Get dimensions
    # Remove "cm" if the user writes it
    depth = float(request.form['depth'].replace('cm', '').strip())
    height = float(request.form['height'].replace('cm', '').strip())
    width = float(request.form['width'].replace('cm', '').strip())

    # Features must be in the same order as during training
    features = [
        category,
        sellable_online,
        other_colors,
        depth,
        height,
        width
    ]

    print("Features:", features)

    # Convert to NumPy array
    final_features = np.array(features).reshape(1, 6)

    # Make prediction
    prediction = model.predict(final_features)

    # Round the result
    output = round(prediction[0], 2)

    return render_template(
        'index.html',
        prediction_text='Furniture prediction price is : $ {}'.format(output)
    )


if __name__ == '__main__':
    app.run(debug=True)

