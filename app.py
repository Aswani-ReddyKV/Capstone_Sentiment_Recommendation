from flask import Flask, request, render_template
import User_Recommendation
from ModelFactory import ModelFactory
import pandas as pd


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    # get user from the html form
    user = request.form['userName']
    # convert text to lowercase
    user = user.lower()
    results_df = User_Recommendation.get_top5_recommendationsNew(user)

    if isinstance(results_df, pd.DataFrame):
        print("Variable is correctly identified as a DataFrame.")
        print(f"DataFrame Columns: {results_df.columns}")
    else:
        print(f"ERROR: Variable is NOT a DataFrame, it is: {type(results_df)}")
    # Handle this error - maybe return an error message or default data

    # --- Pass the DataFrame to the template ---
    # Make sure 'results_df' IS your DataFrame object here!
    try:
         # Get column names as a list
        column_names = results_df.columns.tolist()

        # Get row data as a list of lists
        # results_df.values is a NumPy array, .tolist() converts it
        row_data = results_df.values.tolist()

        # Pass the correct variables to the template
        return render_template("index.html",
                            column_names=column_names,
                                row_data=row_data,
                                     zip=zip) # zip is often used inside the template loop

    except AttributeError as e:
        # This catch block will now correctly identify if results_df wasn't a DataFrame
        print(f"Error: Could not access DataFrame properties. Check if the variable is actually a DataFrame. Error: {e}")
        # Return an error response or render an error template
        return "An error occurred while preparing data for display.", 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred.", 500


if __name__ == '__main__':
    app.run()
