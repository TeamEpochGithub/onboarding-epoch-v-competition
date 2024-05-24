import pandas as pd
"""This script is an example showing the format of submission for predicting pokemon types."""

if __name__ == "__main__":
    data = {
        'id': [1, 2, 3, 4, 5],
        'Bug': [0.5, 0.1, 0.0, 0.2, 0.9],
        'Dark': [0.1, 0.7, 0.2, 0.0, 0.3],
        'Dragon': [0.0, 0.2, 0.8, 0.0, 0.1],
        'Electric': [0.2, 0.0, 0.0, 0.9, 0.1],
        'Fairy': [0.0, 0.1, 0.0, 0.0, 0.8],
        'Fighting': [0.3, 0.6, 0.1, 0.1, 0.0],
        'Fire': [0.0, 0.1, 0.7, 0.0, 0.2],
        'Flying': [0.4, 0.0, 0.0, 0.0, 0.1],
        'Ghost': [0.1, 0.2, 0.5, 0.0, 0.0],
        'Grass': [0.3, 0.0, 0.0, 0.1, 0.6],
        'Ground': [0.0, 0.1, 0.4, 0.0, 0.1],
        'Ice': [0.2, 0.0, 0.1, 0.6, 0.0],
        'Normal': [0.6, 0.0, 0.0, 0.1, 0.2],
        'Poison': [0.0, 0.3, 0.0, 0.0, 0.7],
        'Psychic': [0.1, 0.4, 0.3, 0.0, 0.1],
        'Rock': [0.0, 0.0, 0.2, 0.7, 0.1],
        'Steel': [0.0, 0.1, 0.6, 0.0, 0.0],
        'Water': [0.1, 0.0, 0.0, 0.2, 0.5]
    }

    # Creating the DataFrame
    df = pd.DataFrame(data)

    # Displaying the DataFrame
    print(df)

    #Save to example_submission.csv
    df.to_csv('example_submission.csv', index=False)
