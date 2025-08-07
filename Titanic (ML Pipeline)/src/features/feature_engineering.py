from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging
log = logging.getLogger(__name__)

class CustomRenamer(BaseEstimator, TransformerMixin):
    """Custom transformer to rename columns in a DataFrame."""

    def __init__(self, renaming_dict: dict):
        self.renaming_dict = renaming_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.rename(columns=self.renaming_dict, inplace=True)
        return X

class CustomMapping(BaseEstimator, TransformerMixin):
    """Custom transformer to map variables to arbitrary values."""

    def __init__(self, mapping: dict, variable: str):
        self.mapping = mapping
        self.variable = variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.variable in X.columns:
            X[self.variable] = X[self.variable].map(self.mapping)
            return X

        else:
            raise ValueError(f"Variable '{self.variable}' not found in the DataFrame.")

class CapFareOutliers(BaseEstimator, TransformerMixin):
    """
    Custom transformer to cap outliers in the 'Fare' feature.
    Outliers are defined as values greater than 3 standard deviations from the mean.
    """
    def __init__(self):
        self.fare_bounds = pd.DataFrame()

    def fit(self, X, y=None):
        fare_bounds = X.groupby('Pclass')['Fare'].agg(
                                                    Q1=lambda x: x.quantile(0.25),
                                                    Q3=lambda x: x.quantile(0.75)
                                                    ).round(2)

        fare_bounds['IQR'] = fare_bounds['Q3'] - fare_bounds['Q1']
        fare_bounds['Upper_Bound'] = fare_bounds['Q3'] + 1.5 * fare_bounds['IQR']
        fare_bounds['Lower_Bound'] = (fare_bounds['Q1'] - 1.5 * fare_bounds['IQR']).clip(lower=0)
        fare_bounds.drop(columns=['Q1', 'Q3', 'IQR'], inplace=True)

        self.fare_bounds = fare_bounds
        return self

    def transform(self, X):
        for i in X['Pclass'].unique():
            X.loc[(X['Pclass'] == i) & (X['Fare'] > self.fare_bounds.loc[i, 'Upper_Bound']), 'Fare'] = self.fare_bounds.loc[i, 'Upper_Bound']
            X.loc[(X['Pclass'] == i) & (X['Fare'] < self.fare_bounds.loc[i, 'Lower_Bound']), 'Fare'] = self.fare_bounds.loc[i, 'Lower_Bound']
        return X

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str, group_var: str):
        self.variable = variable
        self.group_var = group_var

    def fit(self, X, y=None):
        self.median_dict_ = X.groupby(self.group_var)[self.variable].median().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for group, median in self.median_dict_.items():
            mask = (X[self.group_var] == group) & (X[self.variable].isnull())
            X.loc[mask, self.variable] = median
        return X

class TitleExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract titles from the 'Name' feature."""

    def __init__(self):
        self.title_mapping = {
            'Mr': 'Mr',
            'Mrs': 'Mrs',
            'Miss': 'Miss',
            'Ms': 'Miss',
            'Mlle': 'Miss',
            'Master': 'Master',
            'Dr': 'Rare',
            'Rev': 'Rare',
            'Col': 'Rare',
            'Major': 'Rare',
            'Capt': 'Rare',
            'Sir': 'Mr',
            'Lady': 'Miss',
            'Don': 'Rare',
            'the Countess': 'Rare',
            'Jonkheer': 'Rare',
            'Mme': 'Mrs',
            'Dona': 'Mrs'
        }

    def fit(self, X, y=None):
        X = X.copy()
        X['Title'] = X['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
        X['Title'] = X['Title'].map(self.title_mapping)

        self.unique_titles = X['Title'].unique()
        dummies = pd.get_dummies(X['Title'], prefix='Title', drop_first=True).astype(int)
        self.dummies_list = dummies.columns
        return self

    def transform(self, X):
        X = X.copy()
        X['Title'] = X['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
        X['Title'] = X['Title'].map(self.title_mapping)

        self.dummies = pd.get_dummies(X['Title'], prefix='Title', drop_first=True).astype(int)
        X = X.join(self.dummies)

        for col in self.dummies_list:
            if col not in X.columns:
                X[col] = 0

        X.drop(columns=['Name', 'Title'], inplace=True)
        return X

class IsFamilyOnBoard(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["isfamilyonboard"] = ((X["SibSp"] > 0) | (X["Parch"] > 0)).astype(int)
        return X

class AgeGroupEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['AgeGroup'] = pd.cut(X['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])
        return X

class TicketCounter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TicketGroupSize'] = X.groupby('Ticket')['Ticket'].transform('count')
        return X