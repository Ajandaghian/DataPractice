from src.config.core import config
from src.features.feature_engineering import TitleExtractor, CustomMapping, CapFareOutliers, \
    GroupMedianImputer, IsFamilyOnBoard, AgeGroupEncoder, TicketCounter
import sklearn.pipeline as pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer, MeanMedianImputer
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.selection import DropFeatures
from feature_engine.discretisation import ArbitraryDiscretiser

import logging
log = logging.getLogger(__name__)


feature_pipeline = pipeline.Pipeline([
    ('age_imputer',
        MeanMedianImputer(variables=['Age'], imputation_method=config['preprocessing_params']['age_imputer_strategy'])),
    ('embarked_imputer',
        CategoricalImputer(imputation_method=config['preprocessing_params']['embarked_imputer_strategy'], variables=['Embarked'])),
    ('fare_imputer',
        GroupMedianImputer(variable='Fare', group_var='Pclass')),
    ('fare_capping',
        CapFareOutliers()),
    ('sex_encoder',
        CustomMapping(variable='Sex', mapping={'male': 0, 'female': 1})),
    ('embarked_encoder',
        SklearnTransformerWrapper(transformer=SklearnOneHotEncoder(drop='first', sparse_output=False),variables=['Embarked'])),
    ('title_extractor',
        TitleExtractor()),
    ('age_group_feature',
        AgeGroupEncoder()),
    ('isfamilyonboard_feature',
        IsFamilyOnBoard()),
    ('ticket_size_feature',
        TicketCounter()),
    ('drop_features',
        DropFeatures(features_to_drop=['PassengerId','Ticket', 'Cabin', 'SibSp', 'Parch'])),
    ('scaling',
        SklearnTransformerWrapper(transformer=StandardScaler(), variables=config['features']['numeric_features']))
])


model_pipeline = pipeline.Pipeline([
    ('model', GradientBoostingClassifier(
        n_estimators=config['model_params']['n_estimators'],
        learning_rate=config['model_params']['learning_rate'],
        max_depth=config['model_params']['max_depth'],
        max_features=config['model_params']['max_features'],
        subsample=config['model_params']['subsample'],
        random_state=config['model_params']['SEED']
    ))
])

