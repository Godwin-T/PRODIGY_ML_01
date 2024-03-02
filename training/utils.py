TRAIN_DATASET_PATH = "../data/house-prices-advanced-regression-techniques/train.csv"

DROP_COLUMNS = ["Id", 'GarageCond', 'EnclosedPorch', 'Condition1', 'Utilities', 
                'GarageCars', 'BsmtFinSF1', 'BsmtCond', 'ExterCond', 'Condition2', 
                'GarageType', 'Heating', 'BsmtFinType2', 'PavedDrive', 'PoolArea', 
                'BsmtFinType1', 'BsmtFinSF2', 'HeatingQC', 'ScreenPorch', 'CentralAir', 
                '3SsnPorch', 'MiscVal', 'SaleCondition', 'BsmtUnfSF', 'GarageQual', 
                'BldgType', 'LotShape', 'LandContour', 'OverallQual', 'RoofMatl', 'Electrical', 
                'LandSlope', 'BsmtHalfBath', 'LowQualFinSF', 'SaleType', 'HouseStyle', 'Exterior2nd', 
                'Street', 'Neighborhood', 'YearBuilt', 'KitchenAbvGr', 'Functional']

TARGET_COLUMN = "SalePrice"

BEST_PARAMS = {"max_depth":35,
               "min_samples_leaf":3,
               "min_samples_split":13,
               "n_estimators":10}
