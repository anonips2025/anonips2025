import pandas as pd

########################################################################################################################
# bank
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/ml/datasets/bank+marketing
# and https://www.openml.org/search?type=data&sort=runs&id=1461&status=active
bank_feature_names = [
    ('age', 'age'),
    ('job', 'type of job'),
    ('marital', 'marital status'),
    ('education', 'education'),
    ('default', 'has credit in default?'),
    ('balance', 'average yearly balance, in euros'),
    ('housing', 'has housing loan?'),
    ('loan', 'has personal loan?'),
    ('contact', 'contact communication type'),
    ('day', 'last contact day of the month'),
    ('month', 'last contact month of year'),
    ('duration', 'last contact duration, in seconds'),
    ('campaign', 'number of contacts performed during this campaign and for this client'),
    ('pdays', 'number of days that passed by after the client was last contacted from a previous campaign'),
    ('previous', 'number of contacts performed before this campaign and for this client'),
    ('poutcome', 'outcome of the previous marketing campaign'),
]
template_config_bank = {
    'pre': {
        'age': lambda x: f"{int(x)}",
        'balance': lambda x: f"{int(x)}",
        'day': lambda x: f"{int(x)}",
        'duration': lambda x: f"{int(x)}",
        'campaign': lambda x: f"{int(x)}",
        'pdays': lambda x: f"{int(x)}" if x != -1 else 'client was not previously contacted',
        'previous': lambda x: f"{int(x)}",
    }
}
template_bank = ' '.join(['The ' + v + ' is {' + k + '}.' for k, v in bank_feature_names])


########################################################################################################################
# blood
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
blood_feature_names = [
    ('recency', 'Recency - months since last donation'),
    ('frequency', 'Frequency - total number of donation'),
    ('monetary', 'Monetary - total blood donated in c.c.'),
    ('time', 'Time - months since first donation'),
]
template_config_blood = {
    'pre': {
        'recency': lambda x: f"{int(x)}",
        'frequency': lambda x: f"{int(x)}",
        'monetary': lambda x: f"{int(x)}",
        'time': lambda x: f"{int(x)}",
    }
}
template_blood = ' '.join(['The ' + v + ' is {' + k + '}.' for k, v in blood_feature_names])


########################################################################################################################
# calhousing
########################################################################################################################
# Use description from: Pace and Barry (1997), "Sparse Spatial Autoregressions", Statistics and Probability Letters.
calhousing_feature_names = [
    ('median_income', 'median income'),
    ('housing_median_age', 'median age'),
    ('total_rooms', 'total rooms'),
    ('total_bedrooms', 'total bedrooms'),
    ('population', 'population'),
    ('households', 'households'),
    ('latitude', 'latitude'),
    ('longitude', 'longitude'),
]
template_config_calhousing = {
    'pre': {
        'median_income': lambda x: f"{x:.4f}",
        'housing_median_age': lambda x: f"{int(x)}",
        'total_rooms': lambda x: f"{int(x)}",
        'total_bedrooms': lambda x: f"{int(x)}",
        'population': lambda x: f"{int(x)}",
        'households': lambda x: f"{int(x)}",
        'latitude': lambda x: f"{x:.2f}",
        'longitude': lambda x: f"{x:.2f}",
    }
}
template_calhousing = ' '.join(['The ' + v + ' is {' + k + '}.' for k, v in calhousing_feature_names])


########################################################################################################################
# creditg
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
creditg_feature_names = [
    ('checking_status', 'Status of existing checking account'),
    ('duration', 'Duration in month'),
    ('credit_history', 'Credit history '),
    ('purpose', 'Purpose'),
    ('credit_amount', 'Credit amount'),
    ('savings_status', 'Savings account/bonds'),
    ('employment', 'Present employment since'),
    ('installment_commitment', 'Installment rate in percentage of disposable income'),
    ('personal_status', 'Personal status and sex'),
    ('other_parties', 'Other debtors / guarantors'),
    ('residence_since', 'Present residence since'),
    ('property_magnitude', 'Property'),
    ('age', 'Age in years'),
    ('other_payment_plans', 'Other installment plans'),
    ('housing', 'Housing'),
    ('existing_credits', 'Number of existing credits at this bank'),
    ('job', 'Job'),
    ('num_dependents', 'Number of people being liable to provide maintenance for'),
    ('own_telephone', 'Telephone'),
    ('foreign_worker', 'foreign worker')
]
checking_status_dict = {'<0': '< 0 DM', '0<=X<200': '0 <= ... < 200 DM', '>=200': '>= 200 DM', 'no checking': 'no checking account'}
credit_history_dict = {'no credits/all paid': 'no credits taken/ all credits paid back duly', 'all paid': 'all credits at this bank paid back duly', 'existing paid': 'existing credits paid back duly till now', 'delayed previously': 'delay in paying off in the past', 'critical/other existing credit': 'critical account/ other credits existing (not at this bank)'}
purpose_dict = {'new car': 'car (new)', 'used car': 'car (used)', 'furniture/equipment': 'furniture/equipment', 'radio/tv': 'radio/television', 'domestic appliance': 'domestic appliances', 'repairs': 'repairs', 'education': 'education', 'retraining': 'retraining', 'business': 'business', 'other': 'others'}
savings_status_dict = {'<100': '... < 100 DM', '100<=X<500': '100 <= ... < 500 DM', '500<=X<1000': '500 <= ... < 1000 DM', '>=1000': '... >= 1000 DM', 'no known savings': 'unknown/ no savings account'}
employment_dict = {'unemployed': 'unemployed', '<1': '... < 1 year', '1<=X<4': '1 <= ... < 4 years', '4<=X<7': '4 <= ... < 7 years', '>=7': '... >= 7 years',}
personal_status_dict = {'female div/dep/mar': 'female : divorced/separated/married', 'male div/sep': 'male : divorced/separated', 'male mar/wid': 'male : married/widowed', 'male single': 'male : single'}
other_parties_dict = {'none': 'none', 'co applicant': 'co-applicant', 'guarantor': 'guarantor'}
property_magnitude_dict = {'car': 'car or other, not in attribute 6', 'life insurance': 'building society savings agreement/ life insurance', 'no known property': 'unknown / no property', 'real estate': 'real estate'}
job_dict = {'high qualif/self emp/mgmt': 'management/ self-employed/ highly qualified employee/ officer', 'skilled': 'skilled employee / official', 'unemp/unskilled non res': 'unemployed/ unskilled - non-resident', 'unskilled resident': 'unskilled - resident'}
own_telephone_dict = {'none': 'none', 'yes': 'yes, registered under the customers name'}
template_config_creditg = {
    'pre': {
        'checking_status': lambda x: checking_status_dict[x],
        'duration': lambda x: f"{int(x)}",
        'credit_history': lambda x: credit_history_dict[x],
        'purpose': lambda x: purpose_dict[x],
        'credit_amount': lambda x: f"{int(x)}",
        'savings_status': lambda x: savings_status_dict[x],
        'employment_status': lambda x: employment_dict[x],
        'installment_commitment': lambda x: f"{int(x)}",
        'personal_status': lambda x: personal_status_dict[x],
        'other_parties': lambda x: other_parties_dict[x],
        'residence_since': lambda x: f"{int(x)}",
        'property_magnitude': lambda x: property_magnitude_dict[x],
        'age': lambda x: f"{int(x)}",
        'existing_credits': lambda x: f"{int(x)}",
        'job': lambda x: job_dict[x],
        'own_telephone': lambda x: own_telephone_dict[x]
    }
}
template_creditg = ' '.join(['The ' + v + ' is {' + k + '}.' for k, v in creditg_feature_names])


########################################################################################################################
# diabetes
########################################################################################################################
# Used descriptions from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf
template_config_diabetes = {
    'pre': {
        'Age': lambda x: f"{int(x)}",
        'Pregnancies': lambda x: f"{int(x)}",
        'BloodPressure': lambda x: f"{int(x)}",
        'SkinThickness': lambda x: f"{int(x)}",
        'Glucose': lambda x: f"{int(x)}",
        'Insulin': lambda x: f"{int(x)}",
        'BMI': lambda x: f"{x:.1f}",
        'DiabetesPedigreeFunction': lambda x: f"{x:.3f}"
    }
}
template_diabetes = 'The Age is {Age}. ' \
                    'The Number of times pregnant is {Pregnancies}. ' \
                    'The Diastolic blood pressure is {BloodPressure}. ' \
                    'The Triceps skin fold thickness is {SkinThickness}. ' \
                    'The Plasma glucose concentration at 2 hours in an oral glucose tolerance test (GTT) is ' \
                    '{Glucose}. ' \
                    'The 2-hour serum insulin is {Insulin}. ' \
                    'The Body mass index is {BMI}. ' \
                    'The Diabetes pedigree function is {DiabetesPedigreeFunction}.'



########################################################################################################################
# heart
########################################################################################################################
# Used descriptions from: https://www.kaggle.com/code/azizozmen/heart-failure-predict-8-classification-techniques
chest_paint_types_list = {'TA': 'typical angina', 'ATA': 'atypical angina', 'NAP': 'non-anginal pain', 'ASY': 'asymptomatic'}
rest_ecg_results = {
    'Normal': 'normal',
    'ST': 'ST-T wave abnormality',
    'LVH': 'probable or definite left ventricular hypertrophy'
}
st_slopes = {'Up': 'upsloping', 'Flat': 'flat', 'Down': 'downsloping'}
template_config_heart = {
    'pre': {
        'Sex': lambda x: 'male' if x == 'M' else 'female',
        'ChestPainType': lambda x: chest_paint_types_list[x],
        'FastingBS': lambda x: 'yes' if x == 1 else 'no',
        'ExerciseAngina': lambda x: 'yes' if x == 'Y' else 'no',
        'ST_Slope': lambda x: st_slopes[x],
        'RestingECG': lambda x: rest_ecg_results[x]
    }
}
template_heart = 'The Age of the patient is {Age}. ' \
                 'The Sex of the patient is {Sex}. ' \
                 'The Chest pain type is {ChestPainType}. ' \
                 'The Resting blood pressure is {RestingBP}. ' \
                 'The Serum cholesterol is {Cholesterol}. ' \
                 'The Fasting blood sugar > 120 mg/dl is {FastingBS}. ' \
                 'The Resting electrocardiogram results is {RestingECG}. ' \
                 'The Maximum heart rate achieved is {MaxHR}. ' \
                 'The Exercise-induced angina is {ExerciseAngina}. ' \
                 'The ST depression induced by exercise relative to rest is {Oldpeak}. ' \
                 'The Slope of the peak exercise ST segment is {ST_Slope}.'


########################################################################################################################
# income
########################################################################################################################
gender_categories = ['female', 'male']
race_categories = ['race not recorded', 'hispanic or latino', 'asian', 'black or african american',
                   'american indian or alaska native', 'white', 'native hawaiian or other pacific islander']
female_name = 'Mary Smith'
male_name = 'James Smith'
female_pronoun = 'she'
male_pronoun = 'he'

# Compiled from https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/1990-census-sic-codes.pdf and
# https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2002-census-occupation-codes.xls
occupation_dict = {
    'Tech-support': 'in the technology and support sector',
    'Craft-repair': 'in the craft and repair sector',
    'Other-service': 'in the service sector',
    'Sales': 'in the sales sector',
    'Exec-managerial': 'in execution and management',
    'Prof-specialty': 'in a professional specialty',
    'Handlers-cleaners': 'in the cleaning and maintenance sector',
    'Machine-op-inspct': 'as a machine operator and inspector',
    'Adm-clerical': 'in office and administrative support',
    'Farming-fishing': 'in the agriculture, forestry, and fisheries sector',
    'Transport-moving': 'in the transportation, communication, and other public utilities sector',
    'Priv-house-serv': 'in their private household',
    'Protective-serv': 'in the protective services sector',
    'Armed-Forces': 'in the armed forces'
}
occupation_dict_list = {
    'Tech-support': 'technology and support sector',
    'Craft-repair': 'craft and repair sector',
    'Other-service': 'service sector',
    'Sales': 'sales sector',
    'Exec-managerial': 'execution and management',
    'Prof-specialty': 'professional specialty',
    'Handlers-cleaners': 'cleaning and maintenance sector',
    'Machine-op-inspct': 'machine operator and inspector',
    'Adm-clerical': 'office and administrative support',
    'Farming-fishing': 'agriculture, forestry, and fisheries sector',
    'Transport-moving': 'transportation, communication, and other public utilities sector',
    'Priv-house-serv': 'private household',
    'Protective-serv': 'protective services sector',
    'Armed-Forces': 'armed forces'
}
workclass_dict = {
    'Private': 'as a private sector employee',
    'Local-gov': 'for the local government',
    'State-gov': 'for the state government',
    'Federal-gov': 'for the federal government',
    'Self-emp-not-inc': 'as an owner of a non-incorporated business, professional practice, or farm',
    'Self-emp-inc': 'as a an owner of a incorporated business, professional practice, or farm',
    'Without-pay': 'without pay in a for-profit family business or farm',
    'Never-worked': 'never worked',
}
workclass_dict_list = {
    'Private': 'private sector employee',
    'Local-gov': 'local government',
    'State-gov': 'state government',
    'Federal-gov': 'federal government',
    'Self-emp-not-inc': 'owner of a non-incorporated business, professional practice, or farm',
    'Self-emp-inc': 'owner of a incorporated business, professional practice, or farm',
    'Without-pay': 'without pay in a for-profit family business or farm',
    'Never-worked': 'never worked',
}
# From: https://www.census.gov/content/dam/Census/library/publications/2007/dec/10_education.pdf
education_dict = {
    'Doctorate': 'has a doctoral degree',
    'Prof-school': 'has a professional degree',
    'Masters': 'has a master\'s degree',
    'Bachelors': 'has a bachelor\'s degree',
    'Assoc-acdm': 'has an associate\'s degree',
    'Assoc-voc': 'went to college for one or more years without a degree',
    'Some-college': 'went to college for less than one year',
    'HS-grad': 'is a high school graduate',
    '12th': 'finished 12th class without diploma',
    '11th': 'finished 11th class',
    '10th': 'finished 10th class',
    '9th': 'finished 9th class',
    '7th-8th': 'finished 8th class',
    '5th-6th': 'finished 6th class',
    '1st-4th': 'finished 4th class',
    'Preschool': 'completed no schooling'
}
education_dict_list = {
    'Doctorate': 'doctoral degree',
    'Prof-school': 'professional degree',
    'Masters': 'master\'s degree',
    'Bachelors': 'bachelor\'s degree',
    'Assoc-acdm': 'associate\'s degree',
    'Assoc-voc': 'college for one or more years without a degree',
    'Some-college': 'college for less than one year',
    'HS-grad': 'high school graduate',
    '12th': 'finished 12th class without diploma',
    '11th': 'finished 11th class',
    '10th': 'finished 10th class',
    '9th': 'finished 9th class',
    '7th-8th': 'finished 8th class',
    '5th-6th': 'finished 6th class',
    '1st-4th': 'finished 4th class',
    'Preschool': 'no schooling'
}
# From https://www.census.gov/programs-surveys/cps/technical-documentation/subject-definitions.html#householder
relationship_dict = {
    'Wife': 'and is the wife of the head of the household',
    'Own-child': 'and is a child of the head of the household',
    'Husband': 'and is the husband of the head of the household',
    'Not-in-family': 'and is not in a family',
    'Other-relative': 'and is an other relative of the head of the household',
    'Unmarried': 'and is not married to the head of the household'
}
relationship_dict_list = {
    'Wife': 'wife',
    'Own-child': 'own child',
    'Husband': 'husband',
    'Not-in-family': 'not in a family',
    'Other-relative': 'other relative',
    'Unmarried': 'unmarried'
}
template_config_income = {
    'pre': {
        'race': lambda r: None if r.lower() == 'other' else r,
        'marital_status': lambda ms: 'married' if ms.lower().startswith('married-') else
        ('never married' if ms.lower() == 'never-married' else ms.lower()),
        'native_country': lambda nc: 'United States' if nc in ['United-States', 'Outlying-US(Guam-USVI-etc)']
        else (None if pd.isna(nc) else ('South Korea' if nc.lower() == 'South' else nc)),
        'occupation': lambda o: occupation_dict_list.get(o, ''),
        'workclass': lambda w: workclass_dict_list.get(w, ''),
        'education': lambda e: education_dict_list.get(e)
    },
}
template_income = 'The Age is {age}. ' \
                  'The Race is {race}. ' \
                  'The Sex is {sex}. ' \
                  'The Marital status is {marital_status}. ' \
                  'The Relation to head of the household is {relationship}. ' \
                  'The Native country is {native_country}. ' \
                  'The Occupation is {occupation}. ' \
                  'The Work class is {workclass}. ' \
                  'The Capital gain last year is {capital_gain}. ' \
                  'The Capital loss last year is {capital_loss}. ' \
                  'The Education is {education}. ' \
                  'The Work hours per week is {hours_per_week}.'


########################################################################################################################
# jungle
########################################################################################################################
# Use description from: https://arxiv.org/abs/1604.07312
jungle_feature_names = [
    ('white_piece0_strength', 'white piece strength'),
    ('white_piece0_file', 'white piece file'),
    ('white_piece0_rank', 'white piece rank'),
    ('black_piece0_strength', 'black piece strength'),
    ('black_piece0_file', 'black piece file'),
    ('black_piece0_rank', 'black piece rank')
]
template_config_jungle = {
    'pre': {
        'white_piece0_strength': lambda x: f"{int(x)}",
        'white_piece0_file': lambda x: f"{int(x)}",
        'white_piece0_rank': lambda x: f"{int(x)}",
        'black_piece0_strength': lambda x: f"{int(x)}",
        'black_piece0_file': lambda x: f"{int(x)}",
        'black_piece0_rank': lambda x: f"{int(x)}",
    }
}
template_jungle = ' '.join(['The ' + v + ' is {' + k + '}.' for k, v in jungle_feature_names])


########################################################################################################################
# Albert dataset
########################################################################################################################
# Use description from: https://openml.org/d/45035
# Feature names remain exactly as in the dataset
albert_feature_names = [
    ('V1', 'V1'),
    ('V2', 'V2'),
    ('V3', 'V3'),
    ('V4', 'V4'),
    ('V5', 'V5'),
    ('V6', 'V6'),
    ('V7', 'V7'),
    ('V8', 'V8'),
    ('V9', 'V9'),
    ('V10', 'V10'),
    ('V11', 'V11'),
    ('V13', 'V13'),
    ('V19', 'V19'),
    ('V22', 'V22'),
    ('V30', 'V30'),
    ('V33', 'V33'),
    ('V35', 'V35'),
    ('V36', 'V36'),
    ('V40', 'V40'),
    ('V41', 'V41'),
    ('V42', 'V42'),
    ('V43', 'V43'),
    ('V45', 'V45'),
    ('V47', 'V47'),
    ('V50', 'V50'),
    ('V51', 'V51'),
    ('V52', 'V52'),
    ('V59', 'V59'),
    ('V63', 'V63'),
    ('V72', 'V72'),
    ('V75', 'V75'),
]

template_config_albert = {
    'pre': {
        # No preprocessing
    }
}

template_albert = ' '.join([f'{k}: {{{k}}}' for k, _ in albert_feature_names])

########################################################################################################################
# COMPAS dataset
########################################################################################################################
# Use description from: https://github.com/propublica/compas-analysis
compas_feature_names = [
    ('sex', 'Sex'),
    ('age', 'Age'),
    ('juv_misd_count', 'Number of juvenile misdemeanor charges'),
    ('priors_count', 'Number of prior charges'),
    ('age_cat_25-45', 'Age category 25-45'),
    ('age_cat_Greaterthan45', 'Age category greater than 45'),
    ('age_cat_Lessthan25', 'Age category less than 25'),
    ('race_African-American', 'Race African-American'),
    ('race_Caucasian', 'Race Caucasian'),
    ('c_charge_degree_F', 'Charge degree felony'),
    ('c_charge_degree_M', 'Charge degree misdemeanor')
]

# Template configuration
template_config_compas = {
    'pre': {
        'sex': lambda x: 'Male' if x == 1 else 'Female',
        'age': lambda x: f"{int(x)} years old",
        'juv_misd_count': lambda x: f"{int(x)} juvenile misdemeanor charge(s)",
        'priors_count': lambda x: f"{int(x)} prior charge(s)",
        'age_cat_25-45': lambda x: 'Age 25-45' if x == 1 else '',
        'age_cat_Greaterthan45': lambda x: 'Age >45' if x == 1 else '',
        'age_cat_Lessthan25': lambda x: 'Age <25' if x == 1 else '',
        'race_African-American': lambda x: 'African-American' if x == 1 else '',
        'race_Caucasian': lambda x: 'Caucasian' if x == 1 else '',
        'c_charge_degree_F': lambda x: 'Felony charge' if x == 1 else '',
        'c_charge_degree_M': lambda x: 'Misdemeanor charge' if x == 1 else ''
    },
    'post': {
        'age_category': lambda row: ', '.join(filter(None, [
            '25-45' if row['age_cat_25-45'] == 1 else None,
            '>45' if row['age_cat_Greaterthan45'] == 1 else None,
            '<25' if row['age_cat_Lessthan25'] == 1 else None
        ])),
        'race': lambda row: ', '.join(filter(None, [
            'African-American' if row['race_African-American'] == 1 else None,
            'Caucasian' if row['race_Caucasian'] == 1 else None
        ])),
        'charge_degree': lambda row: ', '.join(filter(None, [
            'Felony' if row['c_charge_degree_F'] == 1 else None,
            'Misdemeanor' if row['c_charge_degree_M'] == 1 else None
        ]))
    }
}

# Template using natural language descriptions with combined one-hot features
template_compas = (
    "The individual is {sex}. "
    "The subject is {age}. "
    "The subject has {juv_misd_count}. "
    "The subject has {priors_count}. "
    "Age category: {age_category}. "
    "Race: {race}. "
    "Charge degree: {charge_degree}."
)


########################################################################################################################
# Covertype Dataset
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/dataset/31/covertype
covertype_feature_names = [
    ('Elevation', 'elevation in meters'),
    ('Aspect', 'aspect in degrees azimuth'),
    ('Slope', 'slope in degrees'),
    ('Horizontal_Distance_To_Hydrology', 'horizontal distance to nearest surface water'),
    ('Vertical_Distance_To_Hydrology', 'vertical distance to nearest surface water'),
    ('Horizontal_Distance_To_Roadways', 'horizontal distance to nearest roadway'),
    ('Hillshade_9am', 'hillshade index at 9am'),
    ('Hillshade_Noon', 'hillshade index at noon'),
    ('Hillshade_3pm', 'hillshade index at 3pm'),
    ('Horizontal_Distance_To_Fire_Points', 'horizontal distance to nearest wildfire ignition point'),
    ('Wilderness_Area', 'wilderness area designation'),
    ('Soil_Type', 'soil type designation')
]

# Wilderness area names (mapping from one-hot encoded columns)
wilderness_area_names = {
    1: 'Rawah Wilderness Area',
    2: 'Neota Wilderness Area',
    3: 'Comanche Peak Wilderness Area',
    4: 'Cache la Poudre Wilderness Area'
}

# Template configuration
template_config_covertype = {
    'pre': {
        'Elevation': lambda x: f"{int(x)} meters",
        'Aspect': lambda x: f"{int(x)} degrees",
        'Slope': lambda x: f"{int(x)} degrees",
        'Horizontal_Distance_To_Hydrology': lambda x: f"{int(x)} meters",
        'Vertical_Distance_To_Hydrology': lambda x: f"{int(x)} meters",
        'Horizontal_Distance_To_Roadways': lambda x: f"{int(x)} meters",
        'Hillshade_9am': lambda x: f"{int(x)} index",
        'Hillshade_Noon': lambda x: f"{int(x)} index",
        'Hillshade_3pm': lambda x: f"{int(x)} index",
        'Horizontal_Distance_To_Fire_Points': lambda x: f"{int(x)} meters"
    },
    'post': {
        'Wilderness_Area': lambda row: next(
            (wilderness_area_names[i] for i in range(1, 5) 
             if row[f'Wilderness_Area{i}'] == 1),
            'Unknown Wilderness Area'
        ),
        'Soil_Type': lambda row: next(
            (f"Soil Type {i}" for i in range(1, 41) 
             if row[f'Soil_Type{i}'] == 1),
            'Unknown Soil Type'
        )
    }
}

# Template using natural language descriptions
template_covertype = (
    "The location has an elevation of {Elevation}. "
    "The aspect is {Aspect}. The slope is {Slope}. "
    "The horizontal distance to nearest water is {Horizontal_Distance_To_Hydrology}."
    "The vertical distance to nearest water is {Vertical_Distance_To_Hydrology}."
    "The distance to nearest road is {Horizontal_Distance_To_Roadways}."
    "Hillshade indices: {Hillshade_9am} at 9am, {Hillshade_Noon} at noon, "
    "and {Hillshade_3pm} at 3pm. "
    "The distance to fire points is {Horizontal_Distance_To_Fire_Points}."
    "The wilderness area designation is {Wilderness_Area}. The soil type is {Soil_Type}."
)


########################################################################################################################
# Electricity
########################################################################################################################
# Use description from: https://www.openml.org/d/151
electricity_feature_names = [
    ('date', 'Date of the record'),
    ('day', 'Day of week'),
    ('period', 'Time period in the day'),
    ('nswprice', 'New South Wales electricity price'),
    ('nswdemand', 'New South Wales electricity demand'),
    ('vicprice', 'Victoria electricity price'),
    ('vicdemand', 'Victoria electricity demand'),
    ('transfer', 'Scheduled electricity transfer between states')
]

day_dict = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}

# Template configuration
template_config_electricity = {
    'pre': {
        'date': lambda x: f"{float(x):.1%} normalized date",
        'day': lambda x: day_dict[int(x)],
        'period': lambda x: f"{float(x)*24:.1f} hours",
        'nswprice': lambda x: f"${float(x)*100:.2f}/MWh",
        'nswdemand': lambda x: f"{float(x)*1000:.0f}MW",
        'vicprice': lambda x: f"${float(x)*100:.2f}/MWh",
        'vicdemand': lambda x: f"{float(x)*1000:.0f}MW",
        'transfer': lambda x: f"{float(x)*100:.1f}% capacity"
    }
}

# Template using natural language descriptions
template_electricity = ' '.join([f'The {description} is {{{name}}}.' 
    for (name, description) in electricity_feature_names
])


########################################################################################################################
# Eye Movements Dataset
########################################################################################################################
# Use description from: Jarkko Salojarvi, Kai Puolamaki, Jaana Simola, Lauri Kovanen, Ilpo Kojo, Samuel Kaski. Inferring Relevance from Eye Movements: Feature Extraction. Helsinki University of Technology, Publications in Computer and Information Science, Report A82. 3 March 2005
eye_movements_feature_names = [
    ('lineNo', 'line number'),
    ('assgNo', 'assignment number'),
    ('P1stFixation', 'whether first fixation on word'),
    ('P2stFixation', 'whether second fixation on word'),
    ('prevFixDur', 'previous fixation duration in ms'),
    ('firstfixDur', 'first fixation duration in ms'),
    ('firstPassFixDur', 'first pass fixation duration in ms'),
    ('nextFixDur', 'next fixation duration in ms'),
    ('firstSaccLen', 'first saccade length in pixels'),
    ('lastSaccLen', 'last saccade length in pixels'),
    ('prevFixPos', 'previous fixation position'),
    ('landingPos', 'landing position'),
    ('leavingPos', 'leaving position'),
    ('totalFixDur', 'total fixation duration in ms'),
    ('meanFixDur', 'mean fixation duration in ms'),
    ('regressLen', 'regression length in pixels'),
    ('nextWordRegress', 'whether regression to next word'),
    ('regressDur', 'regression duration in ms'),
    ('pupilDiamMax', 'maximum pupil diameter'),
    ('pupilDiamLag', 'pupil diameter lag'),
    ('timePrtctg', 'time percentage'),
    ('titleNo', 'title number'),
    ('wordNo', 'word number')
]

# Template configuration
template_config_eye_movements = {
    'pre': {
        'lineNo': lambda x: f"Line {int(x)}",
        'assgNo': lambda x: f"Assignment {int(x)}",
        'P1stFixation': lambda x: 'yes' if x == 1 else 'no',
        'P2stFixation': lambda x: 'yes' if x == 1 else 'no',
        'prevFixDur': lambda x: f"{int(x)} ms",
        'firstfixDur': lambda x: f"{int(x)} ms",
        'firstPassFixDur': lambda x: f"{int(x)} ms",
        'nextFixDur': lambda x: f"{int(x)} ms",
        'firstSaccLen': lambda x: f"{float(x):.1f} pixels",
        'lastSaccLen': lambda x: f"{float(x):.1f} pixels",
        'prevFixPos': lambda x: f"{float(x):.1f}",
        'landingPos': lambda x: f"{float(x):.1f}",
        'leavingPos': lambda x: f"{float(x):.1f}",
        'totalFixDur': lambda x: f"{int(x)} ms",
        'meanFixDur': lambda x: f"{int(x)} ms",
        'regressLen': lambda x: f"{float(x):.1f} pixels",
        'nextWordRegress': lambda x: 'yes' if x == 1 else 'no',
        'regressDur': lambda x: f"{int(x)} ms" if x > 0 else "no regression",
        'pupilDiamMax': lambda x: f"{float(x):.3f}",
        'pupilDiamLag': lambda x: f"{float(x):.3f}",
        'timePrtctg': lambda x: f"{float(x):.3f}",
        'titleNo': lambda x: f"Title {int(x)}",
        'wordNo': lambda x: f"Word {int(x)}"
    },
    'post': {
        'fixation_pattern': lambda row: (
            f"{'First' if row['P1stFixation'] == 1 else 'Not first'} fixation, "
            f"{'second' if row['P2stFixation'] == 1 else 'not second'} fixation"
        ),
        'movement_summary': lambda row: (
            f"First saccade: {float(row['firstSaccLen']):.1f}px, "
            f"Last saccade: {float(row['lastSaccLen']):.1f}px"
        ),
        'regression_info': lambda row: (
            f"Regression: {'yes' if row['regressLen'] > 0 else 'no'}, "
            f"Length: {float(row['regressLen']):.1f}px, "
            f"Duration: {int(row['regressDur'])}ms" if row['regressDur'] > 0 else "no regression"
        )
    }
}

# Template using natural language descriptions
template_eye_movements = (
    "The word number is {wordNo}. The title number is {titleNo}. The line number is {lineNo}."
    "The fixation pattern is {fixation_pattern}."
    "The first fixation duration is {firstfixDur}, first pass {firstPassFixDur}, "
    "previous {prevFixDur}, next {nextFixDur}, total {totalFixDur}, "
    "mean {meanFixDur}."
    "Positions: previous {prevFixPos}, landing {landingPos}, "
    "leaving {leavingPos}. "
    "Movement: {movement_summary}. "
    "{regression_info}. "
    "Pupil: max {pupilDiamMax}, lag {pupilDiamLag}. "
    "The time percentage is {timePrtctg}."
)


########################################################################################################################
# Road Safety Dataset
########################################################################################################################
road_safety_feature_names = [
    ('Vehicle_Type', 'Vehicle type code'),
    ('Vehicle_Manoeuvre', 'Vehicle manoeuvre code'),
    ('Vehicle_Location-Restricted_Lane', 'Vehicle location in restricted lane code'),
    ('Hit_Object_in_Carriageway', 'Object hit in carriageway code'),
    ('Hit_Object_off_Carriageway', 'Object hit off carriageway code'),
    ('Was_Vehicle_Left_Hand_Drive?', 'Left-hand drive vehicle flag'),
    ('Age_of_Driver', 'Age of driver'),
    ('Age_Band_of_Driver', 'Age band of driver code'),
    ('Engine_Capacity_(CC)', 'Engine capacity in CC'),
    ('Propulsion_Code', 'Propulsion code'),
    ('Age_of_Vehicle', 'Age of vehicle'),
    ('Police_Force', 'Police force code'),
    ('Number_of_Vehicles', 'Number of vehicles involved'),
    ('Number_of_Casualties', 'Number of casualties'),
    ('Urban_or_Rural_Area', 'Urban or rural area code'),
    ('Sex_of_Casualty', 'Sex of casualty'),
    ('Age_of_Casualty', 'Age of casualty'),
    ('Age_Band_of_Casualty', 'Age band of casualty code'),
    ('Pedestrian_Location', 'Pedestrian location code'),
    ('Pedestrian_Movement', 'Pedestrian movement code'),
    ('Casualty_Type', 'Casualty type code'),
    ('Casualty_IMD_Decile', 'Casualty IMD decile')
]

# Only include dictionaries for features we want to convert
left_hand_drive_dict = {
    '0': 'No',
    '1': 'Yes'
}

sex_dict = {
    '1': 'Male',
    '2': 'Female'
}

template_config_road_safety = {
    'pre': {
        'Was_Vehicle_Left_Hand_Drive?': lambda x: left_hand_drive_dict.get(str(int(x)), str(int(x))),
        'Sex_of_Casualty': lambda x: sex_dict.get(str(int(x)), str(int(x))),
        'Age_of_Driver': lambda x: f"{int(x)}",
        'Age_of_Vehicle': lambda x: f"{int(x)}",
        'Engine_Capacity_(CC)': lambda x: f"{int(x)}",
        'Number_of_Vehicles': lambda x: f"{int(x)}",
        'Number_of_Casualties': lambda x: f"{int(x)}",
        'Age_of_Casualty': lambda x: f"{int(x)}",
        'Casualty_IMD_Decile': lambda x: f"{int(x)}"
    }
}

# For the specified features, we'll just use their numerical values directly
template_road_safety = ' '.join(['The ' + v + ' is {' + k + '}.' for k, v in road_safety_feature_names])

########################################################################################################################
# Credit Card Default Dataset
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
credit_card_default_feature_names = [
    ('x1', 'Credit amount (NT dollar)'),
    ('x2', 'Gender'),
    ('x5', 'Age'),
    ('x6', 'September repayment status'),
    ('x7', 'August repayment status'),
    ('x8', 'July repayment status'),
    ('x9', 'June repayment status'),
    ('x10', 'May repayment status'),
    ('x11', 'April repayment status'),
    ('x12', 'September bill amount'),
    ('x13', 'August bill amount'),
    ('x14', 'July bill amount'),
    ('x15', 'June bill amount'),
    ('x16', 'May bill amount'),
    ('x17', 'April bill amount'),
    ('x18', 'September payment amount'),
    ('x19', 'August payment amount'),
    ('x20', 'July payment amount'),
    ('x21', 'June payment amount'),
    ('x22', 'May payment amount'),
    ('x23', 'April payment amount')
]

# Dictionaries for categorical features
gender_dict = {'1': 'Male', '2': 'Female'}
education_dict = {'1': 'Graduate school', '2': 'University', '3': 'High school', '4': 'Others'}
marital_status_dict = {'1': 'Married', '2': 'Single', '3': 'Others'}
repayment_status_dict = {
    '-1': 'Pay duly',
    '1': '1 month delay',
    '2': '2 months delay',
    '3': '3 months delay',
    '4': '4 months delay',
    '5': '5 months delay',
    '6': '6 months delay',
    '7': '7 months delay',
    '8': '8 months delay',
    '9': '9+ months delay'
}

template_config_credit_card_default = {
    'pre': {
        # Categorical features
        'x2': lambda x: gender_dict.get(str(int(x)), str(int(x))),
        'x6': lambda x: repayment_status_dict.get(str(int(x)), str(int(x))),
        'x7': lambda x: repayment_status_dict.get(str(int(x)), str(int(x))),
        'x8': lambda x: repayment_status_dict.get(str(int(x)), str(int(x))),
        'x9': lambda x: repayment_status_dict.get(str(int(x)), str(int(x))),
        'x10': lambda x: repayment_status_dict.get(str(int(x)), str(int(x))),
        'x11': lambda x: repayment_status_dict.get(str(int(x)), str(int(x))),
        
        # Numerical features
        'x1': lambda x: f"{int(x):,} NT$",
        'x5': lambda x: f"{int(x)} years",
        'x12': lambda x: f"{int(x):,} NT$",
        'x13': lambda x: f"{int(x):,} NT$",
        'x14': lambda x: f"{int(x):,} NT$",
        'x15': lambda x: f"{int(x):,} NT$",
        'x16': lambda x: f"{int(x):,} NT$",
        'x17': lambda x: f"{int(x):,} NT$",
        'x18': lambda x: f"{int(x):,} NT$",
        'x19': lambda x: f"{int(x):,} NT$",
        'x20': lambda x: f"{int(x):,} NT$",
        'x21': lambda x: f"{int(x):,} NT$",
        'x22': lambda x: f"{int(x):,} NT$",
        'x23': lambda x: f"{int(x):,} NT$",
    }
}

template_credit_card_default = ' '.join(
    ['The ' + v + ' is {' + k + '}.' for k, v in credit_card_default_feature_names]
)