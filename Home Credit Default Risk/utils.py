import numpy as np  # import numpy
import pandas as pd

# Function to reduce dataframe memory footprint, reduce float and int to the minimum dtype
from scipy.stats import ranksums


def reducemem(self):
    for c in self:
        if self[c].dtype == 'int64':
            if self[c].max() < np.iinfo(np.int32).max and self[c].min() > np.iinfo(np.int32).min:
                self[c] = self[c].astype(np.int32)
            if self[c].max() < np.iinfo(np.int16).max and self[c].min() > np.iinfo(np.int16).min:
                self[c] = self[c].astype(np.int16)
            if self[c].max() < np.iinfo(np.int8).max and self[c].min() > np.iinfo(np.int8).min:
                self[c] = self[c].astype(np.int8)

        if self[c].dtype == 'float64':
            if self[c].max() < np.finfo(np.float32).max and self[c].min() > np.finfo(np.float32).min:
                self[c] = self[c].astype(np.float32)
            if self[c].max() < np.finfo(np.float16).max and self[c].min() > np.finfo(np.float16).min:
                self[c] = self[c].astype(np.float16)


def float_to_int(ser):
    try:
        int_ser = ser.astype(int)
        if (ser == int_ser).all():
            return int_ser
        else:
            return ser
    except ValueError:
        return ser


def multi_assign(df, transform_fn, condition):
    return (df.assign(
        **{col: transform_fn(df[col])
           for col in condition(df)})
    )


def all_float_to_int(df):
    transform_fn = float_to_int
    condition = lambda x: list(x.select_dtypes(include=["float"]).columns)

    return multi_assign(df, transform_fn, condition)


def downcast_all(df, target_type, inital_type=None):
    # Gotta specify floats, unsigned, or integer
    # If integer, gotta be 'integer', not 'int'
    # Unsigned should look for Ints
    if inital_type is None:
        inital_type = target_type

    transform_fn = lambda x: pd.to_numeric(x,downcast=target_type)

    condition = lambda x: list(x.select_dtypes(include=[inital_type]).columns)

    return multi_assign(df, transform_fn, condition)


def corr_feature_with_target(feature, target):
    c0 = feature[target == 0].dropna()
    c1 = feature[target == 1].dropna()

    if set(feature.unique()) == set([0, 1]):
        diff = abs(c0.mean(axis=0) - c1.mean(axis=0))
    else:
        diff = abs(c0.median(axis=0) - c1.median(axis=0))

    p = ranksums(c0, c1)[1] if ((len(c0) >= 20) & (len(c1) >= 20)) else 2

    return [diff, p]


def trim(st):
    st=st.strip(" ")
    st=st.replace(',', '')
    st=st.replace(' ', '_')
    st=st.replace(':', '_')
    return st


def clean(df):
    df = df.rename(columns=lambda x: trim(x))
    df=df.replace(-np.Inf, 0)
    df=df.replace(np.Inf, 0)
    df=df.replace(np.nan, 0)
    return df