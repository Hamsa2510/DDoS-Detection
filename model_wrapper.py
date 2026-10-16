import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model
from subtype_classifier import SubtypeClassifier

VOL_FEATS = [
    'flow_duration','Header_Length','Duration',
    'fin_flag_number','syn_flag_number','rst_flag_number',
    'psh_flag_number','ack_flag_number','ece_flag_number',
    'cwr_flag_number','ack_count','syn_count','fin_count',
    'urg_count','rst_count','Rate','Srate','Drate',
    'Tot sum','Tot size',
]
TMP_FEATS = [
    'IAT','Number','Magnitue','Radius','Covariance',
    'Variance','Weight','Min','Max','AVG','Std',
    'HTTP','HTTPS','DNS','Telnet','SMTP','SSH','IRC',
    'TCP','UDP','DHCP','ARP','ICMP','IPv','LLC',
    'Protocol Type',
]
ENG_FEATS = ['rate_variance', 'iat_entropy', 'burst_lull_ratio']
ALL_FEATS  = VOL_FEATS + TMP_FEATS + ENG_FEATS

LMAP = {0: 'Benign', 1: 'High-Rate DDoS', 2: 'Low-Rate DDoS', 3: 'Flash Crowd'}
WINDOW_SIZE = 10
N_FEATS     = 49
N_VOL       = 20


def build_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, N_FEATS), name='input')
    cnn = layers.Conv1D(64, 3, padding='same', activation='relu', name='cnn1')(inputs)
    cnn = layers.Conv1D(64, 3, padding='same', activation='relu', name='cnn2')(cnn)
    cnn = layers.Conv1D(32, 1, padding='same', activation='relu', name='cnn_proj')(cnn)
    cnn = layers.GlobalAveragePooling1D(name='cnn_gap')(cnn)
    tcn = layers.Conv1D(64, 3, padding='causal', dilation_rate=1, activation='relu', name='tcn1')(inputs)
    tcn = layers.Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu', name='tcn2')(tcn)
    tcn = layers.Conv1D(64, 3, padding='causal', dilation_rate=4, activation='relu', name='tcn3')(tcn)
    tcn = layers.GlobalMaxPooling1D(name='tcn_pool')(tcn)
    merged = layers.Concatenate(name='merge')([cnn, tcn])
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                     name='fd1')(merged)
    x = layers.Dropout(0.3, name='drop1')(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                     name='fd2')(x)
    x = layers.Dropout(0.2, name='drop2')(x)
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                     name='fd3')(x)
    outputs = layers.Dense(4, activation='softmax', name='output')(x)
    return Model(inputs, outputs)


class ModelWrapper:

    def __init__(self, weights_path, scaler_path):
        print('Building model architecture...')
        self.model = build_model()
        print('Loading weights...')
        self.model.load_weights(weights_path)
        self.scaler = joblib.load(scaler_path)
        self.subtype_clf = SubtypeClassifier()
        print('Model and scaler loaded.')

    def _add_engineered_features(self, df):
        eps = 1e-9
        df['rate_variance'] = np.where(
            df['Std'] > 0,
            ((df['Rate'] - df['AVG']) ** 2) / (df['Std'] + eps),
            0.0
        ).astype(np.float32)
        iat_norm  = df['IAT'].clip(lower=0)
        iat_total = iat_norm + eps
        df['iat_entropy'] = -(iat_norm / iat_total * np.log(iat_norm / iat_total + eps)).astype(np.float32)
        df['burst_lull_ratio'] = (df['Rate'] / (df['AVG'] + eps)).clip(upper=100.0).astype(np.float32)
        return df

    def predict_window(self, window):
        x     = window.reshape(1, WINDOW_SIZE, N_FEATS)
        probs = self.model.predict(x, verbose=0)[0]
        idx   = int(np.argmax(probs))
        return {
            'label':      LMAP[idx],
            'class_id':   idx,
            'confidence': float(probs[idx]),
            'probs':      {LMAP[i]: float(p) for i, p in enumerate(probs)}
        }

    def predict_window_with_subtype(self, window, flow_dict):
        """
        Same as predict_window but also returns attack sub-type.
        flow_dict: raw feature dict from last row of window
        """
        result          = self.predict_window(window)
        subtype         = self.subtype_clf.classify(flow_dict, result['label'])
        result['subtype'] = subtype
        return result

    def prepare_window_from_df(self, df_10rows):
        df = df_10rows[VOL_FEATS + TMP_FEATS].copy()
        df = self._add_engineered_features(df)
        X  = df[ALL_FEATS].values.astype(np.float32)
        X  = np.clip(self.scaler.transform(X), 0, 1)
        return X

    def explain_window(self, window):
        import warnings
        warnings.filterwarnings('ignore')

        background     = window.reshape(1, WINDOW_SIZE, N_FEATS)
        baseline_probs = self.model.predict(background, verbose=0)[0]
        pred_class     = int(np.argmax(baseline_probs))
        baseline_score = float(baseline_probs[pred_class])

        importances = []
        for i in range(N_FEATS):
            perturbed          = background.copy()
            perturbed[0, :, i] = 0.0
            perturbed_probs    = self.model.predict(perturbed, verbose=0)[0]
            perturbed_score    = float(perturbed_probs[pred_class])
            importance         = baseline_score - perturbed_score
            importances.append({
                'feature':    ALL_FEATS[i],
                'importance': max(0.0, importance)
            })

        importances.sort(key=lambda x: x['importance'], reverse=True)

        if all(f['importance'] == 0 for f in importances):
            input_var = np.var(window, axis=0)
            importances = [
                {'feature': ALL_FEATS[i], 'importance': float(input_var[i])}
                for i in range(N_FEATS)
            ]
            importances.sort(key=lambda x: x['importance'], reverse=True)

        return importances[:5]


if __name__ == '__main__':
    import pandas as pd

    wrapper = ModelWrapper(
        weights_path='models/model_v2.weights.h5',
        scaler_path='models/minmax_scaler_v2.pkl'
    )

    df     = pd.read_csv('data/test.csv', nrows=10)
    window = wrapper.prepare_window_from_df(df)

    # Test basic prediction
    result = wrapper.predict_window(window)
    print('\nPrediction Result:')
    print(f"  Label      : {result['label']}")
    print(f"  Confidence : {result['confidence']*100:.2f}%")
    print(f"  All probs  : {result['probs']}")

    # Test prediction with subtype
    flow_dict = df.iloc[-1].to_dict()
    result_sub = wrapper.predict_window_with_subtype(window, flow_dict)
    print(f"\nSub-type       : {result_sub['subtype']}")

    # Test feature importance
    print('\nFeature Importance:')
    shap_result = wrapper.explain_window(window)
    for item in shap_result:
        print(f"  {item['feature']:<25} {item['importance']:.5f}")