from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from app.config import AI4I_CSV, MODEL_BUNDLE, MODEL_DIR, MODEL_METRICS
from app.errors import ModelNotReadyError
from app.schemas import EvidenceFeature, FailureModeScore, PredictionResponse, ProcessData

FEATURES = [
    'Type', 'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]
TARGET = 'Machine failure'
MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
MODE_NAMES = {
    'TWF': '공구 마모 고장',
    'HDF': '방열/열 방출 고장',
    'PWF': '전력/출력 조건 고장',
    'OSF': '과부하 고장',
    'RNF': '무작위 고장',
}
MODE_ACTIONS = {
    'TWF': ['공구 마모 상태 점검', '공구 교체 주기 검토'],
    'HDF': ['냉각/방열 상태 점검', '공기 온도와 공정 온도 조건 확인'],
    'PWF': ['회전수와 토크 조합 확인', '출력 조건과 부하 변동 점검'],
    'OSF': ['토크 부하 조건 점검', '공구 마모 상태와 과부하 운전 여부 확인'],
    'RNF': ['추가 데이터 확인', '반복 발생 여부 모니터링'],
}

@dataclass
class ModelBundle:
    failure_model: Pipeline
    mode_model: MultiOutputClassifier
    quantiles: dict[str, dict[str, float]]


def build_pipeline(n_estimators: int = 25) -> Pipeline:
    numeric = FEATURES[1:]
    categorical = ['Type']
    pre = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ])
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight='balanced_subsample',
        min_samples_leaf=2,
        n_jobs=1,
    )
    return Pipeline([('preprocess', pre), ('model', clf)])


class PredictionService:
    def __init__(self, csv_path: Path | None = None, model_path: Path | None = None):
        self.csv_path = csv_path or AI4I_CSV
        self.model_path = model_path or MODEL_BUNDLE
        self.bundle: ModelBundle | None = None

    def load_or_train(self) -> ModelBundle:
        if self.bundle is not None:
            return self.bundle
        if self.model_path.exists():
            try:
                self.bundle = joblib.load(self.model_path)
            except Exception as exc:
                if self.csv_path.exists():
                    self.bundle = self.train()
                    return self.bundle
                raise ModelNotReadyError('Prediction model bundle cannot be loaded and AI4I CSV is missing') from exc
            return self.bundle
        self.bundle = self.train()
        return self.bundle

    def train(self) -> ModelBundle:
        if not self.csv_path.exists():
            raise ModelNotReadyError('AI4I CSV is missing')
        df = pd.read_csv(self.csv_path)
        X = df[FEATURES]
        y = df[TARGET].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        validation_failure_model = build_pipeline()
        validation_failure_model.fit(X_train, y_train)
        y_pred = validation_failure_model.predict(X_test)
        y_proba = validation_failure_model.predict_proba(X_test)[:, list(validation_failure_model.named_steps['model'].classes_).index(1)]

        y_modes = df[MODES].astype(int)
        X_mode_train, X_mode_test, y_mode_train, y_mode_test = train_test_split(X, y_modes, test_size=0.2, random_state=42)
        validation_mode_model = MultiOutputClassifier(build_pipeline(n_estimators=15))
        validation_mode_model.fit(X_mode_train, y_mode_train)
        y_mode_pred = validation_mode_model.predict(X_mode_test)

        failure_model = build_pipeline()
        failure_model.fit(X, y)
        mode_model = MultiOutputClassifier(build_pipeline(n_estimators=15))
        mode_model.fit(X, y_modes)
        quantiles = {
            col: {str(q): float(df[col].quantile(q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
            for col in FEATURES[1:]
        }
        bundle = ModelBundle(failure_model=failure_model, mode_model=mode_model, quantiles=quantiles)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.model_path)
        metrics = {
            'trained_rows': len(df),
            'features': FEATURES,
            'modes': MODES,
            'validation': {
                'test_size': len(X_test),
                'failure_accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
                'failure_precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                'failure_recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                'failure_f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                'failure_roc_auc': round(float(roc_auc_score(y_test, y_proba)), 4),
                'mode_micro_f1': round(float(f1_score(y_mode_test, y_mode_pred, average='micro', zero_division=0)), 4),
                'mode_macro_f1': round(float(f1_score(y_mode_test, y_mode_pred, average='macro', zero_division=0)), 4),
            },
            'thresholds': {
                'failure_predicted': 0.5,
                'mode_predicted': 0.35,
                'risk_caution': 0.2,
                'risk_warning': 0.4,
                'risk_critical': 0.7,
            },
            'limitations': [
                'AI4I is a public synthetic/benchmark-style dataset and is not a plant-specific validated model.',
                'Predictions must be treated as decision-support signals, not maintenance or safety decisions.',
            ],
        }
        MODEL_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        self.bundle = bundle
        return bundle

    def _to_df(self, data: ProcessData) -> pd.DataFrame:
        return pd.DataFrame([{
            'Type': data.type,
            'Air temperature [K]': data.air_temperature_k,
            'Process temperature [K]': data.process_temperature_k,
            'Rotational speed [rpm]': data.rotational_speed_rpm,
            'Torque [Nm]': data.torque_nm,
            'Tool wear [min]': data.tool_wear_min,
        }])

    @staticmethod
    def _positive_probability(model, X: pd.DataFrame) -> float:
        classes = list(model.named_steps['model'].classes_)
        probas = model.predict_proba(X)[0]
        return float(probas[classes.index(1)]) if 1 in classes else 0.0

    def _evidence(self, data: ProcessData, quantiles: dict[str, dict[str, float]]) -> list[EvidenceFeature]:
        ev: list[EvidenceFeature] = []
        def q(col, level): return quantiles[col][level]
        if data.torque_nm >= q('Torque [Nm]', '0.75'):
            ev.append(EvidenceFeature(feature='Torque', direction='high', value=data.torque_nm, tag='torque_high', reason='토크가 학습 데이터 상위 25% 이상으로 부하가 큰 상태입니다.'))
        if data.tool_wear_min >= q('Tool wear [min]', '0.75'):
            ev.append(EvidenceFeature(feature='Tool wear', direction='high', value=data.tool_wear_min, tag='tool_wear_high', reason='공구 마모 시간이 학습 데이터 상위 25% 이상입니다.'))
        if data.air_temperature_k >= q('Air temperature [K]', '0.75'):
            ev.append(EvidenceFeature(feature='Air temperature', direction='high', value=data.air_temperature_k, tag='air_temperature_high', reason='공기 온도가 학습 데이터 상위 25% 이상입니다.'))
        if data.process_temperature_k >= q('Process temperature [K]', '0.75'):
            ev.append(EvidenceFeature(feature='Process temperature', direction='high', value=data.process_temperature_k, tag='process_temperature_high', reason='공정 온도가 학습 데이터 상위 25% 이상입니다.'))
        if data.rotational_speed_rpm <= q('Rotational speed [rpm]', '0.25'):
            ev.append(EvidenceFeature(feature='Rotational speed', direction='low', value=data.rotational_speed_rpm, tag='rpm_low_or_unstable', reason='회전수가 학습 데이터 하위 25% 이하입니다.'))
        elif data.rotational_speed_rpm >= q('Rotational speed [rpm]', '0.9'):
            ev.append(EvidenceFeature(feature='Rotational speed', direction='high', value=data.rotational_speed_rpm, tag='rpm_high', reason='회전수가 학습 데이터 상위 10% 이상입니다.'))
        return ev[:5]

    def _input_warnings(self, data: ProcessData, quantiles: dict[str, dict[str, float]]) -> list[str]:
        checks = [
            ('Air temperature [K]', data.air_temperature_k, '공기 온도'),
            ('Process temperature [K]', data.process_temperature_k, '공정 온도'),
            ('Rotational speed [rpm]', data.rotational_speed_rpm, '회전수'),
            ('Torque [Nm]', data.torque_nm, '토크'),
            ('Tool wear [min]', data.tool_wear_min, '공구 마모 시간'),
        ]
        warnings: list[str] = []
        for col, value, label in checks:
            low = quantiles[col]['0.1']
            high = quantiles[col]['0.9']
            if value < low or value > high:
                warnings.append(f'{label} 값이 학습 데이터 10~90% 범위 밖입니다. 예측 신뢰도를 보수적으로 해석하세요.')
        return warnings

    def recommended_actions(self, predicted_modes: list[str], evidence: list[EvidenceFeature]) -> list[str]:
        actions: list[str] = []
        for mode in predicted_modes:
            actions.extend(MODE_ACTIONS.get(mode, []))
        for e in evidence:
            if e.feature == 'Torque': actions.append('토크 부하 조건 점검')
            if e.feature == 'Tool wear': actions.append('공구 마모 상태 점검')
            if e.feature in ['Air temperature', 'Process temperature']: actions.append('냉각/방열 상태 점검')
            if e.feature == 'Rotational speed': actions.append('회전수와 부하 조건 점검')
        if not actions:
            actions.append('정기 점검 유지 및 추가 공정 데이터 확인')
        return list(dict.fromkeys(actions))[:7]

    def predict(self, data: ProcessData) -> PredictionResponse:
        bundle = self.load_or_train()
        X = self._to_df(data)
        failure_probability = self._positive_probability(bundle.failure_model, X)
        predicted_failure = failure_probability >= 0.5
        mode_scores: list[FailureModeScore] = []
        predicted_modes: list[str] = []
        mode_prob_lists = bundle.mode_model.predict_proba(X)
        for mode, probs in zip(MODES, mode_prob_lists):
            classes = list(bundle.mode_model.estimators_[MODES.index(mode)].named_steps['model'].classes_)
            prob = float(probs[0][classes.index(1)]) if 1 in classes else 0.0
            pred = prob >= 0.35
            mode_scores.append(FailureModeScore(code=mode, name=MODE_NAMES[mode], probability=round(prob, 4), predicted=pred))
            if pred: predicted_modes.append(mode)
        if not predicted_modes and (predicted_failure or failure_probability >= 0.35):
            predicted_modes.append(max(mode_scores, key=lambda x: x.probability).code)
        evidence = self._evidence(data, bundle.quantiles)
        input_warnings = self._input_warnings(data, bundle.quantiles)
        if failure_probability >= 0.7:
            risk = 'Critical'
        elif failure_probability >= 0.4 or predicted_failure:
            risk = 'Warning'
        elif failure_probability >= 0.2 or len(evidence) >= 3:
            risk = 'Caution'
        else:
            risk = 'Normal'
        return PredictionResponse(
            failure_probability=round(failure_probability, 4),
            predicted_failure=predicted_failure,
            risk_level=risk,
            failure_modes=mode_scores,
            predicted_modes=predicted_modes,
            evidence_features=evidence,
            recommended_actions=self.recommended_actions(predicted_modes, evidence),
            input_warnings=input_warnings,
            model_source='AI4I RandomForest model',
            disclaimer='이 결과는 공개 AI4I 데이터 기반 예측/설명용이며 실제 설비 제어 또는 법적 안전 판단을 대체하지 않습니다.',
        )
