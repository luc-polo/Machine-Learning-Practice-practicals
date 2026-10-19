from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
(models, LEVELS, HORIZONS, lat, lon, north, east, distance, nearby) = joblib.load(
    Path(__file__).with_name('model.joblib'))


def rain_features(history, issue):
    t, r, w, d, humid, cloud, _, snow = np.asarray(history).transpose(2, 0, 1)
    angle = np.deg2rad(d[:, -1])
    alignment = (east * np.sin(angle[:, None]) + north * np.cos(angle[:, None]))
    alignment /= np.maximum(distance, 1)
    upwind = np.exp(-distance / 500) * np.maximum(alignment, 0) ** 4
    upwind *= distance > 0
    upwind /= np.maximum(upwind.sum(axis=1, keepdims=True), 1e-8)
    season = 2 * np.pi * issue.dayofyear / 365.25
    solar = 2 * np.pi * (issue.hour + lon / 15) / 24
    columns = {}
    for name, values in zip(['temperature', 'rain', 'wind', 'humidity', 'clouds', 'snow'],
                            [t, r, w, humid, cloud, snow]):
        columns[name] = values[:, -1]
        for k in [3, 6, 24]:
            columns[f'{name}_mean{k}'] = values[:, -k:].mean(axis=1)
        columns[f'{name}_change6'] = values[:, -1] - values[:, -7]
    columns.update(rain_max6=r[:, -6:].max(axis=1), rain_wet24=(r[:, -24:] > 0).mean(axis=1),
                   east_wind=w[:, -1] * np.sin(angle), north_wind=w[:, -1] * np.cos(angle))
    for name, values in [('rain', r[:, -3:].mean(axis=1)), ('humidity', humid[:, -1]),
                         ('clouds', cloud[:, -1]), ('temperature', t[:, -1])]:
        columns['upwind_' + name] = upwind @ values
        columns['nearby_' + name] = nearby @ values
    columns.update(latitude=lat, longitude=lon, season_sin=np.sin(season),
                   season_cos=np.cos(season), solar_sin=np.sin(solar), solar_cos=np.cos(solar),
                   summer_solar=np.cos(season) * np.cos(solar),
                   humid_cloud=humid[:, -1] * cloud[:, -1] / 100,
                   humid_temp=humid[:, -1] * t[:, -1] / 100)
    return pd.DataFrame(columns).replace([np.inf, -np.inf], np.nan)


def risk(model, Z):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(Z)[:, 1]
    return model.decision_function(Z) if hasattr(model, 'decision_function') else model.predict(Z)


class Agent:
    def __init__(self):
        self.models = {}
        for h, (pre, wet_json, amount_json, cal, residuals) in models.items():
            wet, amount = XGBClassifier(), XGBRegressor()
            wet.load_model(wet_json)
            amount.load_model(amount_json)
            self.models[h] = pre, wet, amount, cal, residuals
    def members(self, F, horizons=HORIZONS):
        ensembles = []
        for h in horizons:
            pre, occurrence, amount, calibrator, residuals = self.models[h]
            Z = pre.transform(F)
            p = np.clip(calibrator.predict(risk(occurrence, Z)), .001, .999)
            q = (LEVELS[None, :] - (1 - p[:, None])) / p[:, None]
            residual = np.interp(np.maximum(q, 0), LEVELS, residuals)
            log_amount = amount.predict(Z)[:, None] + residual
            ensembles.append(np.where(q > 0, np.exp(np.clip(log_amount, -5, 5)), 0))
        return np.stack(ensembles, axis=1)
    def predict(self, request):
        F = rain_features(request['history'], pd.Timestamp(request['issue_time']))
        return {'rain': self.members(F, request['horizons']).tolist()}
