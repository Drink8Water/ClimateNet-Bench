# Limitations

ClimateNet-Bench is a research benchmark. It has important limitations that users should understand before drawing scientific conclusions.

## Data Limitations

1. **Reanalysis data, not observations.** ERA5-Land is a model-based reanalysis product that assimilates observations into a physical model. It is not direct measurement. Variables like evaporation are model-derived, not observed.

2. **Spatial resolution.** ERA5-Land's 0.1° (~9 km) resolution may not capture fine-scale processes such as local convection, irrigation effects, or small water bodies.

3. **Temporal resolution.** Monthly aggregation masks sub-monthly dynamics (e.g., extreme rainfall events, heatwaves) that may be important for evaporation processes.

4. **Regional coverage.** The five benchmark regions do not cover all climate zones. Results may not transfer to polar, boreal, or small-island climates.

5. **Data volume.** The current benchmark uses 2019–2023 data (5 years). This is small for deep learning and may not capture decadal climate variability.

## Task Limitations

1. **Single target variable.** The benchmark only predicts evaporation anomaly. It does not jointly model the full surface energy and water balance.

2. **Fixed input window.** The 6-month window is a design choice. Different lead times or window lengths may yield different model rankings.

3. **No exogenous forecasts.** The benchmark assumes perfect knowledge of past climate variables. In a real forecasting setting, some inputs would themselves need to be predicted.

4. **No causal identification.** Predictive skill does not imply causal understanding. A model may perform well for the wrong reasons.

## Model Limitations

1. **Baseline models are simple.** The current model suite uses off-the-shelf ML models with default or lightly tuned hyperparameters. Results should not be interpreted as the best possible performance.

2. **TCN is a single architecture.** Many other deep learning architectures (LSTM, Transformer, Graph Neural Networks, Physics-Informed Neural Networks) are not included yet.

3. **No ensemble methods.** Individual models are evaluated separately. Ensemble or hybrid approaches may perform better.

4. **No online learning.** Models are trained once and evaluated on a fixed test set. Real operational systems would update continuously.

## Benchmark Limitations

1. **Not a physical model replacement.** ClimateNet-Bench evaluates statistical ML models. It does not replace, and should not be compared against, physics-based land surface or climate models (e.g., Noah-MP, CLM, ORCHIDEE) in terms of physical fidelity.

2. **Single metric cannot capture all aspects.** A model that wins on RMSE may be worse on physical consistency or spatial smoothness. Users should consult the full set of metrics.

3. **Synthetic data is for testing only.** The synthetic data generator produces simplified climate-like data. Any results from synthetic data are clearly labeled and should not be reported as benchmark results.

4. **No real-time or operational capability.** This is a research benchmark, not an operational forecasting system.

## Scope Boundaries

ClimateNet-Bench deliberately does **not** include:
- Real-time data ingestion pipelines
- Operational forecasting infrastructure
- Cloud deployment or auto-scaling
- User authentication or multi-tenancy
- Mobile or production-grade frontends
- Causal inference or attribution of climate change
- Policy recommendations or decision support

## When Not to Use ClimateNet-Bench

- For operational drought forecasting — use established monitoring systems.
- For climate change attribution — use dedicated detection and attribution methods.
- For hydrological forecasting — use catchment-scale hydrological models.
- For irrigation scheduling — use field-scale soil moisture monitoring.
- For comparisons against physical models — this benchmark evaluates statistical ML, not physical simulation.

## Future Work

See the GitHub Issues for planned improvements including:
- Additional deep learning architectures
- More benchmark regions and climate zones
- Longer temporal coverage
- Probabilistic forecasting metrics
- Feature attribution benchmarks
- Multi-task learning (joint prediction of multiple variables)
