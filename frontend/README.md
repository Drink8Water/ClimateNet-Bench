# ClimateNet-Bench frontend

Single-page academic results interface for the corrected ERA5-Land benchmark.

The page intentionally reads only the curated formal-result dataset in
`src/data/finalBenchmarkResults.js`. It does not display smoke, synthetic, or
historical `source_data_invalid` runs.

## Development

```bash
npm install
npm run dev
```

## Production build

```bash
npm run build
```
