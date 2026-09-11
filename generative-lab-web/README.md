# Latent Lab — Angular visualization

An English-language, static academic site explaining and comparing recorded PyTorch VAE/DCGAN experiments. No backend, Python execution, model inference, or live training.

```bash
# Node 22.12+ (22.x); see .nvmrc
npm ci
npm start
# http://localhost:4200
```

`npm run build` produces `dist/lab/browser`. Serve that directory using any static HTTP server. The included six-run dataset is in `public/data`. Do not use file:// URLs.

Use the Overview, Explore the Code, Training Replay, and Benchmark navigation tabs. Hover/focus/click the glossary terms, step through source blocks, scrub the epoch/time slider, change seeds, enlarge grids, inspect configurations, and download raw results.

```bash
npx playwright install chromium
npm run test:e2e
```

See [experiment documentation](../demos_generativos/README.md) for training, checkpoint recovery, metric definitions, provenance, and export instructions. All metrics and sample grids come from recorded experiments; missing artifacts display an explicit unavailable state.

To view an existing production build locally, run `npm run preview` and open http://127.0.0.1:4200. This serves only static files.
