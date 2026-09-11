import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', timeout: 30000, workers: 1,
  use: {baseURL: 'http://127.0.0.1:4200', headless: true, screenshot: 'only-on-failure'},
  webServer: {command: 'node tests/serve.mjs', port: 4200, reuseExistingServer: true},
  reporter: 'list'
});
