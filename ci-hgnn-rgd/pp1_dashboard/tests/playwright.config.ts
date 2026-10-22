import { defineConfig } from '../frontend/node_modules/@playwright/test';
import path from 'node:path';

const root = path.resolve(__dirname, '..');
export default defineConfig({
  testDir: __dirname,
  testMatch: 'dashboard.spec.ts',
  outputDir: path.join(root, 'reports/browser-results'),
  reporter: [['list'], ['json', { outputFile: path.join(root, 'reports/browser-tests.json') }]],
  timeout: 90_000,
  expect: { timeout: 30_000 },
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:5173', channel: 'msedge', headless: true, screenshot: 'only-on-failure', trace: 'retain-on-failure' },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1440, height: 1000 } } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
  ],
  webServer: [
    { command: '".venv\\Scripts\\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000', cwd: path.join(root, 'backend'), url: 'http://127.0.0.1:8000/api/health', reuseExistingServer: false, timeout: 90_000 },
    { command: 'npm.cmd run dev', cwd: path.join(root, 'frontend'), url: 'http://127.0.0.1:5173', reuseExistingServer: false, timeout: 90_000 },
  ],
});
